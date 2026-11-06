"""Сервис прогноза: модель, хранилище и ограничение числа одновременных расчётов."""
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .predictor import load_model
from .store import ForecastStore, create_store
from .validators import read_csv, validate

log = logging.getLogger("forecast")


class NotFound(Exception):
    def __init__(self, message, code="NOT_FOUND"):
        super().__init__(message)
        self.code = code


class Conflict(Exception):
    pass


@dataclass
class ForecastResult:
    forecast_id: str
    csv: bytes
    cache_hit: bool
    meta: dict
    timings: dict = field(default_factory=dict)


class Timer:
    def __init__(self):
        self.start = self.last = time.perf_counter()
        self.timings = {}

    def mark(self, stage):
        now = time.perf_counter()
        self.timings[stage] = round((now - self.last) * 1000, 1)
        self.last = now

    def total(self):
        self.timings["total"] = round((time.perf_counter() - self.start) * 1000, 1)
        return self.timings


class ForecastService:
    def __init__(self, model_dir, cache_dir, keep=20, max_concurrent=1, storage_url=""):
        self.model_dir = Path(model_dir)
        self.model = load_model(self.model_dir)
        self.store: ForecastStore = create_store(storage_url, cache_dir, keep)
        self._slots = threading.BoundedSemaphore(max_concurrent)

    def reload(self) -> str:
        self.model = load_model(self.model_dir)
        log.info("модель перезагружена: %s %s", self.model.kind, self.model.version)
        return self.model.version

    def forecast(self, sales_raw: bytes, stock_raw: bytes) -> ForecastResult:
        timer, model = Timer(), self.model
        forecast_id = self.store.key(model.version, sales_raw, stock_raw)
        cached = self.store.csv(forecast_id)
        if cached is not None:
            self.store.set_latest(forecast_id)
            return ForecastResult(forecast_id, cached, True, self.store.meta(forecast_id) or {}, timer.total())
        sales, stock = read_csv(sales_raw, "sales"), read_csv(stock_raw, "stock")
        timer.mark("parse")
        validate(sales, stock)
        timer.mark("validate")
        with self._slots:
            timer.mark("queue")
            prediction = model.predict(sales, stock)
        timer.mark("predict")
        forecast = prediction.forecast
        csv = forecast.to_csv(index=False, date_format="%Y-%m-%d").encode()
        meta = {"forecast_id": forecast_id, "model_kind": model.kind, "model_version": model.version,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": len(forecast),
                "pairs": int(forecast[["warehouse_name", "article"]].drop_duplicates().shape[0]),
                "first_week": str(forecast.week.min().date()), "last_week": str(forecast.week.max().date()),
                "last_sales_date": str(sales["date"].max()), "units": float(forecast.quantity.sum())}
        self.store.put(forecast_id, csv, {**meta, "timings_ms": timer.total()}, prediction.details)
        log.info("прогноз %s: %s строк, %.1f с", forecast_id, len(forecast), timer.timings["total"] / 1000)
        return ForecastResult(forecast_id, csv, False, meta, timer.timings)

    def _resolve(self, forecast_id):
        forecast_id = forecast_id or self.store.latest()
        if forecast_id is None:
            raise NotFound("Прогноза ещё нет: сначала POST /predict с файлами sales и stock", "NO_FORECAST")
        if not self.store.exists(forecast_id):
            raise NotFound(f"Нет прогноза {forecast_id} (возможно, удалён как старый)", "NO_FORECAST")
        return forecast_id

    def csv(self, forecast_id=None):
        forecast_id = self._resolve(forecast_id)
        data = self.store.csv(forecast_id)
        if data is None:
            raise NotFound(f"Нет прогноза {forecast_id} (возможно, удалён как старый)", "NO_FORECAST")
        return forecast_id, data

    def frame(self, forecast_id=None):
        forecast_id = self._resolve(forecast_id)
        return forecast_id, self.store.frame(forecast_id)

    def explain(self, article, warehouse, forecast_id=None):
        forecast_id, model = self._resolve(forecast_id), self.model
        meta = self.store.meta(forecast_id) or {}
        if meta.get("model_version") != model.version:
            raise Conflict(f"Прогноз {forecast_id} сделан моделью {meta.get('model_version')}, а сейчас загружена "
                           f"{model.version}: повторите POST /predict")
        details = self.store.details(forecast_id)
        if details is None:
            raise NotImplementedError("Для этого прогноза нет данных для объяснения")
        try:
            return {"forecast_id": forecast_id, **model.explain(details, article, warehouse)}
        except KeyError:
            raise NotFound(f"В прогнозе {forecast_id} нет пары {warehouse} × {article}")
