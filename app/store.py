"""Хранилище прогнозов в CACHE_DIR. Ключ - хеш версии модели и присланных файлов, поэтому повторный запрос
с теми же данными отдаётся без пересчёта, а воркеры не хранят состояние в памяти."""
import gzip
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import pandas as pd


class ForecastStore:
    def __init__(self, root, keep=20):
        self.root = Path(root)
        self.dir = self.root / "forecasts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep = keep

    @staticmethod
    def key(model_version: str, sales: bytes, stock: bytes) -> str:
        h = hashlib.sha256(model_version.encode())
        for part in (sales, stock):
            h.update(len(part).to_bytes(8, "little"))
            h.update(part)
        return h.hexdigest()[:16]

    def _write(self, path: Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)

    def put(self, forecast_id: str, csv: bytes, meta: dict, details=None) -> None:
        if details is not None:
            self._write(self.dir / f"{forecast_id}.pkl.gz", gzip.compress(pickle.dumps(details), compresslevel=3))
        self._write(self.dir / f"{forecast_id}.json", json.dumps(meta, ensure_ascii=False, default=str).encode())
        # CSV пишем последним: если он есть, прогноз записан целиком
        self._write(self.dir / f"{forecast_id}.csv", csv)
        self.set_latest(forecast_id)
        self._cleanup()

    def set_latest(self, forecast_id: str) -> None:
        self._write(self.root / "latest", forecast_id.encode())

    def latest(self) -> str | None:
        path = self.root / "latest"
        return path.read_text().strip() if path.exists() else None

    def csv_path(self, forecast_id: str) -> Path | None:
        path = self.dir / f"{forecast_id}.csv"
        return path if path.exists() else None

    def csv(self, forecast_id: str) -> bytes | None:
        path = self.csv_path(forecast_id)
        return path.read_bytes() if path else None

    def frame(self, forecast_id: str) -> pd.DataFrame | None:
        path = self.csv_path(forecast_id)
        return pd.read_csv(path, dtype={"week": str, "warehouse_name": str, "article": str}) if path else None

    def meta(self, forecast_id: str) -> dict | None:
        path = self.dir / f"{forecast_id}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def details(self, forecast_id: str):
        path = self.dir / f"{forecast_id}.pkl.gz"
        return pickle.loads(gzip.decompress(path.read_bytes())) if path.exists() else None

    def _cleanup(self) -> None:
        forecasts = sorted(self.dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in forecasts[self.keep:]:
            stem = old.name[:-len(".csv")]
            for ext in (".csv", ".json", ".pkl.gz"):
                (self.dir / f"{stem}{ext}").unlink(missing_ok=True)
