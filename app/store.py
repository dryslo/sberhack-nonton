"""Общий адаптер хранилища прогнозов и реализации для диска и Redis."""
from abc import ABC, abstractmethod
import gzip
import hashlib
import io
import json
import os
import pickle
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

import pandas as pd


class ForecastStore(ABC):
    @staticmethod
    def key(model_version: str, sales: bytes, stock: bytes) -> str:
        h = hashlib.sha256(model_version.encode())
        for part in (sales, stock):
            h.update(len(part).to_bytes(8, "little"))
            h.update(part)
        return h.hexdigest()[:16]

    @abstractmethod
    def put(self, forecast_id: str, csv: bytes, meta: dict, details=None) -> None:
        pass

    @abstractmethod
    def set_latest(self, forecast_id: str) -> None:
        pass

    @abstractmethod
    def latest(self) -> str | None:
        pass

    @abstractmethod
    def exists(self, forecast_id: str) -> bool:
        pass

    @abstractmethod
    def csv(self, forecast_id: str) -> bytes | None:
        pass

    def frame(self, forecast_id: str) -> pd.DataFrame | None:
        data = self.csv(forecast_id)
        return pd.read_csv(io.BytesIO(data), dtype={"week": str, "warehouse_name": str, "article": str}) if data is not None else None

    @abstractmethod
    def meta(self, forecast_id: str) -> dict | None:
        pass

    @abstractmethod
    def details(self, forecast_id: str):
        pass

    def close(self) -> None:
        pass


class LocalForecastStore(ForecastStore):
    def __init__(self, root, keep=20):
        self.root = Path(root)
        self.dir = self.root / "forecasts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep = keep

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

    def exists(self, forecast_id: str) -> bool:
        return self.csv_path(forecast_id) is not None

    def csv(self, forecast_id: str) -> bytes | None:
        path = self.csv_path(forecast_id)
        return path.read_bytes() if path else None

    def frame(self, forecast_id: str) -> pd.DataFrame | None:
        path = self.csv_path(forecast_id)
        return pd.read_csv(path, dtype={"week": str, "warehouse_name": str, "article": str}) if path else None

    def meta(self, forecast_id: str) -> dict | None:
        path = self.dir / f"{forecast_id}.json"
        return json.loads(path.read_bytes()) if path.exists() else None

    def details(self, forecast_id: str):
        path = self.dir / f"{forecast_id}.pkl.gz"
        return pickle.loads(gzip.decompress(path.read_bytes())) if path.exists() else None

    def _cleanup(self) -> None:
        forecasts = sorted(self.dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in forecasts[self.keep:]:
            stem = old.name[:-len(".csv")]
            for ext in (".csv", ".json", ".pkl.gz"):
                (self.dir / f"{stem}{ext}").unlink(missing_ok=True)


class RedisForecastStore(ForecastStore):
    PREFIX = "forecast-store:"

    def __init__(self, url: str, keep=20):
        from redis import Redis

        self.client = Redis.from_url(url, decode_responses=False, socket_connect_timeout=5, socket_timeout=5)
        self.keep = keep
        self.index = self.PREFIX + "index"
        self.latest_key = self.PREFIX + "latest"

    def _record(self, forecast_id: str) -> str:
        return self.PREFIX + "forecast:" + forecast_id

    def put(self, forecast_id: str, csv: bytes, meta: dict, details=None) -> None:
        record = {"csv": csv, "meta": json.dumps(meta, ensure_ascii=False, default=str).encode()}
        if details is not None:
            record["details"] = gzip.compress(pickle.dumps(details), compresslevel=3)

        def save(pipe):
            previous = [item.decode() for item in pipe.lrange(self.index, 0, -1)]
            ordered = [forecast_id] + [item for item in previous if item != forecast_id]
            pipe.multi()
            # Запись, указатель и удаление старых прогнозов видны всем воркерам атомарно.
            pipe.delete(self._record(forecast_id))
            pipe.hset(self._record(forecast_id), mapping=record)
            pipe.set(self.latest_key, forecast_id)
            pipe.delete(self.index)
            pipe.rpush(self.index, *ordered[:self.keep])
            for old in ordered[self.keep:]:
                pipe.delete(self._record(old))

        self.client.transaction(save, self.index)

    def set_latest(self, forecast_id: str) -> None:
        self.client.set(self.latest_key, forecast_id)

    def latest(self) -> str | None:
        value = self.client.get(self.latest_key)
        return value.decode() if value is not None else None

    def exists(self, forecast_id: str) -> bool:
        return bool(self.client.hexists(self._record(forecast_id), "csv"))

    def csv(self, forecast_id: str) -> bytes | None:
        return self.client.hget(self._record(forecast_id), "csv")

    def meta(self, forecast_id: str) -> dict | None:
        data = self.client.hget(self._record(forecast_id), "meta")
        return json.loads(data) if data is not None else None

    def details(self, forecast_id: str):
        data = self.client.hget(self._record(forecast_id), "details")
        return pickle.loads(gzip.decompress(data)) if data is not None else None

    def close(self) -> None:
        self.client.close()


def create_store(url: str, cache_dir, keep=20) -> ForecastStore:
    if keep < 1:
        raise ValueError("MAX_FORECASTS должен быть не меньше 1")
    if not url:
        return LocalForecastStore(cache_dir, keep)
    parsed = urlsplit(url)
    if parsed.scheme in ("redis", "rediss"):
        return RedisForecastStore(url, keep)
    if parsed.scheme == "file" and parsed.netloc in ("", "localhost") and parsed.path:
        return LocalForecastStore(url2pathname(parsed.path), keep)
    raise ValueError("STORAGE_URL должен иметь вид file:///cache, redis://host:6379/0 или rediss://host:6379/0")
