"""Контракт локального и Redis-хранилища, выбор по URL и API с удалённым хранилищем."""
from concurrent.futures import ThreadPoolExecutor
import os

import fakeredis
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.store import ForecastStore, LocalForecastStore, RedisForecastStore, create_store
from conftest import ROOT

CSV = b"week,warehouse_name,article,quantity\n2026-07-27,001,002,3\n"
META = {"warehouse": "Склад", "rows": 1}


@pytest.fixture
def redis_server(monkeypatch):
    server = fakeredis.FakeServer()
    monkeypatch.setattr("redis.Redis.from_url", lambda url, **kwargs: fakeredis.FakeRedis(server=server))
    return server


@pytest.fixture(params=["local", "redis"])
def stores(request, tmp_path, redis_server):
    def new():
        if request.param == "redis":
            return RedisForecastStore("redis://test:6379/0", keep=2)
        return LocalForecastStore(tmp_path, keep=2)

    first, second = new(), new()
    yield first, second
    first.close()
    second.close()


def test_storage_contract(stores):
    writer, reader = stores
    assert reader.latest() is None
    assert not reader.exists("missing")
    for method in (reader.csv, reader.frame, reader.meta, reader.details):
        assert method("missing") is None
    details = {"rows": pd.DataFrame({"article": ["002"], "quantity": [3]})}
    writer.put("first", CSV, META, details)
    assert reader.latest() == "first"
    assert reader.exists("first")
    assert reader.csv("first") == CSV
    assert reader.meta("first") == META
    assert reader.frame("first").iloc[0].to_dict() == {
        "week": "2026-07-27", "warehouse_name": "001", "article": "002", "quantity": 3,
    }
    pd.testing.assert_frame_equal(reader.details("first")["rows"], details["rows"])
    writer.put("second", CSV, META)
    assert reader.latest() == "second"
    reader.set_latest("first")
    assert writer.latest() == "first"


def test_retention_removes_complete_forecast(stores):
    writer, reader = stores
    writer.put("first", CSV, META, {"data": [1]})
    if isinstance(writer, LocalForecastStore):
        # Явный порядок без зависимости от разрешения часов файловой системы.
        os.utime(writer.csv_path("first"), (1, 1))
    writer.put("second", CSV, META)
    writer.put("third", CSV, META)
    assert reader.latest() == "third"
    assert not reader.exists("first")
    assert reader.csv("first") is reader.meta("first") is reader.details("first") is None
    assert reader.exists("second") and reader.exists("third")


def test_store_selection(tmp_path, monkeypatch, redis_server):
    monkeypatch.setenv("STORAGE_URL", "redis://test:6379/0")
    assert Settings.from_env().storage_url == "redis://test:6379/0"
    assert isinstance(create_store("", tmp_path), LocalForecastStore)
    root = tmp_path / "cache with spaces"
    assert create_store(root.as_uri(), tmp_path).root == root
    assert isinstance(create_store("redis://test:6379/0", tmp_path), RedisForecastStore)
    assert isinstance(create_store("rediss://test:6379/0", tmp_path), RedisForecastStore)
    with pytest.raises(ValueError, match="STORAGE_URL"):
        create_store("https://test", tmp_path)
    with pytest.raises(ValueError, match="MAX_FORECASTS"):
        create_store("", tmp_path, keep=0)
    with pytest.raises(TypeError):
        ForecastStore()


def test_redis_concurrent_writers(redis_server):
    def write(index):
        store = RedisForecastStore("redis://test:6379/0", keep=3)
        try:
            store.put(str(index), CSV, {"index": index}, {"index": index})
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(12)))
    store = RedisForecastStore("redis://test:6379/0", keep=3)
    retained = [str(i) for i in range(12) if store.exists(str(i))]
    assert len(retained) == 3
    assert store.latest() in retained
    for item in retained:
        assert store.csv(item) == CSV
        assert store.meta(item) == store.details(item) == {"index": int(item)}
    store.close()


def test_redis_api_shared_across_restarts(tmp_path, redis_server, sample):
    settings = Settings(model_dir=str(ROOT / "model"), cache_dir=str(tmp_path / "unused"),
                        storage_url="redis://test:6379/0")
    files = {"sales": ("sales.csv", sample[0], "text/csv"), "stock": ("stock.csv", sample[1], "text/csv")}
    with TestClient(create_app(settings)) as client:
        assert client.get("/predict/all").status_code == 404
        posted = client.post("/predict", files=files)
        assert posted.status_code == 200, posted.text
        assert posted.headers["x-cache"] == "miss"
        forecast_id = posted.headers["x-forecast-id"]
    with TestClient(create_app(settings)) as client:
        repeated = client.post("/predict", files=files)
        assert repeated.status_code == 200 and repeated.headers["x-cache"] == "hit"
        assert repeated.content == posted.content
        full = client.get("/predict/all", params={"forecast_id": forecast_id})
        assert full.status_code == 200 and full.content == posted.content
        assert full.headers["x-forecast-id"] == forecast_id
        rows = client.get("/predict/all", params={"format": "json"}).json()
        row = rows[0]
        params = {"article": row["article"], "warehouse": row["warehouse_name"]}
        assert client.get("/predict", params=params).status_code == 200
        explanation = client.get("/explain", params=params)
        assert explanation.status_code == 200, explanation.text
        assert explanation.json()["forecast_id"] == forecast_id
        assert client.get("/predict/all", params={"forecast_id": "missing"}).status_code == 404
    assert not (tmp_path / "unused").exists()
