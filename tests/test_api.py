"""Тесты API без Docker: эндпоинты, проверка входа, ошибки, кэш, подмена модели, документация, /explain, reload.

Запуск: pytest tests/test_api.py, в образе: make test.
"""
import io
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from conftest import ROOT

WEEKS = [str(d.date()) for d in pd.date_range("2026-07-27", periods=9, freq="7D")]


def client_for(tmp_path, **settings):
    return TestClient(create_app(Settings(model_dir=str(ROOT / "model"), cache_dir=str(tmp_path / "cache"), **settings)))


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    with client_for(tmp_path_factory.mktemp("api")) as c:
        yield c


@pytest.fixture(scope="module")
def posted(api, sample):
    r = api.post("/predict", files=files(*sample))
    assert r.status_code == 200, r.text
    return r


def files(sales: bytes, stock: bytes):
    return {"sales": ("sales.csv", sales, "text/csv"), "stock": ("stock.csv", stock, "text/csv")}


def edit(raw: bytes, fn) -> bytes:
    df = fn(pd.read_csv(io.BytesIO(raw), dtype={"date": str}))
    return df.to_csv(index=False).encode()


def csv(r) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(r.content), dtype={"week": str})


def test_health_ready_info(api):
    assert api.get("/health").json() == {"status": "ok"}
    ready = api.get("/ready").json()
    assert ready["status"] == "ready" and ready["model_kind"] == "two_stage"
    info = api.get("/model/info").json()
    assert info["version"] == ready["model_version"]
    assert info["train_cutoff"] == "2026-07-26" and len(info["ensemble"]) == 3


def test_predict_format(posted, sample):
    df = csv(posted)
    assert list(df.columns) == ["week", "warehouse_name", "article", "quantity"]
    sales = pd.read_csv(io.BytesIO(sample[0]))
    pairs = sales.warehouse_name.nunique() * sales.article.nunique()
    assert len(df) == pairs * 9 and not df.duplicated(["week", "warehouse_name", "article"]).any()
    assert sorted(df.week.unique()) == WEEKS
    assert (df.quantity >= 0).all() and (df.quantity % 1 == 0).all()
    assert posted.headers["content-type"].startswith("text/csv")
    assert posted.headers["x-cache"] == "miss" and len(posted.headers["x-forecast-id"]) == 16
    assert "predict;dur=" in posted.headers["server-timing"]


def test_predict_repeat_is_cached_and_identical(api, posted, sample):
    again = api.post("/predict", files=files(*sample))
    assert again.status_code == 200 and again.headers["x-cache"] == "hit"
    assert again.content == posted.content and again.headers["x-forecast-id"] == posted.headers["x-forecast-id"]


def test_get_filters(api, posted):
    full = csv(posted)
    article, warehouse = full.article.iloc[0], full.warehouse_name.iloc[0]
    by_article = csv(api.get("/predict", params={"article": article}))
    assert set(by_article.article) == {article} and len(by_article) == full.warehouse_name.nunique() * 9
    by_wh = csv(api.get("/predict", params={"warehouse": warehouse}))
    assert set(by_wh.warehouse_name) == {warehouse} and len(by_wh) == full.article.nunique() * 9
    pair = csv(api.get("/predict", params={"article": article, "warehouse": warehouse}))
    assert len(pair) == 9 and sorted(pair.week) == WEEKS
    by_id = api.get("/predict", params={"article": article, "forecast_id": posted.headers["x-forecast-id"]})
    assert by_id.status_code == 200 and by_id.headers["x-forecast-id"] == posted.headers["x-forecast-id"]


def test_get_errors(api, posted):
    r = api.get("/predict")
    assert r.status_code == 400 and r.json()["code"] == "BAD_REQUEST" and "detail" in r.json()
    r = api.get("/predict", params={"article": "char_article_no_such"})
    assert r.status_code == 404 and r.json()["code"] == "NOT_FOUND"
    r = api.get("/predict", params={"article": "char_article_0", "forecast_id": "0123456789abcdef"})
    assert r.status_code == 404 and r.json()["code"] == "NO_FORECAST"


def test_predict_all(api, posted):
    as_csv = api.get("/predict/all", params={"format": "csv"})
    assert as_csv.status_code == 200 and as_csv.content == posted.content
    as_json = api.get("/predict/all", params={"format": "json"})
    rows = json.loads(as_json.content)
    assert len(rows) == len(csv(posted)) and set(rows[0]) == {"week", "warehouse_name", "article", "quantity"}
    assert api.get("/predict/all").content == posted.content
    r = api.get("/predict/all", params={"format": "xml"})
    assert r.status_code == 400 and r.json()["code"] == "BAD_REQUEST"


@pytest.mark.parametrize("name, change, status, code", [
    ("нет колонки", lambda s, st: (edit(s, lambda d: d.drop(columns="quantity")), st), 422, "VALIDATION_ERROR"),
    ("плохая дата", lambda s, st: (edit(s, lambda d: d.assign(date=d.date.str.replace("-", "."))), st), 422, "VALIDATION_ERROR"),
    ("меньше 180 дней", lambda s, st: (edit(s, lambda d: d[d.date >= "2026-05-01"]),
                                       edit(st, lambda d: d[d.date >= "2026-05-01"])), 422, "VALIDATION_ERROR"),
    ("отрицательные продажи", lambda s, st: (edit(s, lambda d: d.assign(quantity=-d.quantity)), st), 422, "VALIDATION_ERROR"),
    ("пустой файл", lambda s, st: (b"", st), 400, "EMPTY_FILE"),
    ("только заголовок", lambda s, st: (s.split(b"\n")[0] + b"\n", st), 400, "EMPTY_FILE"),
])
def test_validation(api, sample, name, change, status, code):
    r = api.post("/predict", files=files(*change(*sample)))
    assert r.status_code == status, (name, r.text)
    body = r.json()
    assert body["code"] == code and isinstance(body["detail"], str) and body["detail"]


def test_missing_file_field(api, sample):
    r = api.post("/predict", files={"sales": ("sales.csv", sample[0], "text/csv")})
    assert r.status_code == 422 and r.json()["code"] == "VALIDATION_ERROR" and "stock" in r.json()["detail"]


def test_accepted_variants(api, sample):
    # отрицательный остаток допустим; товар в stock может называться char_article (ТЗ) или article (выданные файлы)
    negative = edit(sample[1], lambda d: d.assign(stock=d.stock.where(d.index % 50 != 0, -3)))
    assert api.post("/predict", files=files(sample[0], negative)).status_code == 200
    renamed = edit(sample[1], lambda d: d.rename(columns={"char_article": "article"}))
    r = api.post("/predict", files=files(sample[0], renamed))
    assert r.status_code == 200 and r.headers["x-cache"] == "miss"


def test_explain(api, posted):
    df = csv(posted)
    sold = df[df.quantity > 0].iloc[0]
    r = api.get("/explain", params={"article": sold.article, "warehouse": sold.warehouse_name})
    assert r.status_code == 200, r.text
    e = r.json()
    assert len(e["weeks"]) == 9 and len(e["classifier_top_features"]) == 5 and len(e["regressor_top_features"]) == 5
    week = next(w for w in e["weeks"] if w["week"] == sold.week)
    assert week["quantity"] == sold.quantity and week["sale"] == (sold.quantity > 0)
    assert api.get("/explain", params={"article": "x", "warehouse": "y"}).status_code == 404


def test_baseline_model(tmp_path, sample):
    model = tmp_path / "baseline"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"kind": "baseline", "window": 8, "zero_below": 1}))
    with TestClient(create_app(Settings(model_dir=str(model), cache_dir=str(tmp_path / "cache")))) as c:
        info = c.get("/model/info").json()
        assert info["kind"] == "baseline"
        r = c.post("/predict", files=files(*sample))
        assert r.status_code == 200 and sorted(csv(r).week.unique()) == WEEKS
        assert r.headers["x-model-version"] == info["version"]
        df = csv(r)
        assert c.get("/explain", params={"article": df.article[0], "warehouse": df.warehouse_name[0]}).status_code == 501


def test_docs_on(api):
    for path in ("/", "/docs", "/redoc", "/openapi.json", "/solution"):
        assert api.get(path).status_code == 200, path
    assert "/static/swagger-ui-bundle.js" in api.get("/docs").text
    assert "/static/redoc.standalone.js" in api.get("/redoc").text
    assert "/predict" in api.get("/openapi.json").json()["paths"]


def test_docs_off(tmp_path):
    with client_for(tmp_path, docs=False) as c:
        for path in ("/docs", "/redoc", "/openapi.json", "/solution"):
            assert c.get(path).status_code == 404, path
        assert c.get("/").json()["docs"].startswith("выключена")
        assert c.get("/health").status_code == 200


def test_reload(tmp_path):
    with client_for(tmp_path) as c:
        assert c.post("/model/reload").status_code == 404
    with client_for(tmp_path, admin_token="secret") as c:
        assert c.post("/model/reload", headers={"X-Admin-Token": "wrong"}).status_code == 403
        r = c.post("/model/reload", headers={"X-Admin-Token": "secret"})
        assert r.status_code == 200 and r.json()["model_version"] == r.json()["previous_version"]
