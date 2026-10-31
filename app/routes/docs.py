"""Документация и стартовая страница (флаг DOCS). Swagger UI и ReDoc лежат в app/static, интернет не нужен."""
import html
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse

from ..service import NotFound

STATIC = Path(__file__).resolve().parent.parent / "static"
router = APIRouter(include_in_schema=False)
plain = APIRouter(include_in_schema=False)  # при DOCS=0 вместо router


@router.get("/docs")
async def swagger(request: Request):
    app = request.app
    local = {"swagger_js_url": "/static/swagger-ui-bundle.js", "swagger_css_url": "/static/swagger-ui.css"} \
        if (STATIC / "swagger-ui-bundle.js").exists() else {}
    return get_swagger_ui_html(openapi_url=app.openapi_url, title=f"{app.title} - API", swagger_favicon_url="", **local)


@router.get("/redoc")
async def redoc(request: Request):
    app = request.app
    local = {"redoc_js_url": "/static/redoc.standalone.js"} if (STATIC / "redoc.standalone.js").exists() else {}
    return get_redoc_html(openapi_url=app.openapi_url, title=f"{app.title} - API", redoc_favicon_url="",
                          with_google_fonts=False, **local)


@router.get("/solution")
async def solution():
    page = STATIC / "solution.html"
    if not page.exists():
        raise NotFound("Нет app/static/solution.html (make solution-html)")
    return FileResponse(page, media_type="text/html")


@router.get("/")
async def index(request: Request):
    return HTMLResponse(landing(getattr(request.app.state, "service", None)))


@plain.get("/")
async def index_plain(request: Request):
    return {"service": request.app.title, "docs": "выключена (DOCS=0)", "endpoints": ["/predict", "/predict/all", "/health"]}


def landing(svc) -> str:
    if svc is None:
        model = "<p class=warn>Модель не загружена, см. <a href=/ready>/ready</a>.</p>"
    else:
        info = svc.model.info()
        score = info.get("backtest_score") or {}
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:.4f}</td></tr>" for k, v in score.items())
        model = (f"<p>Модель <b>{html.escape(info['kind'])}</b>, версия <code>{html.escape(info['version'])}</code>"
                 + (f", обучена на данных по {html.escape(str(info['train_cutoff']))}" if info.get("train_cutoff") else "")
                 + f".</p><p>{html.escape(info.get('description', ''))}</p>"
                 + (f"<p>Бэктест (score: 0 значит идеально, 1 значит не лучше прогноза \"как на прошлой неделе\"):</p><table>{rows}</table>" if rows else ""))
    return f"""<!doctype html><html lang=ru><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Прогноз продаж - API</title><style>
body{{font:15px/1.55 system-ui,sans-serif;max-width:780px;margin:40px auto;padding:0 16px;color:#1d1d1b;background:#fcfcfb}}
h1{{font-size:22px;margin-bottom:4px}} a{{color:#2a78d6}} code,pre{{background:#f0efec;border-radius:4px;padding:1px 5px}}
pre{{padding:10px 12px;overflow-x:auto}} table{{border-collapse:collapse}} td{{padding:3px 14px 3px 0}} .warn{{color:#b3261e}}
ul{{padding-left:20px}} .muted{{color:#52514e}}</style></head><body>
<h1>Прогноз продаж: склад × товар</h1><p class=muted>Прогноз на 9 недель вперёд по каждой паре.</p>
{model}
<h2>Документация</h2><ul>
<li><a href=/docs>/docs</a>: документация API (Swagger UI), запросы можно отправлять из браузера</li>
<li><a href=/redoc>/redoc</a>: та же документация в формате ReDoc</li>
<li><a href=/solution>/solution</a>: обзор решения</li>
<li><a href=/model/info>/model/info</a>: описание модели; <a href=/health>/health</a>, <a href=/ready>/ready</a>: проверки</li></ul>
<h2>Примеры</h2>
<pre>curl -X POST http://localhost:8000/predict -F "sales=@sales.csv" -F "stock=@stock.csv" -o forecast.csv
curl "http://localhost:8000/predict?article=char_article_1&amp;warehouse=warehouse_name_3"
curl "http://localhost:8000/predict/all?format=csv" -o full_forecast.csv</pre>
</body></html>"""
