"""HTTP API прогноза продаж (раздел 10 ТЗ): приложение собирается из настроек, маршрутов и обработчиков ошибок."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .errors import add_handlers
from .routes import docs, predict, system
from .service import ForecastService

log = logging.getLogger("forecast")

DESCRIPTION = """Прогноз продаж на 9 недель по каждой паре склад × товар.

1. **POST /predict**: загрузить `sales` и `stock` за последние 180+ дней (CSV в формате организаторов), в ответ CSV
   прогноза с понедельника недели, следующей за последней датой в `sales`.
2. **GET /predict**: прогноз по товару и/или складу; **GET /predict/all**: весь прогноз (CSV или JSON).

Прогнозы хранятся по хешу версии модели и присланных файлов, поэтому повторный POST с теми же данными отдаётся сразу
и совпадает байт в байт. GET без `forecast_id` берёт последний прогноз, а с `forecast_id` (его возвращает POST
в заголовке `X-Forecast-Id`) отдаёт указанный. Ошибки возвращаются как `{"detail": "...", "code": "..."}`."""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.service = ForecastService(settings.model_dir, settings.cache_dir, settings.max_forecasts,
                                                settings.max_concurrent, settings.storage_url)
            log.info("модель %s %s загружена", app.state.service.model.kind, app.state.service.model.version)
        except Exception as e:
            app.state.service = None
            app.state.load_error = f"{type(e).__name__}: {e}"
            log.exception("модель не загрузилась")
        try:
            yield
        finally:
            if app.state.service is not None:
                app.state.service.store.close()

    app = FastAPI(title="Прогноз продаж: склад × товар", description=DESCRIPTION, version="1.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url="/openapi.json" if settings.docs else None)
    app.state.settings = settings
    add_handlers(app)
    app.include_router(system.router)
    if settings.admin_token:
        app.include_router(system.admin)
    app.include_router(predict.router)
    if settings.docs:
        if docs.STATIC.exists():
            app.mount("/static", StaticFiles(directory=docs.STATIC), name="static")
        app.include_router(docs.router)
    else:
        app.include_router(docs.plain)
    return app


app = create_app()
