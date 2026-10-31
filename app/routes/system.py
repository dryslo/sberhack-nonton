"""Служебные эндпоинты: проверки, описание модели и её перезагрузка."""
from typing import Annotated

from fastapi import APIRouter, Depends, Header
from fastapi.concurrency import run_in_threadpool

from ..config import Settings
from ..deps import get_service, get_settings
from ..errors import ApiError
from ..schemas import ErrorResponse, Health, ModelInfo, Ready, Reloaded
from ..service import ForecastService

router = APIRouter(tags=["служебные"])
admin = APIRouter(tags=["служебные"])  # подключается, только если задан ADMIN_TOKEN
Service = Annotated[ForecastService, Depends(get_service)]
NOT_READY = {503: {"model": ErrorResponse, "description": "Модель не загружена"}}


def require_admin(settings: Annotated[Settings, Depends(get_settings)],
                  x_admin_token: Annotated[str | None, Header()] = None):
    if x_admin_token != settings.admin_token:
        raise ApiError(403, "FORBIDDEN", "Неверный X-Admin-Token")


@router.get("/health", summary="Процесс жив", response_model=Health)
async def health():
    return {"status": "ok"}


@router.get("/ready", summary="Модель загружена", response_model=Ready, responses=NOT_READY)
async def ready(service: Service):
    return {"status": "ready", "model_kind": service.model.kind, "model_version": service.model.version}


@router.get("/model/info", summary="Описание модели: обучение, ансамбль, решение, метрики", response_model=ModelInfo,
            response_model_exclude_unset=True, responses=NOT_READY)
async def model_info(service: Service):
    return service.model.info()


@admin.post("/model/reload", summary="Перечитать модель из MODEL_DIR без рестарта", response_model=Reloaded,
            dependencies=[Depends(require_admin)],
            responses={403: {"model": ErrorResponse, "description": "Неверный X-Admin-Token"}, **NOT_READY})
async def model_reload(service: Service):
    previous = service.model.version
    return {"status": "reloaded", "model_version": await run_in_threadpool(service.reload), "previous_version": previous}
