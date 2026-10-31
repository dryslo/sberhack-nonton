from fastapi import Request

from .config import Settings
from .errors import ApiError
from .service import ForecastService


def get_service(request: Request) -> ForecastService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise ApiError(503, "MODEL_NOT_READY", f"Модель не загружена: {getattr(request.app.state, 'load_error', 'ещё грузится')}")
    return service


def get_settings(request: Request) -> Settings:
    return request.app.state.settings
