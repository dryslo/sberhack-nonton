"""Ошибки API в одном формате: {"detail": "...", "code": "..."}, если проблем несколько, ещё список errors."""
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .service import Conflict, NotFound
from .validators import InputError


class ApiError(Exception):
    def __init__(self, status, code, detail):
        super().__init__(detail)
        self.status, self.code, self.detail = status, code, detail


def error(status, code, detail, errors=None):
    body = {"detail": detail, "code": code}
    if errors and len(errors) > 1:
        body["errors"] = errors
    return JSONResponse(body, status_code=status)


def add_handlers(app: FastAPI):
    @app.exception_handler(InputError)
    async def _input(_, e: InputError):
        return error(e.status, e.code, e.errors[0], e.errors)

    @app.exception_handler(NotFound)
    async def _not_found(_, e: NotFound):
        return error(404, e.code, str(e))

    @app.exception_handler(Conflict)
    async def _conflict(_, e: Conflict):
        return error(409, "MODEL_CHANGED", str(e))

    @app.exception_handler(NotImplementedError)
    async def _not_implemented(_, e: NotImplementedError):
        return error(501, "NOT_IMPLEMENTED", str(e) or "Не поддерживается")

    @app.exception_handler(ApiError)
    async def _api(_, e: ApiError):
        return error(e.status, e.code, e.detail)

    @app.exception_handler(RequestValidationError)
    async def _request(_, e: RequestValidationError):
        errors = [f"{'.'.join(str(x) for x in err['loc'] if x != 'body')}: {err['msg']}" for err in e.errors()]
        return error(422, "VALIDATION_ERROR", errors[0] if errors else "Неверный запрос", errors)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_, e: StarletteHTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(e.status_code, "HTTP_ERROR")
        return error(e.status_code, code, str(e.detail))
