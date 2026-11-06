"""Прогноз: расчёт по присланным файлам, выборки из сохранённого прогноза и разбор прогноза по паре."""
import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse

from ..deps import get_service
from ..errors import ApiError
from ..schemas import ErrorResponse, Explanation, ForecastRow
from ..service import ForecastService, NotFound

router = APIRouter(tags=["прогноз"])
Service = Annotated[ForecastService, Depends(get_service)]
ForecastId = Annotated[str | None, Query(description="Какой прогноз (X-Forecast-Id), по умолчанию последний")]

CSV = {"text/csv": {"example": "week,warehouse_name,article,quantity\n2026-07-27,warehouse_name_0,char_article_0,1.0\n"}}
ROWS = {"application/json": {"schema": {"type": "array", "items": ForecastRow.model_json_schema()}}}
ERRORS = {400: {"model": ErrorResponse, "description": "Пустой файл, не CSV или неверный параметр"},
          404: {"model": ErrorResponse, "description": "Прогноза ещё нет или в нём нет такой пары"},
          422: {"model": ErrorResponse, "description": "Данные не прошли проверку (колонки, даты, глубина истории, отрицательные продажи)"}}


def headers(service, forecast_id, filename, extra=None):
    return {"X-Forecast-Id": forecast_id, "X-Model-Version": service.model.version,
            "Content-Disposition": f"attachment; filename={filename}", **(extra or {})}


@router.post("/predict", summary="Прогноз по загруженным данным", response_class=Response,
             responses={200: {"content": CSV, "description": "CSV: week, warehouse_name, article, quantity на 9 недель"}, **ERRORS})
async def predict(service: Service,
                  sales: Annotated[UploadFile, File(description="CSV продаж по дням (формат sales_dataset_hakaton.csv), 180+ дней")],
                  stock: Annotated[UploadFile, File(description="CSV остатков по дням (формат stock_dataset_hakaton.csv), 180+ дней")]):
    result = await run_in_threadpool(service.forecast, await sales.read(), await stock.read())
    timing = ", ".join(f"{stage};dur={ms}" for stage, ms in result.timings.items())
    return Response(result.csv, media_type="text/csv",
                    headers=headers(service, result.forecast_id, "forecast.csv",
                                    {"X-Cache": "hit" if result.cache_hit else "miss", "Server-Timing": timing}))


@router.get("/predict", summary="Прогноз по товару и/или складу", response_class=Response,
            responses={200: {"content": CSV, "description": "CSV прогноза по выбранным парам"}, **ERRORS})
async def predict_filtered(service: Service,
                           article: Annotated[str | None, Query(description="Артикул, например char_article_1")] = None,
                           warehouse: Annotated[str | None, Query(description="Склад, например warehouse_name_3")] = None,
                           forecast_id: ForecastId = None):
    if not article and not warehouse:
        raise ApiError(400, "BAD_REQUEST", "Укажите хотя бы один параметр: article или warehouse")
    forecast_id, df = await run_in_threadpool(service.frame, forecast_id)
    mask = (df.article == article if article else True) & (df.warehouse_name == warehouse if warehouse else True)
    if not mask.any():
        what = " × ".join(x for x in (warehouse, article) if x)
        raise NotFound(f"В прогнозе {forecast_id} нет {what}")
    return Response(df[mask].to_csv(index=False).encode(), media_type="text/csv",
                    headers=headers(service, forecast_id, "filtered_forecast.csv"))


@router.get("/predict/all", summary="Весь прогноз целиком", response_class=Response,
            responses={200: {"content": {**CSV, **ROWS}, "description": "CSV (по умолчанию) или JSON-массив"}, **ERRORS})
async def predict_all(service: Service, format: Annotated[str, Query(description="csv или json")] = "csv",
                      forecast_id: ForecastId = None):
    if format not in ("csv", "json"):
        raise ApiError(400, "BAD_REQUEST", "Параметр format должен быть csv или json")
    if format == "csv":
        forecast_id, data = await run_in_threadpool(service.csv, forecast_id)
        return Response(data, media_type="text/csv", headers=headers(service, forecast_id, "full_forecast.csv"))
    forecast_id, df = await run_in_threadpool(service.frame, forecast_id)

    # JSON отдаём кусками, без сборки всего ответа в памяти
    def rows(chunk=5000):
        yield "["
        for start in range(0, len(df), chunk):
            part = df.iloc[start:start + chunk].to_dict(orient="records")
            yield ("," if start else "") + ",".join(json.dumps(r, ensure_ascii=False) for r in part)
        yield "]"
    return StreamingResponse(rows(), media_type="application/json",
                             headers=headers(service, forecast_id, "full_forecast.json"))


@router.get("/explain", summary="Почему прогноз по паре такой", response_model=Explanation,
            responses={404: ERRORS[404], 409: {"model": ErrorResponse, "description": "Прогноз сделан другой версией модели"},
                       501: {"model": ErrorResponse, "description": "У модели нет объяснений"}})
async def explain(service: Service, article: Annotated[str, Query(description="Артикул")],
                  warehouse: Annotated[str, Query(description="Склад")], forecast_id: ForecastId = None):
    return await run_in_threadpool(service.explain, article, warehouse, forecast_id)
