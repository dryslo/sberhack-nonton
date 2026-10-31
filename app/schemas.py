"""Схемы JSON-ответов. Они же попадают в /docs. CSV на входе проверяет validators.py: построчная проверка
через pydantic на сотнях тысяч строк была бы слишком медленной."""
from typing import Any

from pydantic import BaseModel, ConfigDict


class ErrorResponse(BaseModel):
    detail: str
    code: str
    errors: list[str] | None = None


class Health(BaseModel):
    status: str


class Ready(BaseModel):
    status: str
    model_kind: str
    model_version: str


class Reloaded(BaseModel):
    status: str
    model_version: str
    previous_version: str


class EnsembleMember(BaseModel):
    library: str
    seed: int | None
    classifier_features: int
    regressor_features: int


class ModelInfo(BaseModel):
    # у разных моделей свой набор полей, остальные поля передаются как есть
    model_config = ConfigDict(extra="allow")

    kind: str
    version: str
    description: str
    train_cutoff: str | None = None
    trained_at: str | None = None
    ensemble: list[EnsembleMember] | None = None
    decision: dict[str, Any] | None = None
    window_weeks: int | None = None
    horizon_weeks: int | None = None
    backtest_score: dict[str, float] | None = None
    training: dict[str, Any] | None = None
    libraries: dict[str, str] | None = None
    config: dict[str, Any] | None = None


class ForecastRow(BaseModel):
    week: str
    warehouse_name: str
    article: str
    quantity: float


class WeekExplanation(BaseModel):
    week: str
    probability: float
    threshold: float
    sale: bool
    amount_raw: float
    quantity: float


class FeatureContribution(BaseModel):
    feature: str
    value: float | str | None
    contribution: float


class Explanation(BaseModel):
    forecast_id: str
    warehouse_name: str
    article: str
    weeks: list[WeekExplanation]
    classifier_top_features: list[FeatureContribution]
    regressor_top_features: list[FeatureContribution]
    note: str
