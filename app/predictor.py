"""Загрузка модели и прогноз. Приложение работает только через интерфейс ForecastModel, конкретная
реализация выбирается полем kind в config.json папки модели."""
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from catboost import Pool  # noqa: E402
from core import make_forecast  # noqa: E402
from features import CAT_FEATURES  # noqa: E402
from forecaster import Forecaster, prepare  # noqa: E402
from pipeline import by_horizon, decide  # noqa: E402


@dataclass
class Prediction:
    forecast: pd.DataFrame
    details: dict | None = None


class ForecastModel(Protocol):
    kind: str
    version: str

    def predict(self, sales: pd.DataFrame, stock: pd.DataFrame) -> Prediction:
        """Прогноз на 9 недель с понедельника недели, следующей за последней датой в sales."""

    def info(self) -> dict:
        """Описание модели для /model/info."""

    def explain(self, details: dict, article: str, warehouse: str) -> dict:
        """Разбор прогноза по паре; если не поддерживается - NotImplementedError."""


INTEGER_RULES = {"smape": "целое с наименьшим ожидаемым SMAPE, продажи = 1 + Пуассон(q - 1)", "round": "ближайшее целое"}


class TwoStageModel:
    kind = "two_stage"

    def __init__(self, forecaster: Forecaster):
        self.forecaster = forecaster
        self.version = forecaster.version

    @classmethod
    def load(cls, model_dir):
        return cls(Forecaster.load(model_dir))

    def predict(self, sales, stock):
        d = self.forecaster.predict_details(*prepare(sales, stock))
        used = sorted({c for m in self.forecaster.members for c in m["clf_features"] + m["reg_features"]})
        details = {"p": d["p"], "q": d["q"], "pairs": d["pairs"], "weeks": d["weeks"], "rows": d["rows"][used]}
        return Prediction(d["forecast"], details)

    def info(self):
        i, dec = self.forecaster.info, self.forecaster.decision
        integer = dec.get("integer")
        return {
            "kind": self.kind,
            "version": self.version,
            "description": "классификатор (будет ли продажа на неделе) + регрессор (сколько, если будет) "
                           "+ правило 0 или k под метрику; в ансамбле вероятности усредняются, количества в логарифмах",
            "train_cutoff": i.get("train_cutoff"),
            "trained_at": i.get("trained_at"),
            "ensemble": [{"library": m["settings"].get("library", "catboost"), "seed": m["settings"].get("random_seed"),
                          "classifier_features": len(m["clf_features"]), "regressor_features": len(m["reg_features"])}
                         for m in self.forecaster.members],
            "decision": {"threshold": dec.get("threshold"),
                         "integer": "соответствие прогноз -> целое, выученное по фактам" if isinstance(integer, dict)
                         else INTEGER_RULES.get(integer, integer)},
            "window_weeks": i.get("window_weeks"),
            "horizon_weeks": i.get("horizon"),
            "backtest_score": i.get("backtest"),
            "training": {"classifier": i.get("classifier"), "regressor": i.get("regressor")},
            "libraries": i.get("libraries"),
        }

    def explain(self, details, article, warehouse, top=5):
        pairs, weeks = details["pairs"], details["weeks"]
        found = np.flatnonzero((pairs.article == article).to_numpy() & (pairs.warehouse_name == warehouse).to_numpy())
        if not len(found):
            raise KeyError(f"нет пары {warehouse} × {article}")
        i, n, k = found[0], len(pairs), len(weeks)
        dec = {key: v for key, v in self.forecaster.decision.items() if key in ("threshold", "integer", "amount_scale")}
        p, q = details["p"][i:i + 1], details["q"][i:i + 1]
        values = decide(p, q, **dec)[0]
        thresholds = by_horizon(dec.get("threshold", 0.5), k)
        rows = details["rows"].iloc[[h * n + i for h in range(k)]]
        member = next(m for m in self.forecaster.members if m["settings"].get("library", "catboost") == "catboost")
        return {
            "warehouse_name": warehouse,
            "article": article,
            "weeks": [{"week": str(pd.Timestamp(w).date()), "probability": round(float(p[0, h]), 4),
                       "threshold": float(thresholds[h]), "sale": bool(p[0, h] > thresholds[h]),
                       "amount_raw": round(float(q[0, h]), 3), "quantity": float(values[h])} for h, w in enumerate(weeks)],
            "classifier_top_features": self._top(member["clf"], rows[member["clf_features"]], top),
            "regressor_top_features": self._top(member["reg"], rows[member["reg_features"]], top),
            "note": "вероятность и количество считаются по всему ансамблю, а SHAP по первой CatBoost-модели "
                    "(среднее по 9 неделям): у классификатора в логитах, у регрессора в log1p количества",
        }

    @staticmethod
    def _top(model, X, top):
        shap = model.get_feature_importance(Pool(X, cat_features=model.get_cat_feature_indices()), type="ShapValues")[:, :-1]
        order = np.argsort(-np.abs(shap).mean(axis=0))[:top]
        return [{"feature": X.columns[j], "value": TwoStageModel._value(X.columns[j], X.iloc[0, j]),
                 "contribution": round(float(shap[:, j].mean()), 4)} for j in order]

    @staticmethod
    def _value(feature, v):
        # категории закодированы номером из имени (warehouse_name_3 -> 3), -1 значит "нет значения"
        if pd.isna(v):
            return None
        if feature in CAT_FEATURES:
            return f"{feature}_{int(v)}" if v >= 0 else None
        return round(float(v), 4)


class BaselineModel:
    kind = "baseline"

    def __init__(self, config: dict):
        self.config = {"window": 8, "zero_below": 1.0, **config}
        self.version = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def load(cls, model_dir):
        return cls(json.loads((Path(model_dir) / "config.json").read_text()))

    def predict(self, sales, stock):
        s, st = prepare(sales, stock)
        return Prediction(make_forecast(s, st, "mean", window=self.config["window"], zero_below=self.config["zero_below"]))

    def info(self):
        return {"kind": self.kind, "version": self.version,
                "description": f"среднее за {self.config['window']} недель, меньше {self.config['zero_below']:g} -> 0",
                "config": self.config}

    def explain(self, details, article, warehouse):
        raise NotImplementedError("У базовой модели нет объяснений")


MODELS = {model.kind: model for model in (TwoStageModel, BaselineModel)}


def load_model(model_dir) -> ForecastModel:
    config = json.loads((Path(model_dir) / "config.json").read_text())
    kind = config.get("kind", "two_stage")
    if kind not in MODELS:
        raise ValueError(f"Неизвестный тип модели {kind!r} в {model_dir}/config.json; есть: {', '.join(MODELS)}")
    return MODELS[kind].load(model_dir)
