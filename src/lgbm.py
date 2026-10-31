"""LightGBM для ансамбля: те же признаки и строки обучения, что у CatBoost.

Модели разных библиотек ошибаются по-разному, поэтому их среднее надёжнее среднего нескольких сидов одной модели.
В ансамбле задаётся как {"library": "lightgbm"} (см. pipeline.py). Используется родной API, чтобы не тянуть scikit-learn.
"""
import numpy as np

ROUNDS = 400
PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50, "bagging_fraction": 0.8, "bagging_freq": 1,
          "feature_fraction": 0.8, "lambda_l2": 1.0, "deterministic": True, "force_row_wise": True, "num_threads": 4,
          "verbose": -1}


class Model:
    """kind: "clf" (будет ли продажа, binary) или "reg" (log1p количества). Методы как у CatBoost."""

    def __init__(self, kind, seed, overrides=None):
        self.kind, self.seed, self.booster = kind, seed, None
        self.params = {**PARAMS, **dict(overrides or {})}  # "rounds" задаёт число деревьев

    def fit(self, X, y, sample_weight=None, categorical_feature=()):
        import lightgbm as lgb
        params = {"objective": "binary" if self.kind == "clf" else "regression", "seed": self.seed, **self.params}
        rounds = params.pop("rounds", ROUNDS)
        # категория -1 (нет значения) LightGBM считает пропуском, так и нужно
        data = lgb.Dataset(X, label=np.asarray(y), weight=None if sample_weight is None else np.asarray(sample_weight),
                           categorical_feature=list(categorical_feature), params={"verbose": -1})
        self.booster = lgb.train(params, data, num_boost_round=rounds)
        return self

    def predict(self, X):
        return self.booster.predict(X)

    def predict_proba(self, X):
        p = self.booster.predict(X)
        return np.column_stack([1 - p, p])


def make(kind, seed, overrides=None):
    return Model(kind, seed, overrides)
