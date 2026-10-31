"""Итоговая модель: обучить, сохранить в model/, загрузить и сделать прогноз по присланным данным.

Этим же кодом считается сабмит (notebooks/06_final) и прогноз в контейнере (app/). Признаки берутся из features.Panel,
решение 0 или k из pipeline.decide, как в бэктесте. Окно признаков не длиннее 24 недель, поэтому прогноз
по последним ~180 дням совпадает с прогнозом по всей истории.
"""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

import classifier
import lgbm
import regressor
from core import HORIZON, forecast_weeks, to_long, universe
from features import CAT_FEATURES, WINDOW, Panel
from pipeline import boost_factor, decide, member_settings, member_weights

DECISION = ("threshold", "integer", "shift", "amount_scale", "dispersion", "boost")
LIBRARIES = ("pandas", "numpy", "scipy", "catboost", "lightgbm", "holidays")


def prepare(sales, stock):
    # по ТЗ товар в stock называется char_article, в выданных файлах article: приводим к article
    sales = sales.assign(date=pd.to_datetime(sales["date"], format="%Y-%m-%d"))
    stock = stock.rename(columns={"char_article": "article"})
    stock = stock.assign(date=pd.to_datetime(stock["date"], format="%Y-%m-%d"))
    return sales, stock


def model_version(model_dir):
    h = hashlib.sha256()
    for f in sorted(Path(model_dir).iterdir()):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def library_versions():
    out = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            pass
    return out


class Forecaster:
    """members: участники ансамбля (clf, reg, clf_features, reg_features, target, settings).
    decision: параметры pipeline.decide. info: что записать в config.json."""

    def __init__(self, members, decision, info=None, version=None):
        self.members, self.decision, self.info, self.version = members, decision, info or {}, version

    @classmethod
    def fit(cls, sales, stock, ensemble=({},), **params):
        pairs, weeks = universe(sales, stock), forecast_weeks(sales.date.max())
        members = []
        for m in ensemble:
            c_set, r_set = member_settings(m, "clf"), member_settings(m, "reg")
            c = classifier.fit_predict(sales, stock, pairs, weeks, **c_set)
            r = regressor.fit_predict(sales, stock, pairs, weeks, **r_set)
            members.append({"clf": c["model"], "reg": r["model"], "settings": m,
                            "clf_features": list(c_set.get("features") or classifier.FEATURES),
                            "reg_features": list(r_set.get("features") or regressor.FEATURES),
                            "target": r_set.get("target") or regressor.TARGET})
        info = {"train_cutoff": str(sales.date.max().date()),
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "classifier": {"params": dict(classifier.PARAMS), "half_life": classifier.HALF_LIFE,
                               "origin_step": classifier.ORIGIN_STEP},
                "regressor": {"params": dict(regressor.PARAMS), "half_life": regressor.HALF_LIFE,
                              "origin_step": regressor.ORIGIN_STEP}}
        return cls(members, {k: v for k, v in params.items() if k in DECISION}, info)

    def save(self, model_dir, **extra):
        d = Path(model_dir)
        d.mkdir(parents=True, exist_ok=True)
        members = []
        for i, m in enumerate(self.members):
            files = {}
            for kind in ("clf", "reg"):
                if isinstance(m[kind], lgbm.Model):
                    files[kind] = f"{kind}_{i}.txt"
                    m[kind].booster.save_model(str(d / files[kind]))
                else:
                    files[kind] = f"{kind}_{i}.cbm"
                    m[kind].save_model(str(d / files[kind]))
            members.append({"settings": m["settings"], "files": files, "clf_features": m["clf_features"],
                            "reg_features": m["reg_features"], "target": m["target"]})
        config = {"kind": "two_stage", **self.info, **extra, "members": members, "decision": self.decision,
                  "window_weeks": WINDOW, "horizon": HORIZON, "cat_features": CAT_FEATURES, "libraries": library_versions()}
        (d / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1, default=str))
        self.version = model_version(d)
        return self.version

    @classmethod
    def load(cls, model_dir):
        d = Path(model_dir)
        config = json.loads((d / "config.json").read_text())
        members = []
        for m in config["members"]:
            models = {}
            for kind, file in m["files"].items():
                if file.endswith(".txt"):
                    import lightgbm as lgb
                    models[kind] = lgbm.Model(kind, seed=0)
                    models[kind].booster = lgb.Booster(model_file=str(d / file))
                else:
                    models[kind] = (CatBoostClassifier if kind == "clf" else CatBoostRegressor)()
                    models[kind].load_model(str(d / file))
            members.append({**models, "settings": m["settings"], "clf_features": m["clf_features"],
                            "reg_features": m["reg_features"], "target": m["target"]})
        info = {k: v for k, v in config.items() if k not in ("members", "decision")}
        return cls(members, config["decision"], info, version=model_version(d))

    def predict(self, sales, stock):
        return self.predict_details(sales, stock)["forecast"]

    def predict_details(self, sales, stock):
        # кроме прогноза отдаёт p, q и строки признаков, они нужны для /explain
        pairs, weeks = universe(sales, stock), forecast_weeks(sales.date.max())
        panel = Panel(sales, stock, pairs)
        rows = panel.forecast_rows(weeks)
        n, k = len(pairs), len(weeks)
        settings = [m["settings"] for m in self.members]
        p = np.average([m["clf"].predict_proba(rows[m["clf_features"]])[:, 1].reshape(k, n).T for m in self.members],
                       axis=0, weights=member_weights(settings, "clf"))
        q = np.expm1(np.average([np.log1p(self._amount(m, rows, k, n)) for m in self.members], axis=0,
                                weights=member_weights(settings, "reg")))
        params = dict(self.decision)
        boost = params.pop("boost", None)
        values = decide(p, q, weekly=panel.sales if params.get("dispersion") == "pair" else None,
                        factor=boost_factor(sales, pairs, boost) if boost else None, **params)
        forecast = to_long(pd.DataFrame(values, index=pd.MultiIndex.from_frame(pairs), columns=weeks))
        return {"forecast": forecast, "p": p, "q": q, "rows": rows, "pairs": pairs, "weeks": weeks}

    @staticmethod
    def _amount(m, rows, k, n):
        # как в regressor.fit_predict
        X = rows[m["reg_features"]]
        raw = np.expm1(m["reg"].predict(X)) if m["target"] == "log" else m["reg"].predict(X, prediction_type="Exponent")
        return np.maximum(np.asarray(raw).reshape(k, n).T, 1.0)
