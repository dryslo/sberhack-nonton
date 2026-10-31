"""Регрессор: сколько продадут за неделю, если продажа будет (CatBoost, в ансамбле ещё LightGBM).

Учится только на неделях с продажами, метрики тоже считаются только на них: если продажи нет, любое ненулевое
число даёт в SMAPE 200 %, и важно только решение классификатора.
Метод "regressor" ставит прогноз регрессора там, где по простому правилу будет продажа (продажи хотя бы в половине
из 8 недель). Так виден вклад одного регрессора. Вместе с классификатором их собирает pipeline.py.
Для сравнения везде используется обычное количество: сколько пара продаёт за неделю, когда продаёт.
"""
import gc

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import lgbm

from core import FOLDS, MAIN_FOLD, ON_KAGGLE, SEED, _fold_data, forecast_weeks, pair_info, register, to_long
from features import CAT_FEATURES, GROUPS, Panel, get_panel, select, typical_amount

# признаки про уровень: сколько продают, когда продают, цена, скидка, широта спроса.
# Стартовый набор выбран на глаз, польза групп видна в feature_ablation()
FEATURES = select("лаги", "уровень", "обычное количество", "тренды", "товар на всех складах", "цена и скидка",
                  "география", "категории", "календарь", extra=["nz8", "stock", "stock_mean28"])

# локально облегчённые настройки, на Kaggle полные
PARAMS = {"iterations": 300, "learning_rate": 0.1, "depth": 6, "loss_function": "RMSE",
          "max_ctr_complexity": 1,
          "random_seed": SEED, "verbose": 0, "allow_writing_files": False, "thread_count": -1}
ORIGIN_STEP = 4
ABLATION = {"origin_step": 8, "iterations": 150}
if ON_KAGGLE:
    PARAMS.update(iterations=400, task_type="GPU")
    ORIGIN_STEP = 1
    ABLATION = {"origin_step": 2, "iterations": 500}
SIMPLE = "обычное количество"

# "log": log1p(продажи) с RMSE, прогноз ближе к медиане и занижает крупные недели (прогноз / факт ~0.85).
# "tweedie": штуки с потерей Tweedie (как в M5), прогноз ближе к среднему. На наших фолдах log лучше
TARGET = "log"
TWEEDIE_POWER = 1.5
HALF_LIFE = 26
SEASON_BOOST = 0.0

_CACHE = {}


def fit_predict(sales, stock, pairs, weeks, origin_step=None, features=None, target=None, power=None, half_life=None,
                library="catboost", season_boost=None, **params):
    # количество при условии, что продажа будет: пары x недели прогноза
    step = origin_step or ORIGIN_STEP
    half_life = half_life or HALF_LIFE
    season_boost = SEASON_BOOST if season_boost is None else season_boost
    features = list(FEATURES if features is None else features)
    target, power = target or TARGET, power or TWEEDIE_POWER
    params = {**PARAMS, **params}
    if target == "tweedie":
        params["loss_function"] = f"Tweedie:variance_power={power}"
    key = (sales.date.max(), len(sales), len(stock), len(pairs), tuple(weeks), step, half_life, season_boost, target, library,
           tuple(sorted((k, tuple(sorted(v.items())) if isinstance(v, dict) else v) for k, v in params.items())),
           tuple(features))
    if key not in _CACHE:
        panel = get_panel(sales, stock, pairs)
        train = panel.training_set(step, only_sold=True, columns=features, half_life=half_life,
                                    season_boost=season_boost)
        y = train.pop("target")
        weight = train.pop("weight") if "weight" in train else None
        cats = [c for c in CAT_FEATURES if c in features]
        if library == "lightgbm":  # LightGBM только с log1p
            model = lgbm.make("reg", params.get("random_seed", SEED), params.get("lgbm"))
            model.fit(train, np.log1p(y), sample_weight=weight, categorical_feature=cats)
        else:
            model = CatBoostRegressor(**{k: v for k, v in params.items() if k != "lgbm"}, cat_features=cats)
            model.fit(train, np.log1p(y) if target == "log" else y, sample_weight=weight)
        n_train = len(train)
        del train, y
        gc.collect()
        rows = panel.forecast_rows(weeks)
        X = rows[features]
        raw = np.expm1(model.predict(X)) if target == "log" else model.predict(X, prediction_type="Exponent")
        pred = raw.reshape(len(weeks), len(pairs)).T
        _CACHE[key] = {"amount": np.maximum(pred, 1.0),  # если продажа есть, то хотя бы 1 штука
                       "rows": rows, "model": model, "n_train": n_train}
    return _CACHE[key]


@register("regressor")
def regressor_method(sales, stock, pairs, weeks, gate=0.5, amount="model", **model):
    # продажа будет, если nz8 >= gate. amount="simple" ставит обычное количество вместо регрессора
    if amount == "model":
        r = fit_predict(sales, stock, pairs, weeks, **model)
        rows, q = r["rows"], r["amount"]
    else:
        rows = Panel(sales, stock, pairs).forecast_rows(weeks)
        q = np.repeat(typical_amount(rows, len(pairs))[:, None], len(weeks), axis=1)
    nz8 = rows.nz8.to_numpy()[:len(pairs)]
    values = np.where(nz8[:, None] >= gate, q, 0.0)
    return to_long(pd.DataFrame(values, index=pd.MultiIndex.from_frame(pairs), columns=weeks))


def regression_metrics(fact, pred):
    fact, pred = np.ravel(fact).astype(float), np.ravel(pred).astype(float)
    err = pred - fact
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "WAPE": float(np.abs(err).sum() / fact.sum() * 100),
        "SMAPE": float(np.mean(2 * np.abs(err) / (fact + pred)) * 100),
        "RMSLE": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(fact)) ** 2))),
        "прогноз / факт": float(pred.sum() / fact.sum()),
        "доля завышений": float(np.mean(pred > fact)),
    }


def fold_data(fold, **model):
    pairs, s_cut, st_cut, truth, _ = _fold_data(fold, FOLDS[fold])
    weeks = forecast_weeks(s_cut.date.max())
    r = fit_predict(s_cut, st_cut, pairs, weeks, **model)
    n, k = len(pairs), len(weeks)
    fact = truth.quantity.to_numpy().reshape(k, n).T
    simple = np.repeat(typical_amount(r["rows"], n)[:, None], k, axis=1)
    return fact, r["amount"], simple, r


def evaluate(fold=MAIN_FOLD, **model):
    """Метрики регрессора на неделях с продажами: summary, by_horizon, by_segment, calibration, importance."""
    fact, q, simple, r = fold_data(fold, **model)
    sold = fact > 0
    out = {"summary": pd.DataFrame({"CatBoost": regression_metrics(fact[sold], q[sold]),
                                    SIMPLE: regression_metrics(fact[sold], simple[sold])}).T}

    out["by_horizon"] = pd.DataFrame({
        h: {"SMAPE": regression_metrics(fact[sold[:, h - 1], h - 1], q[sold[:, h - 1], h - 1])["SMAPE"],
            f"SMAPE ({SIMPLE})": regression_metrics(fact[sold[:, h - 1], h - 1], simple[sold[:, h - 1], h - 1])["SMAPE"],
            "прогноз / факт": regression_metrics(fact[sold[:, h - 1], h - 1], q[sold[:, h - 1], h - 1])["прогноз / факт"],
            "недель с продажами": int(sold[:, h - 1].sum())}
        for h in range(1, q.shape[1] + 1)}).T.rename_axis("горизонт, недель")

    typ = simple[:, 0]
    segment = np.select([typ <= 1.5, typ <= 4], ["обычно 1 шт.", "обычно 2-4 шт."], "обычно больше 4 шт.")
    rows = {}
    for name in ["обычно 1 шт.", "обычно 2-4 шт.", "обычно больше 4 шт."]:
        m = (segment[:, None] == name) & sold
        mm, ms = regression_metrics(fact[m], q[m]), regression_metrics(fact[m], simple[m])
        rows[name] = {"недель с продажами": int(m.sum()), "SMAPE": mm["SMAPE"], f"SMAPE ({SIMPLE})": ms["SMAPE"],
                      "WAPE": mm["WAPE"], f"WAPE ({SIMPLE})": ms["WAPE"], "прогноз / факт": mm["прогноз / факт"]}
    out["by_segment"] = pd.DataFrame(rows).T

    # если регрессор говорит 3 штуки, сколько в среднем продают на самом деле
    edges = [1, 1.5, 2, 3, 5, 10, np.inf]
    b = np.digitize(q[sold], edges[1:-1])
    out["calibration"] = pd.DataFrame({
        f"{edges[i]:g}-{edges[i + 1]:g}": {"недель": int((b == i).sum()), "средний прогноз": q[sold][b == i].mean(),
                                          "средний факт": fact[sold][b == i].mean()}
        for i in range(len(edges) - 1) if (b == i).any()}).T.rename_axis("прогноз, шт.")

    out["importance"] = pd.Series(r["model"].get_feature_importance(), index=r["model"].feature_names_,
                                  name="важность").sort_values(ascending=False)
    return {k: v.round(4) for k, v in out.items()}


def summary(folds=FOLDS, **model):
    rows = {}
    for fold in folds:
        fact, q, simple, _ = fold_data(fold, **model)
        sold = fact > 0
        rows[(fold, "CatBoost")] = regression_metrics(fact[sold], q[sold])
        rows[(fold, SIMPLE)] = regression_metrics(fact[sold], simple[sold])
    return pd.DataFrame(rows).T.round(4)


def plot(fold=MAIN_FOLD, **model):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fact, q, simple, _ = fold_data(fold, **model)
    sold = fact > 0
    blue, orange, gray = "#2a78d6", "#eb6834", "#c3c2b7"
    fig, (cal, hor) = plt.subplots(1, 2, figsize=(13, 4.3), layout="constrained")
    edges = np.array([1, 1.5, 2, 3, 5, 10, 20, np.inf])
    top = 0
    for name, pred, color in [("CatBoost", q, blue), (SIMPLE, simple, orange)]:
        d = pd.DataFrame({"прогноз": pred[sold], "факт": fact[sold], "бин": np.digitize(pred[sold], edges[1:-1])})
        g = d.groupby("бин").mean()
        top = max(top, g.to_numpy().max())
        sns.lineplot(data=g, x="прогноз", y="факт", color=color, marker="o", ax=cal, label=name)
        smape = [regression_metrics(fact[sold[:, h], h], pred[sold[:, h], h])["SMAPE"] for h in range(q.shape[1])]
        sns.lineplot(x=np.arange(1, q.shape[1] + 1), y=smape, color=color, marker="o", ax=hor, label=name)
    cal.plot([1, top], [1, top], color=gray, lw=0.8)
    cal.set(xscale="log", yscale="log", title="Средний факт по группам прогноза",
            xlabel="средний прогноз, шт.", ylabel="средний факт, шт.")
    hor.set(title="SMAPE на неделях с продажами по горизонту", xlabel="горизонт, недель", ylabel="SMAPE, %")
    for ax in (cal, hor):
        ax.legend(frameon=False)
    fig.suptitle(f"Регрессор на фолде {fold}", x=0.01, ha="left")
    plt.show()


def feature_ablation(fold=MAIN_FOLD):
    """Польза каждой группы признаков: группы из FEATURES убираем, остальные добавляем.
    Польза > 0 значит, что с группой лучше. Каждая строка - отдельная модель на настройках ABLATION."""
    def metrics(features):
        before = set(_CACHE)
        fact, q, _, _ = fold_data(fold, features=features, **ABLATION)
        for key in set(_CACHE) - before:
            del _CACHE[key]
        m = regression_metrics(fact[fact > 0], q[fact > 0])
        return np.array([m["SMAPE"], m["RMSLE"], m["WAPE"]])

    base = metrics(FEATURES)
    rows = {"набор FEATURES целиком": {"в наборе": "", "что проверили": "", "SMAPE": base[0], "RMSLE": base[1],
                                       "WAPE": base[2]}}
    for group, (cols, _) in GROUPS.items():
        inside = any(c in FEATURES for c in cols)
        variant = [f for f in FEATURES if f not in cols] if inside else FEATURES + [c for c in cols if c not in FEATURES]
        m = metrics(variant)
        gain = m - base if inside else base - m  # все три метрики: меньше лучше
        rows[group] = {"в наборе": "да" if inside else "нет", "что проверили": "убрали" if inside else "добавили",
                       "SMAPE": m[0], "RMSLE": m[1], "WAPE": m[2],
                       "польза для SMAPE": gain[0], "польза для RMSLE": gain[1], "польза для WAPE": gain[2]}
    t = pd.DataFrame(rows).T
    gain = t["польза для SMAPE"].astype(float)
    t["вывод"] = np.where(gain > 0.1, "помогает", np.where(gain < -0.1, "мешает", "почти не влияет"))
    t.loc["набор FEATURES целиком", "вывод"] = ""
    return t


def error_analysis(fold=MAIN_FOLD, top=15, **model):
    """Где регрессор ошибается на неделях с продажами: ошибка и перекос по категориям и складам, ошибка по размеру
    факта (большие недели обычно занижены) и самые большие промахи."""
    fact, q, simple, r = fold_data(fold, **model)
    pairs, s_cut = _fold_data(fold, FOLDS[fold])[:2]
    n, k = q.shape
    info = pair_info(s_cut, pairs)
    look = ["typ8", "mean8", "max8", "nz8", "trend", "price_trend", "stock"]
    pair_feats = r["rows"].iloc[:n][look].reset_index(drop=True)
    long = pd.DataFrame({c: np.repeat(info[c].to_numpy(), k) for c in ["warehouse_name", "article", "category_lvl_1"]})
    long["h"] = np.tile(np.arange(1, k + 1), n)
    long["факт"], long["прогноз"] = fact.ravel(), q.ravel()
    for c in look:
        long[c] = np.repeat(pair_feats[c].to_numpy(), k)
    long = long[long["факт"] > 0].copy()
    long["ошибка"] = long["прогноз"] - long["факт"]
    long["abs"] = long["ошибка"].abs()
    long["smape"] = 2 * long["abs"] / (long["факт"] + long["прогноз"]) * 100
    total = long["abs"].sum()

    def by(col):
        g = long.groupby(col)
        return pd.DataFrame({
            "недель с продажами": g.size(),
            "SMAPE": g["smape"].mean(),
            "WAPE": g["abs"].sum() / g["факт"].sum() * 100,
            "прогноз / факт": g["прогноз"].sum() / g["факт"].sum(),
            "доля всей ошибки": g["abs"].sum() / total,
        }).sort_values("доля всей ошибки", ascending=False)

    level = pd.cut(long["факт"], [0, 1, 3, 9, np.inf], labels=["1 шт.", "2-3 шт.", "4-9 шт.", "10+ шт."])
    g = long.groupby(level, observed=True)
    by_fact = pd.DataFrame({"недель": g.size(), "средний факт": g["факт"].mean(), "средний прогноз": g["прогноз"].mean(),
                            "прогноз / факт": g["прогноз"].sum() / g["факт"].sum(), "доля всей ошибки": g["abs"].sum() / total})
    shown = ["warehouse_name", "article", "category_lvl_1", "h", "факт", "прогноз", "ошибка"] + look
    out = {"by_category": by("category_lvl_1"), "by_warehouse": by("warehouse_name"), "by_fact": by_fact.rename_axis("факт за неделю"),
           "biggest_misses": long.nlargest(top, "abs")[shown]}
    return {k: v.round(3) for k, v in out.items()}
