"""Классификатор: будет ли продажа у пары на неделе (CatBoost, в ансамбле ещё LightGBM).

Метод "classifier" ставит обычное недельное количество пары там, где вероятность выше порога, и 0 в остальных
неделях. Так виден вклад одного классификатора. Количество прогнозирует regressor.py, вместе их собирает pipeline.py.
Для сравнения везде используется простое правило: доля недель с продажами за 8 недель как оценка вероятности.
"""
import gc

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import lgbm

from core import (FOLDS, MAIN_FOLD, ON_KAGGLE, SEED, _fold_data, compute_metrics, forecast_weeks, pair_info, register,
                  score_minmax, to_long)
from features import CAT_FEATURES, GROUPS, Panel, get_panel, select, typical_amount

# признаки про активность: продают ли вообще, как давно, есть ли остаток.
# Стартовый набор выбран на глаз, польза групп видна в feature_ablation()
FEATURES = select("лаги", "частота", "давность", "тренды", "товар на всех складах", "остатки", "категории", "календарь",
                  extra=["mean4", "mean8", "discount8"])

# локально облегчённые настройки (памяти ~8 ГБ), на Kaggle полные
PARAMS = {"iterations": 500, "learning_rate": 0.1, "depth": 6, "loss_function": "Logloss",
          "max_ctr_complexity": 1,  # без комбинаций категорий: памяти 2 ГБ вместо 3.4+, AUC -0.002
          "random_seed": SEED, "verbose": 0, "allow_writing_files": False, "thread_count": -1,"one_hot_max_size": 64}
ORIGIN_STEP = 4
ABLATION = {"origin_step": 8, "iterations": 150}
if ON_KAGGLE:  # глубина 8 и lr 0.05 на сезонных фолдах хуже, больше 400 деревьев не помогает (trees_curve)
    PARAMS.update(iterations=400, task_type="GPU")
    ORIGIN_STEP = 1
    ABLATION = {"origin_step": 2, "iterations": 500}
HALF_LIFE = 26  # через сколько недель от последней недели отсчёта вес строки падает вдвое
SEASON_BOOST = 0.0
SIMPLE = "простое правило"
EPS = 1e-6

_CACHE = {}


def fit_predict(sales, stock, pairs, weeks, origin_step=None, features=None, half_life=None, library="catboost", season_boost=None,
                **params):
    # вероятность продажи: пары x недели прогноза
    step = origin_step or ORIGIN_STEP
    half_life = half_life or HALF_LIFE
    season_boost = SEASON_BOOST if season_boost is None else season_boost
    features = list(FEATURES if features is None else features)
    params = {**PARAMS, **params}
    key = (sales.date.max(), len(sales), len(stock), len(pairs), tuple(weeks), step, half_life, season_boost, library,
           tuple(sorted((k, tuple(sorted(v.items())) if isinstance(v, dict) else v) for k, v in params.items())),
           tuple(features))
    if key not in _CACHE:
        panel = get_panel(sales, stock, pairs)
        train = panel.training_set(step, columns=features, half_life=half_life,
                                    season_boost=season_boost)
        y = (train.pop("target") > 0).astype(int)
        weight = train.pop("weight") if "weight" in train else None
        cats = [c for c in CAT_FEATURES if c in features]
        if library == "lightgbm":
            model = lgbm.make("clf", params.get("random_seed", SEED), params.get("lgbm"))
            model.fit(train, y, sample_weight=weight, categorical_feature=cats)
        else:
            model = CatBoostClassifier(**{k: v for k, v in params.items() if k != "lgbm"}, cat_features=cats)
            model.fit(train, y, sample_weight=weight)
        n_train = len(train)
        del train, y
        gc.collect()
        rows = panel.forecast_rows(weeks)
        p = model.predict_proba(rows[features])[:, 1].reshape(len(weeks), len(pairs)).T
        _CACHE[key] = {"p": p, "rows": rows, "model": model, "n_train": n_train}
    return _CACHE[key]


@register("classifier")
def classifier_method(sales, stock, pairs, weeks, threshold=0.5, **model):
    r = fit_predict(sales, stock, pairs, weeks, **model)
    values = np.where(r["p"] > threshold, typical_amount(r["rows"], len(pairs))[:, None], 0.0)
    return to_long(pd.DataFrame(values, index=pd.MultiIndex.from_frame(pairs), columns=weeks))


def _curve(sold, score):
    # точки ROC и PR; одинаковые оценки идут одной группой, чтобы результат не зависел от порядка строк
    order = np.argsort(-score, kind="mergesort")
    sold, score = sold[order], score[order]
    last = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]
    tp, fp = np.cumsum(sold)[last], np.cumsum(~sold)[last]
    return tp / sold.sum(), fp / (~sold).sum(), tp / (tp + fp)


def roc_auc(sold, score):
    tpr, fpr, _ = _curve(np.ravel(sold).astype(bool), np.ravel(score).astype(float))
    tpr, fpr = np.r_[0, tpr], np.r_[0, fpr]
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def average_precision(sold, score):
    recall, _, precision = _curve(np.ravel(sold).astype(bool), np.ravel(score).astype(float))
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def classification_metrics(sold, score, decision):
    sold, score, d = np.ravel(sold).astype(bool), np.ravel(score).astype(float), np.ravel(decision).astype(bool)
    tp, fp, fn, tn = (float(x.sum()) for x in (d & sold, d & ~sold, ~d & sold, ~d & ~sold))
    recall, spec, prec = tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1)
    pc = np.clip(score, EPS, 1 - EPS)
    mcc = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "ROC AUC": roc_auc(sold, score),
        "PR AUC": average_precision(sold, score),
        "logloss": float(-np.mean(np.where(sold, np.log(pc), np.log(1 - pc)))),
        "Brier": float(np.mean((score - sold) ** 2)),
        "accuracy": (tp + tn) / len(d),
        "balanced accuracy": (recall + spec) / 2,
        "precision": prec,
        "recall": recall,
        "specificity": spec,  # из недель без продаж доля правильно обнулённых, напрямую влияет на SMAPE
        "F1": 2 * prec * recall / max(prec + recall, EPS),
        "MCC": (tp * tn - fp * fn) / mcc if mcc else 0.0,
        'доля "будет продажа"': float(d.mean()),
        "доля недель с продажами": float(sold.mean()),
    }


def fold_data(fold, **model):
    pairs, s_cut, st_cut, truth, _ = _fold_data(fold, FOLDS[fold])
    weeks = forecast_weeks(s_cut.date.max())
    r = fit_predict(s_cut, st_cut, pairs, weeks, **model)
    n, k = len(pairs), len(weeks)
    sold = truth.quantity.to_numpy().reshape(k, n).T > 0
    simple = np.repeat(r["rows"].nz8.to_numpy()[:n, None], k, axis=1)
    return sold, r["p"], simple, r


def evaluate(fold=MAIN_FOLD, threshold=0.5, **model):
    """Метрики классификатора на фолде: summary, confusion, thresholds, by_horizon, by_segment, calibration, importance."""
    sold, p, simple, r = fold_data(fold, **model)
    d = p > threshold
    out = {"summary": pd.DataFrame({
        f"CatBoost, порог {threshold}": classification_metrics(sold, p, d),
        f"{SIMPLE}, >= 0.5": classification_metrics(sold, simple, simple >= 0.5),
    }).T}
    out["confusion"] = pd.DataFrame(
        [[int((~d & ~sold).sum()), int((d & ~sold).sum())], [int((~d & sold).sum()), int((d & sold).sum())]],
        index=["факт: продаж не было", "факт: продажа была"], columns=["прогноз: 0", "прогноз: будет продажа"])

    keys = ["precision", "recall", "specificity", "balanced accuracy", 'доля "будет продажа"']
    out["thresholds"] = pd.DataFrame({
        t: {k: classification_metrics(sold, p, p > t)[k] for k in keys} for t in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8)
    }).T.rename_axis("порог")

    out["by_horizon"] = pd.DataFrame({
        h: {"ROC AUC": roc_auc(sold[:, h - 1], p[:, h - 1]), f"ROC AUC ({SIMPLE})": roc_auc(sold[:, h - 1], simple[:, h - 1]),
            "PR AUC": average_precision(sold[:, h - 1], p[:, h - 1]),
            "balanced accuracy": classification_metrics(sold[:, h - 1], p[:, h - 1], d[:, h - 1])["balanced accuracy"],
            "доля недель с продажами": sold[:, h - 1].mean()}
        for h in range(1, p.shape[1] + 1)}).T.rename_axis("горизонт, недель")

    nz8 = simple[:, 0]
    segment = np.select([nz8 == 0, nz8 < 0.5], ["не продавалась 8 недель", "продажи < 50 % недель"], "продажи >= 50 % недель")
    rows = {}
    for name in ["не продавалась 8 недель", "продажи < 50 % недель", "продажи >= 50 % недель"]:
        m = segment == name
        mm = classification_metrics(sold[m], p[m], d[m])
        rows[name] = {"пар": int(m.sum()), "доля недель с продажами": mm["доля недель с продажами"],
                      "ROC AUC": mm["ROC AUC"], f"ROC AUC ({SIMPLE})": roc_auc(sold[m], simple[m]),
                      "precision": mm["precision"], "recall": mm["recall"], "specificity": mm["specificity"]}
    out["by_segment"] = pd.DataFrame(rows).T

    # если модель говорит 0.7, продажа должна быть примерно в 70 % таких недель
    bins = np.minimum((p * 10).astype(int), 9)
    out["calibration"] = pd.DataFrame({
        f"{b / 10:.1f}-{(b + 1) / 10:.1f}": {"недель": int((bins == b).sum()), "средняя вероятность": p[bins == b].mean(),
                                            "доля с продажей": sold[bins == b].mean()}
        for b in range(10) if (bins == b).any()}).T.rename_axis("вероятность")

    out["importance"] = pd.Series(r["model"].get_feature_importance(), index=r["model"].feature_names_,
                                  name="важность").sort_values(ascending=False)
    return {k: v.round(4) for k, v in out.items()}


def summary(threshold=0.5, folds=FOLDS, **model):
    keys = ["ROC AUC", "PR AUC", "logloss", "Brier", "balanced accuracy", "precision", "recall", "specificity", "MCC"]
    rows = {}
    for fold in folds:
        sold, p, simple, _ = fold_data(fold, **model)
        rows[(fold, "CatBoost")] = classification_metrics(sold, p, p > threshold)
        rows[(fold, SIMPLE)] = classification_metrics(sold, simple, simple >= 0.5)
    return pd.DataFrame(rows).T[keys].round(4)


def rules_comparison(fold=MAIN_FOLD, **model):
    """CatBoost против простых правил и потолка: ROC AUC, PR AUC, SMAPE и score при лучшем пороге, если там,
    где предсказана продажа, ставить обычное количество пары.

    Потолок знает, как часто пара будет продаваться в неделях теста, но не в какие именно недели. Модель, которая
    оценивает вероятность продажи для пары, лучше него не сделает: в какую неделю будет продажа, во многом случайно."""
    sold, p, _, r = fold_data(fold, **model)
    pairs, s_cut, _, truth, base = _fold_data(fold, FOLDS[fold])
    n, k = sold.shape
    first = r["rows"].iloc[:n]
    rep = lambda v: np.repeat(np.asarray(v, float)[:, None], k, axis=1)
    amount = rep(typical_amount(r["rows"], n))
    index, weeks = pd.MultiIndex.from_frame(pairs), forecast_weeks(s_cut.date.max())
    candidates = {
        f"{SIMPLE}: доля недель с продажами за 8": rep(first.nz8),
        "доля недель с продажами за 13": rep(first.nz13),
        "сглаженная доля (свежие недели важнее)": rep(first.ewm_nz),
        "по дням продаж за 8 недель: 1 - (1 - d)^7": rep(1 - (1 - first.days8) ** 7),
        "CatBoost": p,
        "потолок: знает, как часто пара продаётся в тесте": rep(sold.mean(1)),
    }
    rows = {}
    for name, prob in candidates.items():
        best = None
        for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
            m = compute_metrics(truth, to_long(pd.DataFrame(np.where(prob > t, amount, 0.0), index=index, columns=weeks)))
            s = score_minmax(m, base)
            if best is None or s < best[0]:
                best = (s, t, m["smape"])
        rows[name] = {"ROC AUC": roc_auc(sold, prob), "PR AUC": average_precision(sold, prob),
                      "лучший порог": best[1], "SMAPE": best[2], "score": best[0]}
    t = pd.DataFrame(rows).T
    lo, hi = t["score"].iloc[0], t["score"].iloc[-1]
    t["закрыто разрыва до потолка"] = (lo - t["score"]) / (lo - hi)  # 0 как простое правило, 1 как потолок
    return t.round(4)


def trees_curve(fold=MAIN_FOLD, iterations=None, every=None, **model):
    """Метрики после каждых every деревьев одной модели на iterations деревьев (по умолчанию вдвое больше, чем в PARAMS).
    По ней выбирали PARAMS["iterations"]: где ROC AUC перестаёт расти, а logloss начинает."""
    iterations = iterations or 2 * PARAMS["iterations"]
    every = every or max(iterations // 20, 1)
    pairs, s_cut, st_cut, truth, _ = _fold_data(fold, FOLDS[fold])
    weeks = forecast_weeks(s_cut.date.max())
    before = set(_CACHE)
    r = fit_predict(s_cut, st_cut, pairs, weeks, iterations=iterations, **model)
    for key in set(_CACHE) - before:
        del _CACHE[key]
    n, k = len(pairs), len(weeks)
    sold = truth.quantity.to_numpy().reshape(k, n).T > 0
    rows = {}
    stages = r["model"].staged_predict_proba(r["rows"][r["model"].feature_names_], eval_period=every)
    for i, proba in enumerate(stages, start=1):
        p = proba[:, 1].reshape(k, n).T
        pc = np.clip(p, EPS, 1 - EPS)
        rows[min(i * every, iterations)] = {"ROC AUC": roc_auc(sold, p), "PR AUC": average_precision(sold, p),
                                            "logloss": float(-np.mean(np.where(sold, np.log(pc), np.log(1 - pc))))}
    return pd.DataFrame(rows).T.rename_axis("деревьев").round(4)


def plot_curves(fold=MAIN_FOLD, **model):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sold, p, simple, _ = fold_data(fold, **model)
    s = sold.ravel()
    blue, orange, gray = "#2a78d6", "#eb6834", "#c3c2b7"
    fig, (roc, pr, cal) = plt.subplots(1, 3, figsize=(15, 4.3), layout="constrained")
    for name, score, color in [("CatBoost", p.ravel(), blue), (SIMPLE, simple.ravel(), orange)]:
        recall, fpr, precision = _curve(s, score)
        sns.lineplot(x=np.r_[0, fpr], y=np.r_[0, recall], color=color, ax=roc, estimator=None, sort=False,
                     label=f"{name}: AUC {roc_auc(s, score):.3f}")
        sns.lineplot(x=recall, y=precision, color=color, ax=pr, estimator=None, sort=False,
                     label=f"{name}: AP {average_precision(s, score):.3f}")
    roc.plot([0, 1], [0, 1], color=gray, lw=0.8)
    roc.set(title="ROC", xlabel="доля недель без продаж, где предсказана продажа",
            ylabel="доля пойманных недель с продажей")
    pr.axhline(s.mean(), color=gray, lw=0.8)
    pr.set(title="Precision-Recall", xlabel="recall", ylabel="precision")

    bins = np.minimum((p.ravel() * 10).astype(int), 9)
    calib = pd.DataFrame({"вероятность": p.ravel(), "продажа": s, "бин": bins}).groupby("бин").mean()
    cal.plot([0, 1], [0, 1], color=gray, lw=0.8)
    sns.lineplot(data=calib, x="вероятность", y="продажа", color=blue, marker="o", ax=cal, label="CatBoost")
    cal.set(title="Калибровка", xlabel="средняя вероятность в группе",
            ylabel="доля недель с продажей")
    for ax, loc in ((roc, "lower right"), (pr, "lower left"), (cal, "upper left")):
        ax.legend(frameon=False, loc=loc)
    fig.suptitle(f"Классификатор на фолде {fold}", x=0.01, ha="left")
    plt.show()


def feature_ablation(fold=MAIN_FOLD, threshold=0.5):
    """Польза каждой группы признаков: группы из FEATURES убираем, остальные добавляем.
    Польза > 0 значит, что с группой лучше. Каждая строка - отдельная модель на настройках ABLATION."""
    def metrics(features):
        before = set(_CACHE)
        sold, p, _, _ = fold_data(fold, features=features, **ABLATION)
        for key in set(_CACHE) - before:
            del _CACHE[key]
        m = classification_metrics(sold, p, p > threshold)
        return np.array([m["ROC AUC"], m["PR AUC"], -m["logloss"]])

    base = metrics(FEATURES)
    rows = {"набор FEATURES целиком": {"в наборе": "", "что проверили": "", "ROC AUC": base[0], "PR AUC": base[1],
                                       "logloss": -base[2]}}
    for group, (cols, _) in GROUPS.items():
        inside = any(c in FEATURES for c in cols)
        variant = [f for f in FEATURES if f not in cols] if inside else FEATURES + [c for c in cols if c not in FEATURES]
        m = metrics(variant)
        gain = base - m if inside else m - base
        rows[group] = {"в наборе": "да" if inside else "нет", "что проверили": "убрали" if inside else "добавили",
                       "ROC AUC": m[0], "PR AUC": m[1], "logloss": -m[2],
                       "польза для ROC AUC": gain[0], "польза для PR AUC": gain[1], "польза для logloss": gain[2]}
    t = pd.DataFrame(rows).T
    gain = t["польза для ROC AUC"].astype(float)
    t["вывод"] = np.where(gain > 0.001, "помогает", np.where(gain < -0.001, "мешает", "почти не влияет"))
    t.loc["набор FEATURES целиком", "вывод"] = ""
    return t


def error_analysis(fold=MAIN_FOLD, threshold=0.5, top=15, **model):
    """Где классификатор ошибается: доли ошибок по категориям, складам и горизонтам, плюс самые уверенные ошибки
    в обе стороны (false_sales и missed_sales)."""
    sold, p, _, r = fold_data(fold, **model)
    pairs, s_cut = _fold_data(fold, FOLDS[fold])[:2]
    n, k = p.shape
    info = pair_info(s_cut, pairs)
    look = ["nz8", "mean8", "days_since_sale", "stock", "trend"]
    pair_feats = r["rows"].iloc[:n][look].reset_index(drop=True)
    long = pd.DataFrame({c: np.repeat(info[c].to_numpy(), k) for c in ["warehouse_name", "article", "category_lvl_1"]})
    long["h"] = np.tile(np.arange(1, k + 1), n)
    long["вероятность"] = p.ravel()
    long["sold"] = sold.ravel()
    for c in look:
        long[c] = np.repeat(pair_feats[c].to_numpy(), k)
    d = long["вероятность"].to_numpy() > threshold
    long["fp"], long["fn"] = d & ~long.sold, ~d & long.sold
    errors = long.fp.sum() + long.fn.sum()

    def rates(col):
        g = long.groupby(col)
        size, s = g.size(), g[["sold", "fp", "fn"]].sum()
        return pd.DataFrame({
            "недель": size, "доля недель с продажами": s.sold / size,
            'лишние "будет", % недель без продаж': s.fp / (size - s.sold) * 100,
            "пропущенные, % недель с продажами": s.fn / s.sold * 100,
            "доля всех ошибок": (s.fp + s.fn) / errors,
        }).sort_values("доля всех ошибок", ascending=False)

    shown = ["warehouse_name", "article", "category_lvl_1", "h", "вероятность"] + look
    out = {"by_category": rates("category_lvl_1"), "by_warehouse": rates("warehouse_name"), "by_horizon": rates("h").sort_index(),
           "false_sales": long[long.fp].nlargest(top, "вероятность")[shown],
           "missed_sales": long[long.fn].nsmallest(top, "вероятность")[shown]}
    return {k: v.round(3) for k, v in out.items()}
