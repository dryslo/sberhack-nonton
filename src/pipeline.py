"""Итоговая модель: классификатор решает, будет ли продажа на неделе, регрессор - сколько. Иначе 0."""
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

import classifier
import regressor
from core import register, to_long
from features import WINDOW, get_panel

CAP = 20  # от этого количества целое берём простым округлением, на SMAPE это почти не влияет


def pair_dispersion(weekly, min_weeks=4):
    # параметр r отрицательного биномиального для (y - 1) по ненулевым неделям пары за последние WINDOW недель.
    # Если недель мало или разброс не больше пуассоновского, r = inf (то есть Пуассон)
    w = weekly[:, -WINDOW:]
    z = np.where(w > 0, w - 1.0, np.nan)
    n = np.isfinite(z).sum(1)
    m = np.nansum(z, 1) / np.maximum(n, 1)
    v = np.nansum((z - m[:, None]) ** 2, 1) / np.maximum(n - 1, 1)
    r = np.where((n >= min_weeks) & (v > m), m ** 2 / np.maximum(v - m, 1e-9), np.inf)
    return np.maximum(r, 0.2)


def smape_choice(q, r=None):
    # целое с наименьшим ожидаемым SMAPE и сама ошибка (от 0 до 2), если продажи = 1 + Z,
    # где Z пуассоновская со средним q - 1 (или отрицательная биномиальная, если задан r)
    k = np.maximum(np.round(q), 1.0)
    err = 0.8 * np.sqrt(np.maximum(q - 1.0, 0.0)) / np.maximum(q, 1.0)  # для больших q нормальное приближение
    small = q < CAP
    ys = np.arange(1, 4 * CAP + 1)
    lam = np.clip(q[small] - 1.0, 1e-9, None)[:, None]
    pmf = poisson.pmf(ys - 1, lam)
    if r is not None:
        rr = np.broadcast_to(r, q.shape)[small][:, None]
        nb = np.isfinite(rr[:, 0])
        pmf[nb] = nbinom.pmf(ys - 1, rr[nb], rr[nb] / (rr[nb] + lam[nb]))
    pmf = pmf / pmf.sum(1, keepdims=True)
    loss = np.stack([(pmf * 2 * np.abs(ys - j) / (ys + j)).sum(1) for j in range(1, CAP + 1)], axis=1)
    k[small], err[small] = loss.argmin(1) + 1.0, loss.min(1)
    return k, err


def to_integer(q, how):
    # продают обычно 1-2 штуки в неделю, и дробный прогноз (1.6 при факте 1) проигрывает целому в SMAPE
    if how == "round":
        return np.maximum(np.round(q), 1.0)
    if how == "floor":
        return np.maximum(np.floor(q), 1.0)
    return smape_choice(q)[0]


MEMBER_KEYS = ("clf", "reg", "weight", "clf_weight", "reg_weight")


def learn_integer_map(q, y, edges, min_weeks=30):
    # для каждой корзины прогноза [edges[i], edges[i+1]) целое с наименьшим средним SMAPE по неделям с продажей.
    # Если недель в корзине мало, берём smape_choice. Целое не убывает с ростом прогноза
    sold = y > 0
    q, y = q[sold], y[sold]
    ks = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (q >= lo) & (q < hi)
        if inside.sum() < min_weeks:
            ks.append(float(smape_choice(np.array([[(lo + hi) / 2]]))[0][0, 0]))
            continue
        cand = np.arange(max(1, int(np.floor(lo)) - 1), int(np.ceil(hi)) + 2)
        loss = [np.mean(2 * np.abs(y[inside] - k) / (y[inside] + k)) for k in cand]
        ks.append(float(cand[int(np.argmin(loss))]))
    return {"edges": [float(e) for e in edges], "k": list(np.maximum.accumulate(ks))}


def apply_integer_map(q, int_map):
    edges, ks = np.asarray(int_map["edges"]), np.asarray(int_map["k"])
    out = smape_choice(q)[0]
    inside = (q >= edges[0]) & (q < edges[-1])
    out[inside] = ks[np.searchsorted(edges, q[inside], side="right") - 1]
    return out


def member_settings(m, model):
    # общие ключи участника плюс свои для "clf" или "reg"
    return {**{k: v for k, v in m.items() if k not in MEMBER_KEYS}, **m.get(model, {})}


def member_weights(members, model):
    return [m.get(f"{model}_weight", m.get("weight", 1.0)) for m in members]


def by_horizon(threshold, n_weeks):
    # порог числом или парой (недели 1-4, недели 5-9)
    if np.isscalar(threshold):
        return np.full(n_weeks, float(threshold))
    near, far = threshold
    return np.where(np.arange(1, n_weeks + 1) <= 4, near, far)


def boost_factor(sales, pairs, boost):
    # boost вида {"category_lvl_1_4": 1.15}
    cat = pairs.article.map(sales.drop_duplicates("article").set_index("article").category_lvl_1)
    return cat.map(boost).fillna(1.0).to_numpy()


def decide(p, q, weekly=None, factor=None, threshold=0.5, integer=None, shift=None, amount_scale=1.0, dispersion=None):
    # 0 или k по вероятности p и количеству q (пары x недели). Используется и в бэктесте, и в Forecaster
    q = q * amount_scale
    if factor is not None:
        q = q * factor[:, None]
    r = pair_dispersion(weekly)[:, None] if dispersion == "pair" else None
    if shift is not None:
        k, err = smape_choice(q, r)
        return np.where(p > 2 / (4 - err) + shift, k, 0.0)
    if isinstance(integer, dict):
        q = apply_integer_map(q, integer)
    elif integer:
        q = smape_choice(q, r)[0] if integer == "smape" else to_integer(q, integer)
    return np.where(p > by_horizon(threshold, q.shape[1]), q, 0.0)


@register("pipeline")
def pipeline(sales, stock, pairs, weeks, threshold=0.5, boost=None, integer=None, shift=None, ensemble=None,
             amount_scale=1.0, dispersion=None, **regressor_params):
    # threshold: порог продажи, число или пара (недели 1-4, 5-9)
    # integer: "round", "floor", "smape" или соответствие из learn_integer_map
    # amount_scale, boost: множители количества (общий и по категории), нули не меняют
    # dispersion="pair": для целого брать разброс самой пары вместо Пуассона
    # shift: вместо threshold и integer свой порог у каждой пары-недели: k из smape_choice ставим,
    #   если p > 2 / (4 - err) + shift, где err - ожидаемый SMAPE этого k при продаже
    #   (при shift = 0 это то же, что p * (2 - err) > (1 - p) * 2)
    # ensemble: список участников, например [{}, {"random_seed": 7}, {"library": "lightgbm"}].
    #   Вероятности усредняются, количества усредняются в логарифмах. Вес: "weight" или "clf_weight" / "reg_weight"
    members = ensemble or [{}]
    p = np.average([classifier.fit_predict(sales, stock, pairs, weeks, **member_settings(m, "clf"))["p"] for m in members],
                   axis=0, weights=member_weights(members, "clf"))
    q = np.expm1(np.average([np.log1p(regressor.fit_predict(sales, stock, pairs, weeks, **regressor_params,
                                                            **member_settings(m, "reg"))["amount"]) for m in members],
                            axis=0, weights=member_weights(members, "reg")))
    values = decide(p, q, weekly=get_panel(sales, stock, pairs).sales if dispersion == "pair" else None,
                    factor=boost_factor(sales, pairs, boost) if boost else None, threshold=threshold, integer=integer,
                    shift=shift, amount_scale=amount_scale, dispersion=dispersion)
    return to_long(pd.DataFrame(values, index=pd.MultiIndex.from_frame(pairs), columns=weeks))
