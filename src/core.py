"""Общий код: данные, недельная сетка, метрики, простые методы прогноза и бэктест."""
import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("HACK_DATA_DIR", PROJECT / "data"))
OUT_DIR = Path(os.environ.get("HACK_OUT_DIR", PROJECT / "outputs"))
SALES_CSV = DATA_DIR / "sales_dataset_hakaton_train.csv"
STOCK_CSV = DATA_DIR / "stock_dataset_hakaton_train.csv"
LOG_CSV = OUT_DIR / "experiments_log.csv"
SUBMISSION_CSV = OUT_DIR / "submission.csv"
ON_KAGGLE = Path("/kaggle/working").is_dir()

CUTOFF = pd.Timestamp("2026-07-26")  # для обучения можно брать данные только по эту дату включительно
FIRST_WEEK = CUTOFF + pd.Timedelta(days=1)
HORIZON = 9
KEY = ["warehouse_name", "article"]
WKEY = ["week"] + KEY
# дата отсечки (воскресенье), прогноз на 9 недель после неё
FOLDS = {
    "season2025": "2025-07-27",  # тот же сезон, что и тест, главный фолд
    "season2024": "2024-07-28",  # тот же сезон годом раньше, истории всего 30 недель
    "recent":     "2026-05-24",  # самые свежие данные, факт с 2026-05-25 по 2026-07-26
}
MAIN_FOLD = "season2025"
WEIGHTS = {"smape": 0.50, "wape": 0.20, "rmse": 0.15, "bias": 0.10, "rmsle": 0.05}
SEED = 42


_DATA = {}

def load_data(cutoff=CUTOFF):
    # таблицы общие для всех вызовов, на месте их не менять
    if cutoff not in _DATA:
        sales = pd.read_csv(SALES_CSV, parse_dates=["date"])
        stock = pd.read_csv(STOCK_CSV, parse_dates=["date"]).rename(columns={"char_article": "article"})
        if cutoff is not None:
            sales, stock = sales[sales.date <= cutoff], stock[stock.date <= cutoff]
        _DATA[cutoff] = (sales, stock)
    return _DATA[cutoff]


def universe(sales, stock=None):
    whs, arts = set(sales.warehouse_name), set(sales.article)
    if stock is not None:
        whs |= set(stock.warehouse_name)
        arts |= set(stock.article)
    return pd.MultiIndex.from_product([sorted(whs), sorted(arts)], names=KEY).to_frame(index=False)


def last_full_sunday(date):
    date = pd.Timestamp(date).normalize()
    return date - pd.Timedelta(days=(date.weekday() + 1) % 7)


def forecast_weeks(last_date):
    # с понедельника недели, следующей за неделей last_date (как в ТЗ)
    last = pd.Timestamp(last_date).normalize()
    return pd.date_range(last + pd.Timedelta(days=7 - last.weekday()), periods=HORIZON, freq="7D")


def weekly_matrix(sales, pairs, first, n):
    # пары x n недель с понедельника first, нет продаж = 0
    first = pd.Timestamp(first)
    weeks = pd.date_range(first, periods=n, freq="7D")
    s = sales[(sales.date >= first) & (sales.date < first + pd.Timedelta(weeks=n))]
    week = first + pd.to_timedelta((s.date - first).dt.days // 7 * 7, unit="D")
    q = s.groupby([s.warehouse_name, s.article, week.rename("week")]).quantity.sum()
    return q.unstack(fill_value=0.0).reindex(index=pd.MultiIndex.from_frame(pairs), columns=weeks, fill_value=0.0)


def weekly_history(sales, pairs, n):
    end = last_full_sunday(sales.date.max())
    return weekly_matrix(sales, pairs, end - pd.Timedelta(days=7 * n - 1), n)


def to_long(m):
    pairs = m.index.to_frame(index=False)
    return pd.DataFrame({
        "week": np.repeat(m.columns.values, len(pairs)),
        "warehouse_name": np.tile(pairs.warehouse_name.values, m.shape[1]),
        "article": np.tile(pairs.article.values, m.shape[1]),
        "quantity": m.to_numpy(float).T.ravel(),
    })


def daily_history(sales, pairs, days):
    end = sales.date.max()
    start = end - pd.Timedelta(days=days - 1)
    q = sales[sales.date >= start].groupby(KEY + ["date"]).quantity.sum()
    return q.unstack(fill_value=0.0).reindex(index=pd.MultiIndex.from_frame(pairs),
                                             columns=pd.date_range(start, end, freq="D"), fill_value=0.0)


def compute_metrics(truth, pred):
    # то же, что compute_all_metrics в score.py организаторов, но векторно
    df = truth[WKEY + ["quantity"]].merge(pred[WKEY + ["quantity"]], on=WKEY, how="left",
                                          suffixes=("_true", "_pred"))
    y = df.quantity_true.to_numpy(float)
    p = df.quantity_pred.fillna(0).clip(lower=0).to_numpy(float)
    codes, uniq = pd.factorize(pd.MultiIndex.from_arrays([df.warehouse_name, df.article]))
    bc = lambda w: np.bincount(codes, weights=w, minlength=len(uniq))

    den = (np.abs(y) + np.abs(p)) / 2
    r = np.divide(np.abs(y - p), den, out=np.zeros_like(y), where=den > 0)  # 0/0 считается как 0
    n = bc(np.ones_like(y))
    my, mp = bc(y) / n, bc(p) / n
    ok = my != 0  # пары без продаж в bias не участвуют
    err, total = np.abs(y - p).sum(), np.abs(y).sum()
    return {
        "smape": float((bc(r) / n).mean() * 100),
        "wape":  float(err / total * 100) if total else (0.0 if err == 0 else 100.0),
        "rmse":  float(np.sqrt(np.mean((y - p) ** 2))),
        "bias":  float((np.abs(mp[ok] - my[ok]) / my[ok] * 100).mean()) if ok.any() else 0.0,
        "rmsle": float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2))),
    }


def score_minmax(m, base):
    # 0 - идеальный прогноз, 1 - не лучше наивного
    total = 0.0
    for k, w in WEIGHTS.items():
        total += w * (min(m[k] / base[k], 1.0) if base[k] else float(m[k] > 0))
    return total


METHODS = {}

def register(name):
    def deco(fn):
        METHODS[name] = fn
        return fn
    return deco


def constant(values, weeks):
    m = np.repeat(values.to_numpy(float)[:, None], len(weeks), axis=1)
    return to_long(pd.DataFrame(m, index=values.index, columns=weeks))


@register("naive_last")  # бейзлайн организаторов: продажи за последнюю неделю
def naive_last(sales, stock, pairs, weeks):
    return constant(weekly_history(sales, pairs, 1).iloc[:, -1], weeks)

@register("zeros")
def zeros(sales, stock, pairs, weeks):
    return constant(pd.Series(0.0, index=pd.MultiIndex.from_frame(pairs)), weeks)

@register("mean")
def mean_window(sales, stock, pairs, weeks, window=8):
    return constant(weekly_history(sales, pairs, window).mean(axis=1), weeks)

@register("median")
def median_window(sales, stock, pairs, weeks, window=8):
    return constant(weekly_history(sales, pairs, window).median(axis=1), weeks)


def make_forecast(sales, stock, method="median", pairs=None, scale=1.0, zero_below=0.0, **params):
    weeks = forecast_weeks(sales.date.max())
    pairs = universe(sales, stock) if pairs is None else pairs
    pred = METHODS[method](sales, stock, pairs, weeks, **params)
    q = pred["quantity"] * scale
    if zero_below > 0:
        q = q.where(q >= zero_below, 0.0)
    pred["quantity"] = q.clip(lower=0).fillna(0.0)
    return pred


_CACHE = {}

def _fold_data(fold, cutoff):
    if fold not in _CACHE:
        sales, stock = load_data()
        cutoff = pd.Timestamp(cutoff)
        pairs = universe(sales, stock)
        s_cut, st_cut = sales[sales.date <= cutoff], stock[stock.date <= cutoff]
        truth = to_long(weekly_matrix(sales, pairs, cutoff + pd.Timedelta(days=1), HORIZON))
        base = compute_metrics(truth, make_forecast(s_cut, st_cut, "naive_last", pairs=pairs))
        _CACHE[fold] = (pairs, s_cut, st_cut, truth, base)
    return _CACHE[fold]


def backtest(method, note="", folds=FOLDS, **params):
    run = int(pd.read_csv(LOG_CSV, usecols=["run"]).run.max()) + 1 if LOG_CSV.exists() else 1
    rows = []
    for fold, cutoff in folds.items():
        pairs, s_cut, st_cut, truth, base = _fold_data(fold, cutoff)
        pred = make_forecast(s_cut, st_cut, method, pairs=pairs, **params)
        if set(pred.week) != set(truth.week):
            raise ValueError(f"{fold}: недели прогноза не совпадают с неделями факта")
        m = compute_metrics(truth, pred)
        rows.append({"run": run, "method": method, "params": str(params), "fold": fold, **m,
                     "score": score_minmax(m, base), "pred/true": pred.quantity.sum() / truth.quantity.sum(),
                     "note": note})
    res = pd.DataFrame(rows)
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(LOG_CSV, mode="a", header=not LOG_CSV.exists(), index=False)
    return res.round(3)


def journal():
    j = pd.read_csv(LOG_CSV).fillna({"note": ""})
    scores = j.pivot_table(index="run", columns="fold", values="score")
    scores = scores[[f for f in FOLDS if f in scores]].add_prefix("score_")
    main = j[j.fold == MAIN_FOLD].set_index("run")
    out = (main[["method", "params", "note"]].join(scores)
           .join(main[["smape", "wape", "rmse", "bias", "rmsle", "pred/true"]]))
    out = out[~out.duplicated(["method", "params", "note"], keep="last")]
    return out.sort_values(f"score_{MAIN_FOLD}").round(3)


def pair_info(sales, pairs):
    info = sales.drop_duplicates("article").set_index("article")[["category_lvl_1", "category_lvl_2"]]
    return pairs.reset_index(drop=True).join(info, on="article")


def error_breakdown(method, fold=MAIN_FOLD, top=15, **params):
    """Разбор ошибки метода на фолде, dict таблиц: kinds (по типам ошибок), by_category, by_warehouse, worst_pairs."""
    pairs, s_cut, st_cut, truth, _ = _fold_data(fold, FOLDS[fold])
    pred = make_forecast(s_cut, st_cut, method, pairs=pairs, **params)
    df = truth[WKEY + ["quantity"]].merge(pred[WKEY + ["quantity"]], on=WKEY, how="left", suffixes=("", "_pred"))
    df = df.merge(pair_info(s_cut, pairs), on=KEY, how="left")
    y = df.quantity.to_numpy(float)
    p = df.quantity_pred.fillna(0).clip(lower=0).to_numpy(float)
    df["quantity_pred"] = p
    den = (y + p) / 2
    df["smape"] = np.divide(np.abs(y - p), den, out=np.zeros_like(y), where=den > 0) * 100
    df["abs_err"] = np.abs(y - p)
    df["kind"] = np.select([(y == 0) & (p == 0), (y == 0) & (p > 0), (y > 0) & (p == 0)],
                           ["ноль угадан", "лишняя продажа", "пропущенная продажа"], "ошибка количества")
    n_points, total = len(df), y.sum()

    g = df.groupby("kind")
    kinds = pd.DataFrame({"недель": g.size(), "доля недель": g.size() / n_points,
                          "вклад в SMAPE": g.smape.sum() / n_points, "вклад в WAPE": g.abs_err.sum() / total * 100})
    kinds = kinds.reindex(["ноль угадан", "лишняя продажа", "пропущенная продажа", "ошибка количества"]).fillna(0)
    kinds.loc["итого"] = kinds.sum()
    out = {"kinds": kinds}

    def by(col):
        # вклады по группам в сумме дают общий SMAPE и WAPE
        g = df.groupby(col)
        return pd.DataFrame({
            "пар": g.size() / HORIZON,
            "SMAPE в группе": g.smape.mean(),
            "вклад в SMAPE": g.smape.sum() / n_points,
            "WAPE в группе": g.abs_err.sum() / g.quantity.sum() * 100,
            "вклад в WAPE": g.abs_err.sum() / total * 100,
            "прогноз / факт": g.quantity_pred.sum() / g.quantity.sum(),
        }).sort_values("вклад в SMAPE", ascending=False)

    out["by_category"], out["by_warehouse"] = by("category_lvl_1"), by("warehouse_name")
    pair = df.groupby(KEY).agg(category_lvl_1=("category_lvl_1", "first"), факт=("quantity", "sum"),
                               прогноз=("quantity_pred", "sum"), SMAPE=("smape", "mean"), ошибка=("abs_err", "sum"))
    out["worst_pairs"] = pair.sort_values("ошибка", ascending=False).head(top)
    return {k: v.round(3) for k, v in out.items()}


def check_submission(sub, pairs=None, first_week=FIRST_WEEK):
    # те же проверки, что validate_submission в score.py
    pairs = universe(*load_data()) if pairs is None else pairs
    cols = ["week", "warehouse_name", "article", "quantity"]
    missing = set(cols) - set(sub.columns)
    if missing:
        return [f"нет колонок {sorted(missing)}"]
    problems = []
    if sub[cols].isna().any().any():
        problems.append("пропуски в обязательных полях")
    q = pd.to_numeric(sub.quantity, errors="coerce")
    if q.isna().any():
        problems.append("quantity с пропусками или не числами")
    elif (q < 0).any():
        problems.append("отрицательные значения")
    weeks = pd.to_datetime(sub.week, errors="coerce")
    expected = pd.date_range(first_week, periods=HORIZON, freq="7D")
    if weeks.isna().any() or not (weeks.dt.weekday == 0).all():
        problems.append("week должны быть датами-понедельниками")
    elif not pd.DatetimeIndex(weeks.unique()).sort_values().equals(expected):
        problems.append(f"нужны ровно недели с {expected[0].date()} по {expected[-1].date()}")
    if sub.duplicated(WKEY).any():
        problems.append("дубли по week + warehouse_name + article")
    keys = set(zip(weeks, sub.warehouse_name.astype(str), sub.article.astype(str)))
    need = {(w, wh, a) for w in expected for wh, a in zip(pairs.warehouse_name, pairs.article)}
    if keys != need:
        problems.append(f"ключи не совпадают с эталоном: нет {len(need - keys)}, лишних {len(keys - need)}")
    if len(sub) != len(need):
        problems.append(f"строк {len(sub)}, нужно {len(need)}")
    return problems
