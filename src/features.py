"""Признаки. Строка - пара x неделя отсчёта T x горизонт h, признаки считаются только по данным по неделю T включительно.

Обучающая выборка и строки для прогноза строятся одним кодом. Какие признаки берёт каждая модель, задано
списками FEATURES в classifier.py и regressor.py.
"""
import numpy as np
import pandas as pd
import holidays

from core import HORIZON, KEY, last_full_sunday, weekly_matrix

WINDOW = 24  # в контейнер приходит ~180 дней, поэтому почти все признаки по 24 неделям
YEAR = 52  # признаки прошлого года, при 180 днях истории они пустые
PAD = WINDOW + YEAR + 2
MIN_HISTORY = 8
STOCK_DAYS = 28
SEASON_WEEKS = (31, 39)  # недели года, на которые приходится тест (27.07-27.09)
CAT_FEATURES = ["warehouse_name", "category_lvl_1", "category_lvl_2", "characteristic"]

# группа: (колонки, описание). T - последняя неделя истории, целевая неделя T + h
GROUPS = {
    "лаги": (["lag1", "lag2", "lag3", "lag4"], "продажи пары за 1-4 последние недели"),
    "уровень": (["mean4", "mean8", "mean13", "mean24", "med8", "max8"],
                "средние за 4/8/13/24 недели, медиана и максимум недельных продаж за 8 недель"),
    "частота": (["nz4", "nz8", "nz13", "nz24", "ewm_nz", "days4", "days8"],
                "доля недель с продажами (за 4-24 недели и сглаженная, где свежие недели весят больше) и доля дней с продажами (4/8 недель)"),
    "обычное количество": (["typ8", "typ24", "qty_per_day8", "max_day8"],
                           "сколько продают, когда продают: за неделю (8/24 недели), за день продаж, максимум за день"),
    "категория на складе": (["whcat_trend", "whcat2_trend", "cat_nz4", "whcat_nz4", "cat_nz_trend"],
                            "4 недели к 24 у пары склада и категории 1-го и 2-го уровня; доля пар категории с продажами "
                            "за 4 недели, всего и на складе; она же к доле за 24 недели (категория разогревается или остывает)"),
    "всплески": (["spikes16", "spike_share16", "bulk24"],
                 "недель-всплесков за 16 недель (от 5 шт. и в 3+ раза выше обычной недели пары за 8 недель до неё), их доля "
                 "в штуках; доля штук из крупных строк (от 10 шт. за день в одном городе и департаменте) за 24 недели"),
    "давность": (["since_last", "since_first", "days_since_sale", "hist_weeks"],
                 "недель с последней и первой продажи, дней с последней продажи, недель истории"),
    "тренды": (["trend", "art_trend", "wh_trend", "cat_trend", "cat2_trend"],
               "4 недели к 24: у пары, у товара на всех складах, у всего склада, у категории 1-го и 2-го уровня"),
    "товар на всех складах": (["art_mean4", "art_nz4"], "продажи того же товара по всем складам и доля складов с продажами"),
    "остатки": (["stock", "stock_min7", "stock_mean28", "stock_le0_28", "stock_days28", "days_since_stock"],
                "остаток на начало воскресенья недели T, минимум за 7 дней, средний за 28, доля дней с остатком <= 0, дней с записью, "
                "дней с последнего положительного остатка"),
    "появление продажи": (["restock_days", "restocked7", "stock_change7", "stock_max28", "art_nz1"],
                          "дней с последнего пополнения (остаток был <= 0 и стал > 0), было ли оно за неделю, изменение "
                          "остатка за неделю, максимум остатка за 28 дней, доля складов с продажей товара на прошлой неделе"),
    "цена и скидка": (["price8", "price_trend", "discount8"], "средняя цена за штуку за 8 недель, она же к 24 неделям, доля скидки"),
    "динамика скидки": (["discount_trend"], "доля скидки за 8 недель к 24 неделям: скидку углубляют или сворачивают"),
    "география": (["cities8", "divisions8"], "в скольких городах и департаментах покупают за неделю с продажами"),
    "категории": (CAT_FEATURES, "склад и атрибуты товара"),
    "календарь": (["h", "week_of_year", "month", "sep1", "holidays"],
                  "горизонт и календарь целевой недели: номер недели года, месяц, 1 сентября, число официальных выходных"),
    "прошлый год": (["ly_nz3", "ly_mean3", "ly_lift", "art_ly_lift", "cat_ly_lift"],
                    "та же неделя год назад и соседние с ней: доля недель с продажами и среднее у пары; подъём к 8 неделям "
                    "по неделю отсчёта год назад у пары, товара на всех складах и категории. Если истории за год нет, пусто"),
}


def select(*groups, extra=()):
    cols = [c for g in groups for c in GROUPS[g][0]] + list(extra)
    return list(dict.fromkeys(cols))


def overview(*selected):
    # selected: пары (имя набора, список признаков)
    rows = {}
    for group, (cols, text) in GROUPS.items():
        rows[group] = {"признаки": ", ".join(cols), "что значат": text}
        for name, features in selected:
            n = sum(c in features for c in cols)
            rows[group][name] = "все" if n == len(cols) else ("нет" if n == 0 else f"{n} из {len(cols)}")
    return pd.DataFrame(rows).T


def typical_amount(rows, n_pairs):
    # сколько пара продаёт за неделю, когда продаёт: по 8 неделям, если нет, то по WINDOW, иначе 1
    first = rows.iloc[:n_pairs]
    return first.typ8.fillna(first[f"typ{WINDOW}"]).fillna(1.0).to_numpy(float)


def _code(values):
    # category_lvl_2_17 -> 17, так код не зависит от того, какие значения попали в данные
    s = pd.Series(np.asarray(values, dtype=object)).astype(str).str.rsplit("_", n=1).str[-1]
    return pd.to_numeric(s, errors="coerce").fillna(-1).astype(np.int32).to_numpy()


def _matrix(index, df, cols, values, n_cols, fill):
    rows = index.get_indexer(pd.MultiIndex.from_frame(df[KEY]))
    ok = rows >= 0
    m = np.full((len(index), n_cols), fill, dtype=float)
    if np.isnan(fill):
        m[rows[ok], cols[ok]] = values[ok]
    else:
        np.add.at(m, (rows[ok], cols[ok]), values[ok])
    return m


_PANELS = {}


def get_panel(sales, stock, pairs):
    key = (sales.date.max(), len(sales), len(stock), len(pairs))
    if key not in _PANELS:
        _PANELS[key] = Panel(sales, stock, pairs)
    return _PANELS[key]


class Panel:
    """Матрицы пары x недели (для остатков пары x дни), из них считаются признаки на любую неделю T."""

    def __init__(self, sales, stock, pairs):
        first = sales.date.min().normalize()
        self.start = first + pd.Timedelta(days=(7 - first.weekday()) % 7)  # первый полный понедельник
        self.end = last_full_sunday(sales.date.max())
        self.n = (self.end - self.start).days // 7 + 1
        self.pairs = pairs.reset_index(drop=True)
        index = pd.MultiIndex.from_frame(self.pairs)
        pad = np.full((len(index), PAD), np.nan)

        def weekly(df, col, fill):
            return np.hstack([pad, _matrix(index, df, df.w.to_numpy(), df[col].to_numpy(float), self.n, fill)])

        self.sales = np.hstack([pad, weekly_matrix(sales, self.pairs, self.start, self.n).to_numpy(float)])
        s = sales[(sales.date >= self.start) & (sales.date <= self.end)]
        day = (s.date - self.start).dt.days.to_numpy()
        s = s.assign(day=day, w=day // 7,
                     money=np.exp(s.sales_sum),  # суммы даны в логарифмах, по ТЗ перед агрегацией нужен np.exp
                     disc=np.exp(s.discount_sum).fillna(0.0))
        daily = s.groupby(KEY + ["day", "w"], as_index=False).quantity.sum()
        by_week = daily.groupby(KEY + ["w"], as_index=False).agg(days=("day", "size"), max_day=("quantity", "max"),
                                                                 last_day=("day", "max"))
        self.days = weekly(by_week, "days", 0.0)
        self.max_day = weekly(by_week, "max_day", 0.0)
        self.last_day = weekly(by_week, "last_day", np.nan)
        extra = s.groupby(KEY + ["w"], as_index=False).agg(money=("money", "sum"), disc=("disc", "sum"),
                                                           cities=("cities_name", "nunique"),
                                                           divisions=("composite_division", "nunique"))
        self.money, self.disc = weekly(extra, "money", 0.0), weekly(extra, "disc", 0.0)
        self.cities, self.divisions = weekly(extra, "cities", 0.0), weekly(extra, "divisions", 0.0)
        self.bulk = weekly(s[s.quantity >= 10].groupby(KEY + ["w"], as_index=False).quantity.sum(), "quantity", 0.0)

        # всплеск: неделя от 5 шт. и в 3+ раза выше среднего ненулевых недель за 8 недель до неё (если их было 3+)
        q = np.nan_to_num(self.sales)
        zero = np.zeros((len(q), 1))
        csum, cnz = np.hstack([zero, q.cumsum(1)]), np.hstack([zero, (q > 0).cumsum(1)])
        j = np.arange(q.shape[1])
        lo = np.maximum(j - 8, 0)
        n8 = cnz[:, j] - cnz[:, lo]
        typical = np.where(n8 >= 3, (csum[:, j] - csum[:, lo]) / np.maximum(n8, 1), np.nan)
        self.spike = q >= np.maximum(5, 3 * typical)

        self.day0 = self.start - pd.Timedelta(days=STOCK_DAYS)
        st = stock[(stock.date >= self.day0) & (stock.date <= self.end)]
        st = st.groupby(KEY + ["date"], as_index=False)["stock"].sum()
        sday = (st.date - self.day0).dt.days.to_numpy()
        self.stock = _matrix(index, st, sday, st["stock"].to_numpy(float), (self.end - self.day0).days + 1, np.nan)

        info = sales.drop_duplicates("article").set_index("article")[CAT_FEATURES[1:]]
        cats = self.pairs.join(info, on="article")
        self.cats = {c: _code(cats[c]) for c in CAT_FEATURES}
        self.wh = pd.factorize(self.pairs.warehouse_name, sort=True)[0]
        self.art = pd.factorize(self.pairs.article, sort=True)[0]
        self.cat1 = pd.factorize(self.cats["category_lvl_1"], sort=True)[0]
        self.cat2 = pd.factorize(self.cats["category_lvl_2"], sort=True)[0]
        # рост категории сильно разный по складам (у 1_4 от 1.4 до 3.1 раза), поэтому отдельно склад + категория
        self.whcat1 = pd.factorize(self.wh * (self.cat1.max() + 1) + self.cat1)[0]
        self.whcat2 = pd.factorize(self.wh * (self.cat2.max() + 1) + self.cat2)[0]
        self.holidays = set(holidays.RU(years=range(self.start.year, self.end.year + 2)))

    def features(self, t, horizons, target=False, columns=None):
        """Признаки на неделю отсчёта t для горизонтов horizons, строки идут горизонт за горизонтом."""
        c = t + PAD
        win, win8 = slice(c - WINDOW + 1, c + 1), slice(c - 7, c + 1)
        w = self.sales[:, win]
        avail = ~np.isnan(w)
        sold = (w > 0) & avail
        hw = min(t + 1, WINDOW)

        f = {f"lag{k}": w[:, -k] for k in (1, 2, 3, 4)}
        for k in (4, 8, 13, WINDOW):
            f[f"mean{k}"] = np.nanmean(w[:, -k:], 1)
            f[f"nz{k}"] = sold[:, -k:].sum(1) / avail[:, -k:].sum(1)
        decay = 0.5 ** (np.arange(WINDOW)[::-1] / 4)
        f["ewm_nz"] = (sold * decay).sum(1) / (avail * decay).sum(1)
        f["med8"] = np.nanmedian(w[:, -8:], 1)
        f["max8"] = np.nanmax(w[:, -8:], 1)
        for k in (8, WINDOW):
            cnt = sold[:, -k:].sum(1)
            f[f"typ{k}"] = np.where(cnt > 0, np.nansum(w[:, -k:], 1) / np.maximum(cnt, 1), np.nan)
        q8, q_all = np.nansum(w[:, -8:], 1), np.nansum(w, 1)
        days8 = self.days[:, win8].sum(1)
        f["qty_per_day8"] = np.where(days8 > 0, q8 / np.maximum(days8, 1), np.nan)
        f["max_day8"] = self.max_day[:, win8].max(1)
        any_sold = sold.any(1)
        f["since_last"] = np.where(any_sold, np.argmax(sold[:, ::-1], 1), WINDOW)
        f["since_first"] = np.where(any_sold, WINDOW - 1 - np.argmax(sold, 1), np.nan)
        last_day = np.fmax.reduce(self.last_day[:, win], axis=1)
        f["days_since_sale"] = np.where(np.isnan(last_day), 7 * WINDOW, 7 * t + 6 - last_day)
        f["hist_weeks"] = np.full(len(w), hw)
        f["trend"] = f["mean4"] / np.where(f[f"mean{WINDOW}"] > 0, f[f"mean{WINDOW}"], np.nan)
        dd = self.days[:, win8]
        f["days4"] = dd[:, -4:].sum(1) / 28
        f["days8"] = days8 / 56

        # товар на всех складах, склад целиком, категории
        s4 = np.nansum(w[:, -4:], 1)
        n_art = np.bincount(self.art)
        art4, art_all = np.bincount(self.art, s4), np.bincount(self.art, q_all)
        f["art_mean4"] = (art4 / (4 * n_art))[self.art]
        f["art_nz4"] = (np.bincount(self.art, (s4 > 0).astype(float)) / n_art)[self.art]
        f["art_nz1"] = (np.bincount(self.art, (np.nan_to_num(w[:, -1]) > 0).astype(float)) / n_art)[self.art]
        f["art_trend"] = ((art4 / 4) / np.where(art_all > 0, art_all / hw, np.nan))[self.art]
        wh4, wh_all = np.bincount(self.wh, s4), np.bincount(self.wh, q_all)
        f["wh_trend"] = ((wh4 / 4) / np.where(wh_all > 0, wh_all / hw, np.nan))[self.wh]
        for name, codes in (("cat_trend", self.cat1), ("cat2_trend", self.cat2)):
            c4, c_all = np.bincount(codes, s4), np.bincount(codes, q_all)
            f[name] = ((c4 / 4) / np.where(c_all > 0, c_all / hw, np.nan))[codes]
        for name, codes in (("whcat_trend", self.whcat1), ("whcat2_trend", self.whcat2)):
            c4, c_all = np.bincount(codes, s4), np.bincount(codes, q_all)
            f[name] = ((c4 / 4) / np.where(c_all > 0, c_all / hw, np.nan))[codes]
        sold4, sold_all = (s4 > 0).astype(float), (q_all > 0).astype(float)
        for name, codes in (("cat_nz4", self.cat1), ("whcat_nz4", self.whcat1)):
            f[name] = (np.bincount(codes, sold4) / np.bincount(codes))[codes]
        f["cat_nz_trend"] = (np.bincount(self.cat1, sold4) / np.maximum(np.bincount(self.cat1, sold_all), 1))[self.cat1]

        sp, w16 = self.spike[:, c - 15: c + 1], w[:, -16:]
        q16 = np.nansum(w16, 1)
        f["spikes16"] = sp.sum(1)
        f["spike_share16"] = np.where(q16 > 0, np.nansum(np.where(sp, w16, 0.0), 1) / np.maximum(q16, 1e-9), np.nan)
        f["bulk24"] = np.where(q_all > 0, np.nansum(self.bulk[:, win], 1) / np.maximum(q_all, 1e-9), np.nan)

        # цена и скидка: суммы в непонятной шкале, поэтому смотрим на отношения
        money8, money_all = np.nansum(self.money[:, win8], 1), np.nansum(self.money[:, win], 1)
        price8 = np.where(q8 > 0, money8 / np.maximum(q8, 1e-9), np.nan)
        price_all = np.where(q_all > 0, money_all / np.maximum(q_all, 1e-9), np.nan)
        f["price8"] = price8
        f["price_trend"] = price8 / np.where(price_all > 0, price_all, np.nan)
        f["discount8"] = np.where(money8 > 0, np.nansum(self.disc[:, win8], 1) / np.maximum(money8, 1e-9), np.nan)
        disc_all = np.where(money_all > 0, np.nansum(self.disc[:, win], 1) / np.maximum(money_all, 1e-9), np.nan)
        f["discount_trend"] = f["discount8"] / np.where(disc_all > 0, disc_all, np.nan)

        cnt8 = sold[:, -8:].sum(1)
        f["cities8"] = np.where(cnt8 > 0, np.nansum(self.cities[:, win8], 1) / np.maximum(cnt8, 1), np.nan)
        f["divisions8"] = np.where(cnt8 > 0, np.nansum(self.divisions[:, win8], 1) / np.maximum(cnt8, 1), np.nan)

        # остатки за STOCK_DAYS дней по воскресенье T включительно
        T = self.start + pd.Timedelta(days=7 * t + 6)
        d = (T - self.day0).days
        s = self.stock[:, d - STOCK_DAYS + 1: d + 1]
        rec = ~np.isnan(s)
        nrec = rec.sum(1)
        pos = (s > 0) & rec
        f["stock"] = s[:, -1]
        f["stock_min7"] = np.fmin.reduce(s[:, -7:], axis=1)
        f["stock_mean28"] = np.where(nrec > 0, np.nansum(s, 1) / np.maximum(nrec, 1), np.nan)
        f["stock_le0_28"] = np.where(nrec > 0, ((s <= 0) & rec).sum(1) / np.maximum(nrec, 1), np.nan)
        f["stock_days28"] = nrec
        f["days_since_stock"] = np.where(pos.any(1), np.argmax(pos[:, ::-1], 1), np.where(nrec > 0, STOCK_DAYS, np.nan))
        # пополнение: остаток был <= 0 и стал > 0, нет записи считаем как нет остатка
        s0 = np.nan_to_num(s, nan=0.0)
        up = (s0[:, 1:] > 0) & (s0[:, :-1] <= 0)
        f["restock_days"] = np.where(up.any(1), np.argmax(up[:, ::-1], 1), np.nan)
        f["restocked7"] = (f["restock_days"] <= 6).astype(float)
        f["stock_change7"] = (s0[:, -1] - s0[:, -8]) / np.maximum(np.abs(f["stock_mean28"]), 1.0)
        f["stock_max28"] = np.fmax.reduce(s, axis=1)

        # до сих пор по одному значению на пару, дальше повторяем на все горизонты и добавляем календарь
        k, n_pairs = len(horizons), len(self.pairs)
        out = pd.DataFrame({name: np.tile(np.asarray(v, dtype=np.float32), k) for name, v in f.items()})
        for name, codes in self.cats.items():
            out[name] = np.tile(codes, k)
        mondays = [T + pd.Timedelta(days=1 + 7 * (h - 1)) for h in horizons]
        week_days = [pd.date_range(m, periods=7) for m in mondays]
        out["h"] = np.repeat(np.asarray(horizons, dtype=np.float32), n_pairs)
        out["week_of_year"] = np.repeat([m.isocalendar().week for m in mondays], n_pairs).astype(np.float32)
        out["month"] = np.repeat([(m + pd.Timedelta(days=3)).month for m in mondays], n_pairs).astype(np.float32)
        out["sep1"] = np.repeat([float(any(x.month == 9 and x.day == 1 for x in wd)) for wd in week_days], n_pairs)
        out["holidays"] = np.repeat([float(sum(x.date() in self.holidays for x in wd)) for wd in week_days], n_pairs)

        # прошлый год: целевая неделя год назад с соседними и 8 недель до недели отсчёта год назад
        def lift(now, before):
            return (now / 3) / np.where(before > 0, before / 8, np.nan)

        last_year = {name: [] for name in GROUPS["прошлый год"][0]}
        for h in horizons:
            ly = c + h - YEAR
            cur, prev = self.sales[:, ly - 1: ly + 2], self.sales[:, ly - h - 7: ly - h + 1]
            if np.isnan(cur[0]).any() or np.isnan(prev[0]).any():
                for parts in last_year.values():
                    parts.append(np.full(n_pairs, np.nan))
                continue
            cs, ps = cur.sum(1), prev.sum(1)
            last_year["ly_nz3"].append((cur > 0).mean(1))
            last_year["ly_mean3"].append(cs / 3)
            last_year["ly_lift"].append(lift(cs, ps))
            last_year["art_ly_lift"].append(lift(np.bincount(self.art, cs)[self.art], np.bincount(self.art, ps)[self.art]))
            last_year["cat_ly_lift"].append(lift(np.bincount(self.cat1, cs)[self.cat1], np.bincount(self.cat1, ps)[self.cat1]))
        for name, parts in last_year.items():
            out[name] = np.concatenate(parts).astype(np.float32)

        if columns is not None:
            out = out[list(columns)]
        if target:
            out["target"] = np.concatenate([self.sales[:, c + h] for h in horizons]).astype(np.float32)
        return out

    def training_set(self, origin_step=1, only_sold=False, columns=None, half_life=None, season_boost=0.0):
        # недели отсчёта идут от последней возможной назад с шагом origin_step, горизонты только с известным фактом.
        # half_life: вес строки 0.5 ** (недель до последней недели отсчёта / half_life).
        # season_boost: строки с целевой неделей в SEASON_WEEKS весят в 1 + season_boost раз больше
        parts = []
        last = self.n - 2
        for t in range(last, MIN_HISTORY - 2, -origin_step):
            hs = [h for h in range(1, HORIZON + 1) if t + h <= self.n - 1]
            part = self.features(t, hs, target=True, columns=columns)
            if half_life or season_boost:
                w = 0.5 ** ((last - t) / half_life) if half_life else 1.0
                season = [1.0 + season_boost * (SEASON_WEEKS[0] <= (self.start + pd.Timedelta(days=7 * (t + h))).isocalendar().week
                                                <= SEASON_WEEKS[1]) for h in hs]
                part["weight"] = (w * np.repeat(np.asarray(season), len(self.pairs))).astype(np.float32)
            parts.append(part[part.target > 0] if only_sold else part)
        return pd.concat(parts, ignore_index=True)

    def forecast_rows(self, weeks):
        # неделя отсчёта - последняя полная неделя, горизонт - номер недели прогноза после неё
        return self.features(self.n - 1, [((w - self.end).days - 1) // 7 + 1 for w in weeks])
