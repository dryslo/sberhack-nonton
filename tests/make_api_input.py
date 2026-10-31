"""Входные файлы для API в формате организаторов из data/*.csv (в stock товар называется char_article, как в ТЗ).

    python tests/make_api_input.py sample  # ~30 товаров на всех складах за 200 дней -> data/sample_*.csv
    python tests/make_api_input.py full    # все товары за последние 180 дней -> outputs/api_input/

sample нужен для README, test_api.sh и тестов. full нужен, чтобы проверить, что прогноз контейнера совпадает
с submission/submission.csv. Берутся данные только по 2026-07-26 включительно.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SALES, STOCK = ROOT / "data" / "sales_dataset_hakaton_train.csv", ROOT / "data" / "stock_dataset_hakaton_train.csv"
LAST = "2026-07-26"
CHUNK = 500_000


def read(path, first, articles=None):
    # файлы большие, читаем кусками
    parts = []
    for part in pd.read_csv(path, chunksize=CHUNK, dtype=str):
        keep = (part.date >= first) & (part.date <= LAST)
        if articles is not None:
            keep &= part.article.isin(articles)
        parts.append(part[keep])
    return pd.concat(parts, ignore_index=True)


def pick_articles(first, n=30):
    # товары с разной частотой продаж, чтобы пример был похож на настоящие данные
    volume = pd.Series(dtype=float)
    cats = {}
    for part in pd.read_csv(SALES, chunksize=CHUNK, usecols=["date", "article", "category_lvl_1", "quantity"]):
        part = part[(part.date >= first) & (part.date <= LAST)]
        volume = volume.add(part.groupby("article").quantity.sum(), fill_value=0)
        cats.update(part.drop_duplicates("article").set_index("article").category_lvl_1.to_dict())
    ranked = volume.sort_values(ascending=False).index.tolist()
    step = max(len(ranked) // n, 1)
    chosen = ranked[::step][:n]
    for cat in ("category_lvl_1_5", "category_lvl_1_4"):
        if not any(cats.get(a) == cat for a in chosen):
            chosen[-1] = next(a for a in ranked if cats.get(a) == cat)
    return sorted(chosen)


def write(sales, stock, out_sales, out_stock):
    out_sales.parent.mkdir(parents=True, exist_ok=True)
    sales.sort_values(["date", "warehouse_name", "article"]).to_csv(out_sales, index=False)
    stock.rename(columns={"article": "char_article"}).sort_values(["date", "warehouse_name", "char_article"]).to_csv(out_stock, index=False)
    print(f"{out_sales.relative_to(ROOT)}: {len(sales)} строк, {sales.date.min()} - {sales.date.max()}, "
          f"товаров {sales.article.nunique()}, складов {sales.warehouse_name.nunique()}")
    print(f"{out_stock.relative_to(ROOT)}: {len(stock)} строк")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
    if mode == "sample":
        first = str((pd.Timestamp(LAST) - pd.Timedelta(days=199)).date())
        articles = pick_articles(first)
        write(read(SALES, first, articles), read(STOCK, first, articles), ROOT / "data" / "sample_sales.csv",
              ROOT / "data" / "sample_stock.csv")
    elif mode == "full":
        first = str((pd.Timestamp(LAST) - pd.Timedelta(days=179)).date())
        write(read(SALES, first), read(STOCK, first), ROOT / "outputs" / "api_input" / "sales_last180.csv",
              ROOT / "outputs" / "api_input" / "stock_last180.csv")
    else:
        sys.exit("режим: sample или full")
