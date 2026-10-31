"""Проверка входных CSV по разделу 10.4 ТЗ."""
import io

import pandas as pd

SALES_COLUMNS = ["date", "warehouse_name", "article", "quantity", "sales_sum", "discount_sum", "composite_division",
                 "cities_name", "category_lvl_1", "category_lvl_2", "category_lvl_3", "characteristic"]
STOCK_COLUMNS = ["date", "warehouse_name", "char_article", "stock"]
MIN_HISTORY_DAYS = 180


class InputError(Exception):
    def __init__(self, status, code, errors):
        super().__init__(errors[0])
        self.status, self.code, self.errors = status, code, errors


def read_csv(raw: bytes, name: str) -> pd.DataFrame:
    if not raw or not raw.strip():
        raise InputError(400, "EMPTY_FILE", [f"Файл {name} пустой"])
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype={"date": str})
    except Exception as e:
        raise InputError(400, "BAD_CSV", [f"Файл {name} не читается как CSV: {e}"])
    if df.empty:
        raise InputError(400, "EMPTY_FILE", [f"В файле {name} нет строк, только заголовок"])
    return df


def _history(df, name, errors):
    try:
        dates = pd.to_datetime(df["date"], format="%Y-%m-%d")
    except (ValueError, TypeError):
        errors.append(f"{name}: даты должны быть в формате YYYY-MM-DD")
        return
    days = (dates.max() - dates.min()).days + 1
    if days < MIN_HISTORY_DAYS:
        errors.append(f"{name}: история {days} дн. (с {dates.min().date()} по {dates.max().date()}), "
                      f"нужно не меньше {MIN_HISTORY_DAYS}")


def validate(sales: pd.DataFrame, stock: pd.DataFrame) -> None:
    errors = []
    missing = [c for c in SALES_COLUMNS if c not in sales.columns]
    if missing:
        errors.append(f"sales: нет колонок {', '.join(missing)}")
    # в выданных файлах товар в stock называется article, в ТЗ - char_article: принимаем оба
    missing = [c for c in STOCK_COLUMNS if c not in stock.columns and not (c == "char_article" and "article" in stock.columns)]
    if missing:
        errors.append(f"stock: нет колонок {', '.join(missing)}")
    if errors:
        raise InputError(422, "VALIDATION_ERROR", errors)

    _history(sales, "sales", errors)
    _history(stock, "stock", errors)
    quantity = pd.to_numeric(sales["quantity"], errors="coerce")
    if quantity.isna().any():
        errors.append(f"sales: quantity не число в {int(quantity.isna().sum())} строках")
    elif (quantity < 0).any():
        errors.append(f"sales: отрицательные quantity в {int((quantity < 0).sum())} строках")
    # отрицательный stock - резерв, это не ошибка
    if pd.to_numeric(stock["stock"], errors="coerce").isna().any():
        errors.append("stock: stock должен быть числом")
    if errors:
        raise InputError(422, "VALIDATION_ERROR", errors)
