"""Медленный тест на полных данных: прогноз по последним 180 дням должен совпасть с submission/submission.csv.

Вход готовит python tests/make_api_input.py full. Запуск: pytest -m slow.
"""
import pandas as pd
import pytest

from conftest import ROOT

INPUT = ROOT / "outputs" / "api_input"
KEY = ["week", "warehouse_name", "article"]


@pytest.mark.slow
def test_last_180_days_reproduce_submission():
    if not (INPUT / "sales_last180.csv").exists():
        pytest.skip("нет outputs/api_input, сначала python tests/make_api_input.py full")
    from app.predictor import load_model

    model = load_model(ROOT / "model")
    sales = pd.read_csv(INPUT / "sales_last180.csv", dtype={"date": str})
    stock = pd.read_csv(INPUT / "stock_last180.csv", dtype={"date": str})
    forecast = model.predict(sales, stock).forecast
    submission = pd.read_csv(ROOT / "submission" / "submission.csv", parse_dates=["week"])
    both = submission.merge(forecast, on=KEY, how="outer", suffixes=("_submission", "_model"), indicator=True)
    assert len(both) == 79101 and (both["_merge"] == "both").all()
    assert (both.quantity_submission == both.quantity_model).all(), both[both.quantity_submission != both.quantity_model].head()
