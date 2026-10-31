import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample():
    # data/sample_*.csv в формате организаторов, 200 дней
    return (ROOT / "data" / "sample_sales.csv").read_bytes(), (ROOT / "data" / "sample_stock.csv").read_bytes()
