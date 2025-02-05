import pytest
import pandas as pd
from project import Firstplot, Secondplot

def test_Firstplot():
    sample_data = pd.DataFrame({
        "Column1": [10, 15, 20, 25, 30],
        "Column2": [5, 10, 15, 20, 25]
    })

    try:
        Firstplot(sample_data)
    except Exception as e:
        pytest.fail(f"Firstplot failed with error: {e}")

def test_Secondplot():
    con = pd.Series([10, 15, 12, 18, 20])
    ftd_grn = pd.Series([9, 14, 11, 17, 19])

    try:
        Secondplot(con, ftd_grn, 2.3, 0.01)
    except Exception as e:
        pytest.fail(f"Secondplot failed with error: {e}")
