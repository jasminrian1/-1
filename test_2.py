import pytest
import pandas as pd
from project import ttestImplemnt, cohen_d

def test_ttest_valid():
    con = pd.Series([10, 15, 12, 18, 20])
    ftd_grn = pd.Series([9, 14, 11, 17, 19])

    t_stat, p_value = ttestImplemnt(con, ftd_grn)
    
    assert isinstance(t_stat, float), "T-test should return a float t-statistic"
    assert isinstance(p_value, float), "T-test should return a float p-value"
    assert p_value >= 0 and p_value <= 1, "P-value should be between 0 and 1"

def test_cohen_d():
    con = pd.Series([10, 15, 12, 18, 20])
    ftd_grn = pd.Series([9, 14, 11, 17, 19])

    d_value = cohen_d(con, ftd_grn)
    
    assert isinstance(d_value, float), "Cohen's d should return a float"
    assert d_value >= 0, "Effect size should be non-negative"
