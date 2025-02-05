import pytest
import pandas as pd
from project import Loaddata, Clean

def test_Clean():
    sample_data = pd.DataFrame({
        "Column1": ["1", "2", "three", "4"],"Column2": ["5", "six", "7", "8"]
    })
    columns_to_clean = ["Column1", "Column2"]
    cleaned_data = Clean(sample_data, columns_to_clean)
    
    assert cleaned_data is not None, "Clean function should return a DataFrame"
    assert cleaned_data["Column1"].dtype=="float64", "Column1 should be numeric"
    assert cleaned_data["Column2"].dtype=="float64", "Column2 should be numeric"
