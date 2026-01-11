import numpy as np
import pytest
from ndimpute import impute

def test_auto_string_parsing_simple():
    # Simple case with <
    data = ["<0.5", "1.0", "2.0", "<0.5", "3.0"]
    df = impute(data)

    assert len(df) == 5
    # Depending on how status is inferred (mixed vs left), it might be -1 or True.
    # If inferred type is 'left', status is boolean (True/False).
    # If inferred type is 'mixed', status is int (-1, 0, 1).
    # With only '<' present, detect_and_parse returns 'left'.
    # api.py converts status to boolean if 'left'.
    # But wait, api.py says:
    # if censoring_type == 'mixed': ... else: status = np.array(status, dtype=bool)
    # detect_and_parse returns 'left' if only '<'.
    # So status is boolean.

    assert df['censoring_status'].sum() == 2 # 2 left censored (True)
    assert np.all(df.loc[[0, 3], 'is_imputed'])
    assert np.all(df.loc[[1, 2, 4], 'is_imputed'] == False)

    # Check values
    # <0.5 imputed should be <= 0.5 (clipped at LOD)
    assert np.all(df.loc[[0, 3], 'imputed_value'] <= 0.5)

def test_auto_string_parsing_mixed():
    # Mixed case with < and >
    data = ["<0.5", "5.0", ">10.0", "7.0"]
    # This should be detected as mixed
    # <0.5 -> Left (-1)
    # >10.0 -> Right (1)

    df = impute(data)

    # Check status column interpretation
    # Default mixed status is integer: -1, 0, 1
    # But the API returns 'censoring_status' column.

    status = df['censoring_status'].values
    assert status[0] == -1
    assert status[1] == 0
    assert status[2] == 1
    assert status[3] == 0

    # Check imputed values
    # Left < 0.5
    assert df.loc[0, 'imputed_value'] <= 0.5
    # Right > 10.0
    assert df.loc[2, 'imputed_value'] >= 10.0

def test_auto_string_custom_markers():
    # Custom markers
    data = ["LOD_0.5", "1.0", "CENS_10.0"]

    # We need to pass markers
    df = impute(data, left_marker="LOD_", right_marker="CENS_")

    status = df['censoring_status'].values
    assert status[0] == -1
    assert status[1] == 0
    assert status[2] == 1

    assert df.loc[0, 'imputed_value'] <= 0.5
    assert df.loc[2, 'imputed_value'] >= 10.0

def test_string_input_consistency():
    # Compare string input vs manual status input
    raw_vals = [0.5, 1.0, 2.0, 0.5, 3.0]
    status = [True, False, False, True, False]

    # Note: detect_and_parse defaults to 'left' type if only '<' found.
    # Manual impute needs censoring_type='left' to match.
    df_manual = impute(raw_vals, status=status, censoring_type='left', random_state=42)

    str_vals = ["<0.5", "1.0", "2.0", "<0.5", "3.0"]
    df_auto = impute(str_vals, random_state=42)

    # Should be identical
    np.testing.assert_allclose(
        df_manual['imputed_value'].values,
        df_auto['imputed_value'].values
    )

def test_auto_detect_fail():
    # Invalid string format should eventually raise ValueError
    # because status is missing and auto-detect failed.
    data = ["bad_format", "1.0"]
    with pytest.raises(ValueError, match="Status argument is required"):
        impute(data)
