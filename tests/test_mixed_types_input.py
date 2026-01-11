import numpy as np
from ndimpute import impute

def test_mixed_types_numeric_and_strings():
    # Test mixing floats/ints with strings in the input list
    data = [10.0, "<0.5", 5, ">100"]
    # 10.0 -> Obs
    # <0.5 -> Left
    # 5 -> Obs
    # >100 -> Right

    df = impute(data, method='parametric', dist='lognormal')

    # Verify values parsed correctly
    expected_values = np.array([10.0, 0.5, 5.0, 100.0])
    np.testing.assert_allclose(df['original_value'], expected_values)

    # Verify status (Mixed -> -1, 0, 1)
    status = df['censoring_status'].values
    expected_status = np.array([0, -1, 0, 1])
    np.testing.assert_array_equal(status, expected_status)

    # Verify imputation logic
    # Left < 0.5
    assert df.loc[1, 'imputed_value'] <= 0.5
    # Right > 100
    assert df.loc[3, 'imputed_value'] >= 100.0
