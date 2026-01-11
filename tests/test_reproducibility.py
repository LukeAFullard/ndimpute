import numpy as np
import pytest
import pandas as pd
from ndimpute.api import impute

def test_random_state_reproducibility():
    """
    Test that random_state ensures reproducibility across different calls.
    """
    # 1. Setup Data
    np.random.seed(42)
    # Left Censoring
    n = 100
    data = np.random.lognormal(0, 1, n)
    lod = np.percentile(data, 30)
    values = data.copy()
    status = values < lod
    values[status] = lod

    # 2. Test Left ROS
    res1 = impute(values, status, method='ros', censoring_type='left', impute_type='stochastic', random_state=123)
    res2 = impute(values, status, method='ros', censoring_type='left', impute_type='stochastic', random_state=123)
    res3 = impute(values, status, method='ros', censoring_type='left', impute_type='stochastic', random_state=456)

    # Assert identical results for same seed
    np.testing.assert_array_almost_equal(res1['imputed_value'], res2['imputed_value'],
                                         err_msg="Left ROS: Same seed should produce identical results")

    # Assert different results for different seed (for censored part)
    # Check only censored values
    mask = res1['censoring_status'].astype(bool)
    assert not np.allclose(res1.loc[mask, 'imputed_value'], res3.loc[mask, 'imputed_value']), \
        "Left ROS: Different seeds should produce different results"

def test_random_state_interval():
    """
    Test random_state for Interval ROS.
    """
    # Interval Data
    left = np.array([0, 0, 5, 10, 15])
    right = np.array([5, 5, 10, 15, 20])
    # [0,5], [0,5] are censored.

    intervals = np.column_stack((left, right))

    res1 = impute(intervals, method='ros', censoring_type='interval', impute_type='stochastic', random_state=999)
    res2 = impute(intervals, method='ros', censoring_type='interval', impute_type='stochastic', random_state=999)
    res3 = impute(intervals, method='ros', censoring_type='interval', impute_type='stochastic', random_state=111)

    np.testing.assert_array_almost_equal(res1['imputed_value'], res2['imputed_value'],
                                         err_msg="Interval ROS: Same seed should produce identical results")

    # Check difference for censored [0, 5]
    # Indices 0 and 1 are [0, 5]
    assert not np.allclose(res1.loc[0:1, 'imputed_value'], res3.loc[0:1, 'imputed_value']), \
         "Interval ROS: Different seeds should produce different results"

def test_random_state_mixed():
    """
    Test random_state for Mixed ROS.
    """
    values = np.array([5.0, 5.0, 10.0, 20.0, 30.0])
    status = np.array([-1, -1, 0, 0, 0]) # 2 left censored

    # Suppress warning for test
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res1 = impute(values, status, method='ros', censoring_type='mixed', impute_type='stochastic', random_state=777)
        res2 = impute(values, status, method='ros', censoring_type='mixed', impute_type='stochastic', random_state=777)
        res3 = impute(values, status, method='ros', censoring_type='mixed', impute_type='stochastic', random_state=888)

    np.testing.assert_array_almost_equal(res1['imputed_value'], res2['imputed_value'],
                                         err_msg="Mixed ROS: Same seed should produce identical results")

    assert not np.allclose(res1.loc[0:1, 'imputed_value'], res3.loc[0:1, 'imputed_value']), \
        "Mixed ROS: Different seeds should produce different results"

def test_random_state_parametric():
    """
    Test random_state for Parametric imputation (stochastic).
    """
    # Right Censored Data
    values = np.array([10, 20, 30, 40, 50])
    status = np.array([0, 0, 1, 1, 0]) # Right censored

    res1 = impute(values, status, method='parametric', censoring_type='right', impute_type='stochastic', random_state=55)
    res2 = impute(values, status, method='parametric', censoring_type='right', impute_type='stochastic', random_state=55)
    res3 = impute(values, status, method='parametric', censoring_type='right', impute_type='stochastic', random_state=66)

    np.testing.assert_array_almost_equal(res1['imputed_value'], res2['imputed_value'],
                                         err_msg="Parametric Right: Same seed should produce identical results")

    # Check censored
    mask = (status == 1)
    assert not np.allclose(res1.loc[mask, 'imputed_value'], res3.loc[mask, 'imputed_value']), \
        "Parametric Right: Different seeds should produce different results"

if __name__ == "__main__":
    pytest.main([__file__])
