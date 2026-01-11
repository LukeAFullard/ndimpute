import numpy as np
import pytest
from ndimpute.api import impute

def test_interval_ros_stochastic_randomness():
    """
    Test that when random_state=None, Interval ROS uses Random Sampling
    (different results each time), adhering to standard API conventions.
    """
    # Create dataset with identical censored intervals
    # 10 identical intervals [0, 5]
    n_cens = 10
    left = np.zeros(n_cens)
    right = np.full(n_cens, 5.0)

    # 20 observed values > 5 to fit regression
    n_obs = 20
    obs = np.linspace(6, 10, n_obs)

    # Combine
    left_all = np.concatenate([left, obs])
    right_all = np.concatenate([right, obs])

    intervals = np.column_stack((left_all, right_all))

    # Run Imputation with random_state=None (Should be Random)
    res1 = impute(intervals, method='ros', censoring_type='interval', impute_type='stochastic', random_state=None, dist='lognormal')
    res2 = impute(intervals, method='ros', censoring_type='interval', impute_type='stochastic', random_state=None, dist='lognormal')

    imputed1 = res1['imputed_value'].values[:n_cens]
    imputed2 = res2['imputed_value'].values[:n_cens]

    # Check 1: Results should be different (Randomness)
    assert not np.allclose(imputed1, imputed2), "Interval ROS with random_state=None should produce different results."

    # Check 2: Values should be distinct (within a single run)
    # With N=10 random draws, duplicates are extremely unlikely (float).
    unique_vals = np.unique(imputed1)
    assert len(unique_vals) == n_cens, "Imputed values should be distinct (randomly sampled) for identical intervals."

def test_interval_ros_seeded_reproducibility():
    """
    Test that when random_state is provided, it is reproducible.
    """
    n_cens = 10
    left = np.zeros(n_cens)
    right = np.full(n_cens, 5.0)
    n_obs = 20
    obs = np.linspace(6, 10, n_obs)
    left_all = np.concatenate([left, obs])
    right_all = np.concatenate([right, obs])
    intervals = np.column_stack((left_all, right_all))

    res1 = impute(intervals, method='ros', censoring_type='interval', impute_type='stochastic', random_state=42, dist='lognormal')
    res2 = impute(intervals, method='ros', censoring_type='interval', impute_type='stochastic', random_state=42, dist='lognormal')

    np.testing.assert_array_almost_equal(res1['imputed_value'], res2['imputed_value'],
                                         err_msg="Seeded run should be reproducible.")

if __name__ == "__main__":
    pytest.main([__file__])
