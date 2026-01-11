import numpy as np
import pytest
from ndimpute.api import impute
from ndimpute._parametric import impute_right_conditional, impute_mixed_parametric

def test_parametric_right_stochastic_variance():
    """
    Test that stochastic imputation produces variance for identical censored values,
    whereas mean imputation produces constant values.
    """
    # 20 observations: 10 obs ~ N(10, 2), 10 censored at > 8
    # Right censoring means > C.
    # We have many values censored at > 10.

    values = np.array([5., 6., 7., 8., 9.] * 4 + [10.] * 20) # 20 observed, 20 censored
    censored = np.array([False]*20 + [True]*20)

    # 1. Mean Imputation (Deterministic)
    df_mean = impute(values, censored, method='parametric', censoring_type='right', dist='normal', impute_type='mean')
    imputed_mean = df_mean.loc[censored, 'imputed_value'].values

    # All imputed values for the same censoring limit (10) should be identical
    assert np.allclose(imputed_mean, imputed_mean[0]), "Mean imputation should be deterministic for same limit"

    # 2. Stochastic Imputation
    df_stoch = impute(values, censored, method='parametric', censoring_type='right', dist='normal', impute_type='stochastic', random_state=42)
    imputed_stoch = df_stoch.loc[censored, 'imputed_value'].values

    # Should have variance
    assert np.var(imputed_stoch) > 1e-4, "Stochastic imputation should preserve variance"

    # Should be different from mean
    assert not np.allclose(imputed_stoch, imputed_mean[0])

    # Should be greater than limit (10)
    assert np.all(imputed_stoch >= 10.0), "Imputed values must be > censoring limit"


def test_parametric_mixed_stochastic_bounds():
    """
    Test that stochastic imputation respects bounds for both left and right censoring.
    """
    # Mixed: -1 (Left < L), 0 (Obs), 1 (Right > R)
    values = np.array([10., 10., 10., 50., 60., 70., 100., 100., 100.])
    status = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1]) # 3 Left < 10, 3 Obs, 3 Right > 100

    # Use Uniform random state
    rng = np.random.default_rng(42)

    # Test Lognormal Stochastic
    imputed = impute_mixed_parametric(values, status, dist='lognormal', impute_type='stochastic', random_state=42)

    left_imputed = imputed[status == -1]
    right_imputed = imputed[status == 1]

    # Left censored (< 10) must be <= 10
    assert np.all(left_imputed <= 10.0)
    assert np.all(left_imputed > 0.0) # Lognormal must be positive
    assert np.var(left_imputed) > 0 # Should vary

    # Right censored (> 100) must be >= 100
    assert np.all(right_imputed >= 100.0)
    assert np.var(right_imputed) > 0

def test_parametric_reproducibility():
    """
    Test that random_state ensures reproducibility.
    """
    values = np.array([10.] * 10 + [20.] * 10)
    censored = np.array([False]*10 + [True]*10) # Right censored > 20

    run1 = impute_right_conditional(values, censored, dist='weibull', impute_type='stochastic', random_state=123)
    run2 = impute_right_conditional(values, censored, dist='weibull', impute_type='stochastic', random_state=123)
    run3 = impute_right_conditional(values, censored, dist='weibull', impute_type='stochastic', random_state=456)

    assert np.allclose(run1, run2), "Same seed should produce same results"
    assert not np.allclose(run1, run3), "Different seed should produce different results"

def test_parametric_distributions_stochastic():
    """
    Verify all distributions support stochastic mode.
    """
    values = np.array([5, 10, 15, 20]*5)
    cens = np.array([False, False, True, True]*5) # Censored at 15 and 20

    for dist in ['normal', 'lognormal', 'weibull']:
        res = impute_right_conditional(values, cens, dist=dist, impute_type='stochastic', random_state=1)
        # Check constraints
        assert np.all(res[cens] >= values[cens])
        assert np.var(res[cens]) > 0
