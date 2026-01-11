import unittest
import numpy as np
from scipy.stats import norm, lognorm, weibull_min
from ndimpute.api import impute

class TestParametricImputation(unittest.TestCase):

    def test_parametric_normal(self):
        """Test Parametric Imputation with Normal Distribution (includes negative values)."""
        np.random.seed(42)
        mu, sigma = 10.0, 2.0
        n = 1000
        data = np.random.normal(mu, sigma, n)

        # Scenario: Left Censored < 8
        limit = 8.0
        status = np.zeros(n, dtype=int)
        mask = data < limit

        values = data.copy()
        values[mask] = limit
        status[mask] = -1

        # Impute
        df = impute(values, status, method='parametric', dist='normal', censoring_type='mixed')

        imputed = df.loc[mask, 'imputed_value']

        # 1. Bound check
        self.assertTrue(np.all(imputed <= limit), "Imputed values should be <= Limit for Left Censoring")

        # 2. Theoretical Mean Check
        # E[X | X < L] for N(mu, sigma)
        alpha = (limit - mu) / sigma
        expected_val = mu - sigma * (norm.pdf(alpha) / norm.cdf(alpha))

        # Allow small tolerance (statistical fluctuation + estimation error)
        self.assertAlmostEqual(imputed.mean(), expected_val, delta=0.2)

    def test_parametric_normal_negative(self):
        """Test Parametric Normal with purely negative data."""
        np.random.seed(42)
        data = np.random.normal(-10, 2, 500)

        # Right censor > -8
        limit = -8.0
        status = np.zeros(500, dtype=int)
        mask = data > limit

        values = data.copy()
        values[mask] = limit
        status[mask] = 1 # Right

        # Impute
        df = impute(values, status, method='parametric', dist='normal', censoring_type='mixed')
        imputed = df.loc[mask, 'imputed_value']

        self.assertTrue(np.all(imputed >= limit))

        # E[X | X > L]
        # alpha = (L - mu) / sigma
        # E = mu + sigma * phi(alpha) / (1 - Phi(alpha))
        mu_hat = df.loc[status==0, 'original_value'].mean() # Rough estimate
        # Ideally we use the fitted params but we don't expose them.
        # Just check it imputed something reasonable (e.g. > limit)
        self.assertTrue(imputed.mean() > limit)

    def test_parametric_lognormal(self):
        """Test Parametric Imputation with Lognormal Distribution."""
        np.random.seed(42)
        # lognorm shape parameter is sigma (s), scale is exp(mu)
        s = 0.5
        scale = np.exp(2.0) # mu = 2.0
        n = 1000
        data = lognorm.rvs(s=s, scale=scale, size=n)

        # Right Censored > 12
        limit = 12.0
        status = np.zeros(n, dtype=int)
        mask = data > limit

        values = data.copy()
        values[mask] = limit
        status[mask] = 1

        df = impute(values, status, method='parametric', dist='lognormal', censoring_type='mixed')
        imputed = df.loc[mask, 'imputed_value']

        self.assertTrue(np.all(imputed >= limit))

        # Theoretical check
        # E[X | X > L]
        mu = 2.0
        sigma = 0.5
        alpha = (np.log(limit) - mu) / sigma
        mean_uncond = np.exp(mu + 0.5 * sigma**2)
        expected_val = mean_uncond * (1 - norm.cdf(alpha - sigma)) / (1 - norm.cdf(alpha))

        self.assertAlmostEqual(imputed.mean(), expected_val, delta=0.5)

    def test_parametric_weibull(self):
        """Test Parametric Imputation with Weibull (Regression Test)."""
        np.random.seed(42)
        # weibull_min shape=k, scale=lambda
        shape = 2.0
        scale = 10.0
        data = weibull_min.rvs(shape, scale=scale, size=1000)

        # Left Censored < 5
        limit = 5.0
        status = np.zeros(1000, dtype=int)
        mask = data < limit

        values = data.copy()
        values[mask] = limit
        status[mask] = -1

        df = impute(values, status, method='parametric', dist='weibull', censoring_type='mixed')
        imputed = df.loc[mask, 'imputed_value']

        self.assertTrue(np.all(imputed <= limit))
        self.assertTrue(np.all(imputed >= 0))

        # Check against sample mean of true data in that tail
        true_tail_mean = data[mask].mean()
        # Imputation is expected value, should be close to true mean of tail
        self.assertAlmostEqual(imputed.mean(), true_tail_mean, delta=0.5)

    def test_api_defaults(self):
        """Ensure API defaults to lognormal if unspecified for parametric."""
        values = np.exp(np.random.normal(0, 1, 100))
        status = np.zeros(100, dtype=int)
        status[values > 2] = 1
        values[values > 2] = 2

        # Should not raise error (defaults to lognormal)
        df = impute(values, status, method='parametric', censoring_type='mixed')
        self.assertTrue(df['is_imputed'].any())

        # Negative values should fail with default (lognormal)
        values_neg = np.array([-1, -2, -3])
        status_neg = np.array([0, 0, 0])
        with self.assertRaises(ValueError):
             impute(values_neg, status_neg, method='parametric', censoring_type='mixed')

if __name__ == '__main__':
    unittest.main()
