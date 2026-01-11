import unittest
import numpy as np
from ndimpute.api import impute
from ndimpute._ros_right import impute_ros_right

class TestRightImputation(unittest.TestCase):
    def test_ros_right_basic(self):
        np.random.seed(42)
        # Right censored data: Survival times
        # Uncensored: Observed failures
        # Censored: Censoring times (subject left study)

        uncensored = np.random.weibull(a=1.5, size=50) * 100
        censored_times = np.random.uniform(50, 200, size=10)

        # For right censoring:
        # If uncensored, we observe T.
        # If censored, we observe C, and we know T > C.

        values = np.concatenate([uncensored, censored_times])
        # Status: True if censored (event did NOT happen)
        # Note: In survival analysis (lifelines), event_observed=1 usually means uncensored.
        # Our API says: "True if censored". So 1 = Censored.
        status = np.concatenate([np.zeros(50, dtype=bool), np.ones(10, dtype=bool)])

        df_result = impute(values, status, method='ros', censoring_type='right')

        # Uncensored should be same
        unc_mask = ~df_result['censoring_status']
        np.testing.assert_array_almost_equal(
            df_result.loc[unc_mask, 'imputed_value'],
            df_result.loc[unc_mask, 'original_value']
        )

        # Censored values should be imputed > original value (T > C)
        cens_mask = df_result['censoring_status']
        original_cens = df_result.loc[cens_mask, 'original_value']
        imputed_cens = df_result.loc[cens_mask, 'imputed_value']

        # In Reverse ROS, we fit a line to the upper tail.
        # Imputed values should generally be larger than the censoring time.
        # (Though strictly speaking regression doesn't guarantee T > C for every point if fit is poor,
        # but conceptually it should be close).

        # Check that on average we increased the values
        self.assertTrue(np.mean(imputed_cens) > np.mean(original_cens))

    def test_parametric_right(self):
        np.random.seed(42)
        uncensored = np.random.weibull(a=2.0, size=100) * 50
        censored_times = np.array([40.0] * 20)

        values = np.concatenate([uncensored, censored_times])
        status = np.concatenate([np.zeros(100, dtype=bool), np.ones(20, dtype=bool)])

        df_result = impute(values, status, method='parametric', censoring_type='right')

        cens_mask = df_result['censoring_status']
        original_cens = df_result.loc[cens_mask, 'original_value']
        imputed_cens = df_result.loc[cens_mask, 'imputed_value']

        # For conditional mean, it is mathematically guaranteed that E[T|T>C] > C
        self.assertTrue(np.all(imputed_cens > original_cens))

if __name__ == '__main__':
    unittest.main()
