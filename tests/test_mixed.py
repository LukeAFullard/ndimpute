import unittest
import numpy as np
from ndimpute.api import impute

class TestMixedImputation(unittest.TestCase):
    def test_mixed_substitution(self):
        # Data: [10 (Obs), <2 (Left), >20 (Right)]
        values = np.array([10.0, 2.0, 20.0])
        status = np.array([0, -1, 1])

        # Strategies: Left=Half (1.0), Right=Multiple*1.1 (22.0)
        df = impute(values, status, censoring_type='mixed', method='substitution',
                    left_strategy='half',
                    right_strategy='multiple', right_multiplier=1.1)

        expected = np.array([10.0, 1.0, 22.0])
        np.testing.assert_array_almost_equal(df['imputed_value'].values, expected)

    def test_mixed_parametric_basic(self):
        # Synthetic mixed data
        # Use simple Weibull
        np.random.seed(42)
        true_data = np.random.weibull(a=2.0, size=100) * 10

        # Create censoring
        # Left censor if < 5 (Status -1)
        # Right censor if > 15 (Status 1)
        # Else observed (Status 0)

        values = true_data.copy()
        status = np.zeros(100, dtype=int)

        mask_left = values < 5
        mask_right = values > 15

        # Replace values with limits
        values[mask_left] = 5.0
        status[mask_left] = -1

        values[mask_right] = 15.0
        status[mask_right] = 1

        df = impute(values, status, censoring_type='mixed', method='parametric')

        # Check Left Imputations < 5
        left_imputed = df.loc[df['censoring_status'] == -1, 'imputed_value']
        self.assertTrue(np.all(left_imputed < 5.0))
        # Also generally > 0 for Weibull
        self.assertTrue(np.all(left_imputed > 0.0))

        # Check Right Imputations > 15
        right_imputed = df.loc[df['censoring_status'] == 1, 'imputed_value']
        self.assertTrue(np.all(right_imputed > 15.0))

        # Check Observed Unchanged
        obs_imputed = df.loc[df['censoring_status'] == 0, 'imputed_value']
        obs_orig = df.loc[df['censoring_status'] == 0, 'original_value']
        np.testing.assert_array_equal(obs_imputed, obs_orig)

    def test_mixed_ros_heuristic(self):
        # Data inspired by a lognormal distribution
        np.random.seed(42)
        true_data = np.random.lognormal(mean=2, sigma=0.5, size=100) # Increased size

        values = true_data.copy()
        status = np.zeros(100, dtype=int)

        # Left censor low values (< 4)
        mask_left = values < 4
        values[mask_left] = 4.0 # LOD
        status[mask_left] = -1

        # Right censor high values (> 15)
        mask_right = values > 15
        values[mask_right] = 15.0 # End of study
        status[mask_right] = 1

        # Ensure we have enough data
        if np.sum(status == 0) < 5:
             self.skipTest("Not enough observed data generated for ROS test")

        df = impute(values, status, censoring_type='mixed', method='ros')

        left_imputed = df.loc[df['censoring_status'] == -1, 'imputed_value']
        right_imputed = df.loc[df['censoring_status'] == 1, 'imputed_value']

        # Check Order Preservation (This is the critical heuristic check)
        # The mean of Left Imputed should be < Mean Observed < Mean Right Imputed
        mean_left = np.mean(left_imputed)
        mean_obs = np.mean(df.loc[df['censoring_status'] == 0, 'imputed_value'])
        mean_right = np.mean(right_imputed)

        self.assertLess(mean_left, mean_obs, "Mean Left Imputed should be less than Mean Observed")
        self.assertLess(mean_obs, mean_right, "Mean Observed should be less than Mean Right Imputed")

        # Basic sanity checks (not strictly bound enforcing as ROS is regression)
        # At least 50% should be on the correct side of the limit
        self.assertGreater(np.mean(left_imputed < 4.0), 0.5)
        self.assertGreater(np.mean(right_imputed > 15.0), 0.5)

if __name__ == '__main__':
    unittest.main()
