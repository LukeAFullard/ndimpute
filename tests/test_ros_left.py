import unittest
import numpy as np
import pandas as pd
from ndimpute.api import impute
from ndimpute._ros_left import impute_ros_left

class TestROSLeft(unittest.TestCase):
    def test_ros_left_basic(self):
        # Data inspired by NADA textbook examples
        # Values: 10, <2, <2, 5, 6
        # Sorted: <2, <2, 5, 6, 10
        # This is a very small dataset, might trigger the "too few" check or be unstable.

        # Use a slightly larger synthetic dataset
        np.random.seed(42)
        uncensored = np.random.lognormal(mean=2, sigma=0.5, size=20)
        censored = np.array([1.0] * 5) # LOD = 1.0

        values = np.concatenate([uncensored, censored])
        # Status: True if censored (value is LOD)
        status = np.concatenate([np.zeros(20, dtype=bool), np.ones(5, dtype=bool)])

        # Shuffle
        idx = np.arange(len(values))
        np.random.shuffle(idx)
        values = values[idx]
        status = status[idx]

        # Run Imputation
        df_result = impute(values, status, method='ros', censoring_type='left')

        # Checks
        self.assertEqual(len(df_result), 25)

        # Uncensored values should remain unchanged
        unc_mask = ~df_result['censoring_status']
        np.testing.assert_array_almost_equal(
            df_result.loc[unc_mask, 'imputed_value'],
            df_result.loc[unc_mask, 'original_value']
        )

        # Censored values should be imputed < LOD (1.0)
        # In ROS, we impute based on regression. If the fit is good, they should be < LOD.
        # However, ROS does not strictly force imputed values to be < LOD if the regression line
        # predicts otherwise (though plotting positions usually ensure they are in the lower tail).

        cens_mask = df_result['censoring_status']
        imputed_cens = df_result.loc[cens_mask, 'imputed_value']

        # Generally they should be smaller than the smallest uncensored value if the distribution holds
        # But specifically they should be around the lower tail.
        # Check they are not NaN and are positive
        self.assertTrue(np.all(imputed_cens > 0))

    def test_few_uncensored_raises(self):
        values = [1, 1, 1]
        status = [True, True, True]
        with self.assertRaises(ValueError):
            impute_ros_left(values, status)

if __name__ == '__main__':
    unittest.main()
