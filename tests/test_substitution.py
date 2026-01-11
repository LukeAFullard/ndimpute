import unittest
import numpy as np
from ndimpute.api import impute

class TestSubstitution(unittest.TestCase):
    def test_left_sub_half(self):
        values = np.array([10.0, 4.0, 6.0])
        status = np.array([False, True, False]) # 4.0 is censored (<4.0)

        # Default strategy='half'
        df = impute(values, status, method='substitution', censoring_type='left')

        expected = np.array([10.0, 2.0, 6.0])
        np.testing.assert_array_equal(df['imputed_value'].values, expected)

    def test_left_sub_zero(self):
        values = np.array([10.0, 4.0])
        status = np.array([False, True])

        df = impute(values, status, method='substitution', censoring_type='left', strategy='zero')

        expected = np.array([10.0, 0.0])
        np.testing.assert_array_equal(df['imputed_value'].values, expected)

    def test_left_sub_lod(self):
        values = np.array([10.0, 4.0])
        status = np.array([False, True])

        df = impute(values, status, method='substitution', censoring_type='left', strategy='value')

        expected = np.array([10.0, 4.0])
        np.testing.assert_array_equal(df['imputed_value'].values, expected)

    def test_left_sub_multiple(self):
        values = np.array([10.0, 2.0])
        status = np.array([False, True]) # <2.0

        # Multiplier 0.5 -> 2.0 * 0.5 = 1.0
        df = impute(values, status, method='substitution', censoring_type='left',
                   strategy='multiple', multiplier=0.5)

        expected = np.array([10.0, 1.0])
        np.testing.assert_array_equal(df['imputed_value'].values, expected)

    def test_right_sub_value(self):
        values = np.array([100.0, 50.0])
        status = np.array([False, True]) # 50.0 is censored (>50.0)

        df = impute(values, status, method='substitution', censoring_type='right', strategy='value')

        expected = np.array([100.0, 50.0])
        np.testing.assert_array_equal(df['imputed_value'].values, expected)

    def test_right_sub_multiple(self):
        values = np.array([100.0, 10.0])
        status = np.array([False, True]) # >10.0

        # Multiplier 1.1 -> 10.0 * 1.1 = 11.0
        df = impute(values, status, method='substitution', censoring_type='right',
                   strategy='multiple', multiplier=1.1)

        expected = np.array([100.0, 11.0])
        np.testing.assert_array_equal(df['imputed_value'].values, expected)

    def test_invalid_strategy(self):
        values = [1, 2]
        status = [False, True]
        with self.assertRaises(ValueError):
            impute(values, status, method='substitution', censoring_type='left', strategy='invalid')

    def test_missing_multiplier(self):
        values = [1, 2]
        status = [False, True]
        with self.assertRaises(ValueError):
            impute(values, status, method='substitution', censoring_type='left', strategy='multiple')

if __name__ == '__main__':
    unittest.main()
