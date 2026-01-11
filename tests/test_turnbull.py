import unittest
import numpy as np
from ndimpute._turnbull import turnbull_em, predict_turnbull

class TestTurnbull(unittest.TestCase):
    def test_turnbull_basic(self):
        # Example using exact observations and intervals:
        # 1. Obs: 2
        # 2. Interval: [1, 3]
        # 3. Interval: [2, 4]

        left = np.array([2, 1, 2])
        right = np.array([2, 3, 4])

        intervals, probs = turnbull_em(left, right)

        # Verify that intervals are returned and probabilities sum to 1.
        self.assertTrue(len(intervals) > 0)
        self.assertAlmostEqual(np.sum(probs), 1.0)

        # Check if the singleton [2, 2] is identified as an equivalence interval.
        # This is expected because the exact observation at 2 creates a boundary.
        has_2 = False
        for interval in intervals:
            if interval[0] == 2 and interval[1] == 2:
                has_2 = True
        self.assertTrue(has_2)

    def test_predict_turnbull(self):
        # Setup: Intervals [1, 1] (prob 0.5) and [3, 3] (prob 0.5).
        intervals = np.array([[1, 1], [3, 3]])
        probs = np.array([0.5, 0.5])

        # Evaluate Survival Function S(t) = P(T > t).
        # t=0: Both intervals are > 0. S(0) = 1.0.
        # t=1: Interval [1,1] is not strictly > 1. [3,3] > 1. S(1) = 0.5.
        # t=2: [3,3] > 2. S(2) = 0.5.
        # t=3: [3,3] is not strictly > 3. S(3) = 0.0.

        times = np.array([0, 0.9, 1, 1.1, 2.9, 3, 3.1])
        surv = predict_turnbull(intervals, probs, times)

        expected = np.array([1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0])
        np.testing.assert_array_almost_equal(surv, expected)

    def test_predict_turnbull_intervals(self):
        # Setup: Single interval (1, 2) with probability 1.0.
        # Represented as [1, 2] internally.
        intervals = np.array([[1, 2]])
        probs = np.array([1.0])

        # Survival Function Logic:
        # S(t) is calculated conservatively as the lower bound of survival probability.
        # If t <= start, then the entire interval mass is considered > t (S(t) includes this mass).
        # If t > start, we cannot guarantee the mass is > t, so it is excluded.

        times = np.array([0.5, 1.0, 1.1, 2.0])
        surv = predict_turnbull(intervals, probs, times)

        # t=0.5: 1 >= 0.5 -> Include mass (1.0).
        # t=1.0: 1 >= 1.0 -> Include mass (1.0).
        # t=1.1: 1 < 1.1 -> Exclude mass (0.0). Conservative estimate.

        expected = np.array([1.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(surv, expected)

if __name__ == '__main__':
    unittest.main()
