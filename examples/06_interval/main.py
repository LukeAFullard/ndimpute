import numpy as np
import pandas as pd
from ndimpute import impute

def main():
    print("=== Interval Censoring Example (Turnbull ROS) ===\n")

    # 1. Define Interval Data
    # Shape must be (N, 2) where each row is [Left_Bound, Right_Bound]
    # - [10, 10]: Exact observation at 10 (Left == Right)
    # - [0, 5]: Interval censored between 0 and 5
    # - [5, 10]: Interval censored between 5 and 10
    # - [20, np.inf]: Right censored at 20 (Open interval)

    # Create a synthetic dataset that looks somewhat Lognormal
    # Observed: 10, 12, 15
    # Intervals: (0, 5), (5, 8), (20, inf)

    bounds = np.array([
        [10.0, 10.0],  # Exact
        [12.0, 12.0],  # Exact
        [15.0, 15.0],  # Exact
        [0.0, 5.0],    # Interval (0, 5]
        [5.0, 8.0],    # Interval (5, 8]
        [20.0, np.inf] # Right Censored [20, inf)
    ])

    print("Input Data (Bounds):")
    print(pd.DataFrame(bounds, columns=['Left', 'Right']))
    print("-" * 30)

    # 2. Impute using 'ros' method
    # This uses the Turnbull Estimator to find plotting positions for intervals
    # and then fits a ROS model.
    # By default, impute_type='stochastic' for intervals to preserve variance.

    # We set a random_state for reproducibility
    df = impute(bounds,
                method='ros',
                censoring_type='interval',
                dist='lognormal',
                random_state=42)

    print("\nImputation Results:")
    print(df[['original_left', 'original_right', 'imputed_value']])

    # verify that imputed values are within bounds
    # Note: For Right Censored [20, inf), imputed should be > 20
    # For Interval [0, 5], imputed should be > 0 and <= 5

    print("\nVerification:")
    for i, row in df.iterrows():
        l, r = row['original_left'], row['original_right']
        val = row['imputed_value']

        within_bounds = (val >= l) and (val <= r)
        status_str = "OK" if within_bounds else "FAIL"
        print(f"Row {i}: [{l:<4}, {r:<4}] -> {val:.4f}  ({status_str})")

if __name__ == "__main__":
    main()
