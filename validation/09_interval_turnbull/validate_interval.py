import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ndimpute import impute

def run_validation():
    print("Running Validation: 09 Interval Censoring (Turnbull ROS)")

    np.random.seed(42)

    # 1. Generate True Data (Lognormal)
    n = 100
    true_vals = np.random.lognormal(mean=2, sigma=0.5, size=n)

    # 2. Generate Intervals
    # Create inspection times
    # 0, 2, 4, 6, 8, 10, ...
    inspection_times = np.arange(0, 50, 2.0)

    left_bounds = []
    right_bounds = []

    for t in true_vals:
        # Find interval (t_i, t_{i+1}] containing t
        idx = np.searchsorted(inspection_times, t)
        if idx == 0:
            # t < first inspection time
            L, R = 0, inspection_times[0]
        elif idx >= len(inspection_times):
            # t > last inspection time
            L, R = inspection_times[-1], np.inf
        else:
            L, R = inspection_times[idx-1], inspection_times[idx]

        left_bounds.append(L)
        right_bounds.append(R)

    left = np.array(left_bounds)
    right = np.array(right_bounds)
    bounds = np.column_stack((left, right))

    print(f"Data Generated. N={n}. Example intervals: {bounds[:3]}")

    # 3. Run Imputation
    df_res = impute(bounds, censoring_type='interval', method='ros', dist='lognormal')
    imputed = df_res['imputed_value'].values

    # 4. Validation
    # Check 1: Constraints
    # Imputed value must be within (L, R)
    # Note: ROS regression might predict outside if distribution mismatch, but truncated expectation
    # logic should theoretically keep it inside or near bounds.
    # Actually my implementation does NOT clamp interval ROS to bounds explicitly at the end!
    # It calculates E[T|L<T<R] using the fitted distribution.
    # If the fit is good, it should be inside.

    in_bounds = (imputed >= left) & (imputed <= right) # Use <= for R if closed
    # Handle infinite right
    in_bounds[np.isinf(right)] = (imputed[np.isinf(right)] >= left[np.isinf(right)])

    print(f"Percentage inside bounds: {np.mean(in_bounds)*100:.1f}%")

    # Check 2: Mean Recovery
    true_mean = np.mean(true_vals)
    imp_mean = np.mean(imputed)
    bias = imp_mean - true_mean

    print(f"True Mean: {true_mean:.4f}")
    print(f"Imputed Mean: {imp_mean:.4f}")
    print(f"Bias: {bias:.4f} ({(bias/true_mean)*100:.1f}%)")

    # 5. Plot
    plt.figure(figsize=(10, 6))
    plt.hist(true_vals, bins=30, alpha=0.3, color='gray', label='True', density=True)
    plt.hist(imputed, bins=30, alpha=0.5, color='blue', label='Imputed', histtype='step', linewidth=2, density=True)
    plt.title("Interval Censoring Reconstruction")
    plt.legend()
    plt.savefig("interval_check.png")
    print("Saved interval_check.png")

    if abs(bias) < 0.5:
        print("[PASS] Mean recovered successfully.")
    else:
        print("[FAIL] High bias.")

if __name__ == "__main__":
    run_validation()
