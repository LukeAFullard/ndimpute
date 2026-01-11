import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ndimpute import impute

def run_validation():
    print("Running Validation: 04 Mixed Censoring Stress")

    # 1. Setup Parameters
    # Weibull
    shape = 1.5
    scale = 100.0
    n_samples = 500

    # Limits
    limit_left = 20.0
    limit_right = 150.0

    np.random.seed(42)

    # 2. Generate Data
    true_values = np.random.weibull(a=shape, size=n_samples) * scale

    # True Mean
    true_mean_analytical = scale * gamma(1 + 1/shape)
    true_mean_sample = np.mean(true_values)

    print(f"True Analytical Mean: {true_mean_analytical:.4f}")
    print(f"True Sample Mean:     {true_mean_sample:.4f}")

    # 3. Apply Mixed Censoring
    # -1: Left, 0: Obs, 1: Right
    status = np.zeros(n_samples, dtype=int)
    values = true_values.copy()

    mask_left = true_values < limit_left
    status[mask_left] = -1
    values[mask_left] = limit_left

    mask_right = true_values > limit_right
    status[mask_right] = 1
    values[mask_right] = limit_right

    pct_left = np.mean(mask_left) * 100
    pct_right = np.mean(mask_right) * 100
    print(f"Censoring: Left={pct_left:.1f}%, Right={pct_right:.1f}%")

    # 4. Method 1: Parametric (Generalized Likelihood)
    print("\n--- Method 1: Parametric (Mixed) ---")
    # Data is Weibull, so we MUST specify dist='weibull' as default is now lognormal
    df_param = impute(values, status, method='parametric', censoring_type='mixed', dist='weibull')
    mean_param = df_param['imputed_value'].mean()
    bias_param = mean_param - true_mean_sample
    print(f"Imputed Mean: {mean_param:.4f}")
    print(f"Bias: {bias_param:.4f}")

    # 5. Method 2: Heuristic ROS (Left then Right)
    print("\n--- Method 2: Heuristic ROS ---")
    df_ros = impute(values, status, method='ros', censoring_type='mixed')
    mean_ros = df_ros['imputed_value'].mean()
    bias_ros = mean_ros - true_mean_sample
    print(f"Imputed Mean: {mean_ros:.4f}")
    print(f"Bias: {bias_ros:.4f}")

    # 6. Visualization
    plt.figure(figsize=(12, 6))

    # True Data (Reference)
    plt.hist(true_values, bins=50, alpha=0.3, color='gray', label='True Data (Unobserved)', density=True)

    # Observed Data (Middle)
    obs_data = values[status == 0]
    # We plot observed just to show the gap
    # plt.hist(obs_data, bins=30, alpha=0.3, color='green', label='Observed Only', density=True)

    # Parametric Imputation
    plt.hist(df_param['imputed_value'], bins=50, alpha=0.5, color='blue', label='Parametric Imputation', histtype='step', linewidth=2, density=True)

    # ROS Imputation
    plt.hist(df_ros['imputed_value'], bins=50, alpha=0.5, color='red', label='Heuristic ROS', histtype='step', linewidth=2, density=True)

    plt.axvline(limit_left, color='black', linestyle='--', label='Left Limit')
    plt.axvline(limit_right, color='black', linestyle='--', label='Right Limit')

    plt.title(f"Mixed Censoring Reconstruction (Left<{limit_left}, Right>{limit_right})")
    plt.xlabel("Value")
    plt.legend()
    plt.savefig("mixed_reconstruction.png")
    print("\nSaved mixed_reconstruction.png")

    # 7. Success Criteria
    # Parametric should be very accurate (it assumes Weibull and data IS Weibull)
    # ROS is heuristic, might be slightly worse but should be reasonable.

    if abs(bias_param) < 5.0: # 5.0 is about 5% of mean ~90
        print("\n[PASS] Parametric reconstruction successful.")
    else:
        print("\n[FAIL] Parametric bias too high.")

if __name__ == "__main__":
    run_validation()
