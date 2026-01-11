import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ndimpute import impute

def weibull_pdf(x, shape, scale):
    return (shape / scale) * ((x / scale)**(shape - 1)) * np.exp(- (x / scale)**shape)

def weibull_sf(x, shape, scale):
    return np.exp(- (x / scale)**shape)

def run_validation():
    print("Running Validation: 06 Comparative Right Censoring")
    print("Comparing Parametric (Weibull) vs Reverse ROS (Lognormal) on Weibull Data")

    # 1. Setup Parameters
    shape = 2.0
    scale = 50.0
    n_samples = 1000
    censoring_time = 60.0

    np.random.seed(42)

    # 2. Generate Data (Weibull)
    true_values = np.random.weibull(a=shape, size=n_samples) * scale

    status = true_values > censoring_time # True if Censored
    values = true_values.copy()
    values[status] = censoring_time

    print(f"Censoring Rate: {np.mean(status)*100:.1f}%")

    # 3. Calculate True Conditional Mean (Analytical)
    S_C = weibull_sf(censoring_time, shape, scale)
    numerator, _ = quad(lambda t: t * weibull_pdf(t, shape, scale), censoring_time, np.inf)
    true_cond_mean = numerator / S_C
    print(f"True Conditional Mean (>60): {true_cond_mean:.4f}")

    # 4. Method A: Parametric (Weibull) - The "Correct" Model
    df_param = impute(values, status, method='parametric', censoring_type='right', dist='weibull')
    mean_param = df_param.loc[status, 'imputed_value'].mean()
    bias_param = mean_param - true_cond_mean

    # 5. Method B: Reverse ROS (Lognormal) - The "Robust/Incorrect" Model
    # ROS assumes lognormal on 1/x. 1/Weibull is Fréchet.
    # This checks robustness to misspecification.
    df_ros = impute(values, status, method='ros', censoring_type='right')
    mean_ros = df_ros.loc[status, 'imputed_value'].mean()
    bias_ros = mean_ros - true_cond_mean

    # 6. Method C: Substitution (Value) - Baseline
    df_sub = impute(values, status, method='substitution', censoring_type='right', strategy='value')
    mean_sub = df_sub.loc[status, 'imputed_value'].mean()
    bias_sub = mean_sub - true_cond_mean

    print("\n--- Results (Bias) ---")
    print(f"Parametric (Weibull): {bias_param:.4f}")
    print(f"Reverse ROS:          {bias_ros:.4f}")
    print(f"Substitution (C):     {bias_sub:.4f}")

    # 7. Visualization
    plt.figure(figsize=(10, 6))

    # True Tail
    true_tail = true_values[true_values > censoring_time]
    plt.hist(true_tail, bins=30, alpha=0.3, color='gray', label='True Tail', density=True)

    # Imputed Tails
    plt.hist(df_param.loc[status, 'imputed_value'], bins=30, alpha=0.5, color='blue', label='Parametric', histtype='step', linewidth=2, density=True)
    plt.hist(df_ros.loc[status, 'imputed_value'], bins=30, alpha=0.5, color='red', label='Reverse ROS', histtype='step', linewidth=2, density=True)

    plt.axvline(true_cond_mean, color='gray', linestyle='--', label='True Mean')

    plt.title("Right Censoring Imputation Comparison")
    plt.xlabel("Value")
    plt.legend()
    plt.savefig("method_comparison.png")
    print("Saved method_comparison.png")

    # 8. Success Criteria
    # Parametric should be best (lowest bias).
    # Substitution should be worst (negative bias).
    # ROS should be somewhere in between or reasonable.

    if abs(bias_param) < abs(bias_sub):
        print("\n[PASS] Parametric significantly outperforms Substitution.")
    else:
        print("\n[FAIL] Parametric failed to outperform Substitution.")

    if abs(bias_param) < abs(bias_ros):
        print("[INFO] Parametric (True Model) outperforms ROS (Non-parametric).")
    else:
        print("[INFO] ROS outperformed Parametric (Unexpected for this data).")

if __name__ == "__main__":
    run_validation()
