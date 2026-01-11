import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
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
    print("Running Validation: 03 Weibull Reliability (Parametric)")

    # 1. Setup Parameters
    # Weibull(k=2, lambda=50) -> Rayleigh-like but scaled
    shape = 2.0
    scale = 50.0
    n_samples = 1000
    censoring_time = 60.0

    np.random.seed(42)

    # 2. Generate Data
    # scipy.stats.weibull_min uses c=shape, scale=scale
    true_values = np.random.weibull(a=shape, size=n_samples) * scale

    # Apply Right Censoring
    status = true_values > censoring_time # True if Censored (Event > 60)
    values = true_values.copy()
    values[status] = censoring_time

    print(f"Total Samples: {n_samples}")
    print(f"Censored Samples: {np.sum(status)} ({np.mean(status)*100:.1f}%)")

    # 3. Calculate Analytical Truth (Expected Conditional Mean)
    # E[T | T > C] = C + Integral_C_inf S(t) dt / S(C)
    # Or just Integral_C_inf t * f(t) dt / S(C)

    S_C = weibull_sf(censoring_time, shape, scale)

    # Integral of t * f(t) from C to inf
    numerator, _ = quad(lambda t: t * weibull_pdf(t, shape, scale), censoring_time, np.inf)

    true_conditional_mean = numerator / S_C
    print(f"\nAnalytical True Conditional Mean (T > {censoring_time}): {true_conditional_mean:.4f}")

    # 4. Run ndimpute
    # Explicitly requesting 'weibull' distribution as the data is Weibull.
    # Default is 'lognormal' which would cause bias here.
    df_result = impute(values, status, method='parametric', censoring_type='right', dist='weibull')

    # 5. Compare
    # Get imputed values for the censored subset
    imputed_censored = df_result.loc[status, 'imputed_value']

    mean_imputed = imputed_censored.mean()
    print(f"ndimpute Average Imputed Value: {mean_imputed:.4f}")

    bias = mean_imputed - true_conditional_mean
    pct_error = (bias / true_conditional_mean) * 100

    print(f"Bias: {bias:.4f}")
    print(f"Percent Error: {pct_error:.4f}%")

    # 6. Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(true_values, bins=50, alpha=0.3, label='True Data (Unobserved)', color='gray', density=True)
    plt.hist(df_result['imputed_value'], bins=50, alpha=0.5, label='Imputed Data', color='blue', density=True)
    plt.axvline(censoring_time, color='red', linestyle='--', label='Censoring Time')
    plt.axvline(true_conditional_mean, color='green', linestyle='-', label='True Cond. Mean')
    plt.axvline(mean_imputed, color='blue', linestyle=':', label='Imputed Cond. Mean')

    plt.title("Weibull Reliability: True vs Imputed Distribution")
    plt.xlabel("Time to Failure")
    plt.legend()
    plt.savefig("distribution_check.png")
    print("Saved distribution_check.png")

    # 7. Success Criteria
    # The parametric fit on N=1000 should be very good.
    # Expect error < 1%
    if abs(pct_error) < 1.0:
        print("\n[PASS] Validation successful.")
    else:
        print("\n[FAIL] Validation exceeded tolerance.")

if __name__ == "__main__":
    run_validation()
