import numpy as np
import pandas as pd
from ndimpute import impute
import sys

def check_readiness():
    print("Running Scientific Readiness Check...")

    # 1. Variance Preservation (Lognormal Left Censored)
    np.random.seed(42)
    n = 1000
    mu, sigma = 4.0, 1.0
    data = np.random.lognormal(mean=mu, sigma=sigma, size=n)

    limit = np.percentile(data, 50)
    censored_mask = data < limit
    values = data.copy()
    values[censored_mask] = limit

    # Impute (Default ROS -> Kaplan-Meier -> Stochastic)
    df = impute(values, censored_mask, method='ros', censoring_type='left', dist='lognormal')
    imputed = df['imputed_value'].values

    var_orig = np.var(np.log(data))
    var_imp = np.var(np.log(imputed))
    ratio = var_imp / var_orig

    print(f"\n[Variance Check]")
    print(f"Original Variance: {var_orig:.4f}")
    print(f"Imputed Variance:  {var_imp:.4f}")
    print(f"Ratio:             {ratio:.4f}")

    # Tolerance: +/- 20% is acceptable for ROS estimation.
    # Previous implementation was ~0.83 (17% bias).
    # Current implementation ~1.11 (11% bias).
    # Ideally 1.0, but sample noise and fit error contribute.
    if 0.80 < ratio < 1.20:
        print("PASS: Variance preserved within tolerance.")
    else:
        print("FAIL: Variance collapsed or inflated significantly.")
        sys.exit(1)

    # 2. Distribution Shape (Unique Values)
    cens_vals = imputed[censored_mask]
    n_unique = len(np.unique(cens_vals))
    print(f"\n[Shape Check]")
    print(f"Censored Count: {sum(censored_mask)}")
    print(f"Unique Imputed Values: {n_unique}")

    if n_unique > sum(censored_mask) * 0.9:
         print("PASS: Imputed values are distributed.")
    else:
         print("FAIL: Imputed values are clumped (conditional mean spike).")
         sys.exit(1)

    print("\nReadiness Check Passed!")

if __name__ == "__main__":
    check_readiness()
