# Example 07: Auto Distribution Selection

This example demonstrates the automatic distribution selection feature (`dist='auto'`) for Robust ROS imputation.

## Overview

The `dist='auto'` option allows the `impute` function to automatically choose the best-fitting distribution (Normal or Lognormal) based on the Regression on Order Statistics (ROS) $R^2$ score (Probability Plot Correlation Coefficient).

This is useful when the underlying distribution of the data is unknown or when processing multiple datasets with varying characteristics.

## Scenarios

1.  **Left Censoring (Lognormal Data):** Synthetic data generated from a Lognormal distribution is censored on the left. The algorithm correctly identifies 'lognormal' as the best fit.
2.  **Mixed Censoring (Normal Data):** Synthetic data generated from a Normal distribution is censored on both left and right sides. The algorithm correctly identifies 'normal' as the best fit.

## Running the Example

```bash
python main.py
```


## Code (`main.py`)

```python
"""
Example 07: Auto Distribution Selection

This example demonstrates the `dist='auto'` feature, which automatically selects
the best-fitting distribution (Lognormal vs Normal) for ROS imputation based on the
Regression on Order Statistics $R^2$ score.

We demonstrate two cases:
1. Left Censoring with Lognormal Data (Auto-Detect should pick Lognormal)
2. Mixed Censoring with Normal Data (Auto-Detect should pick Normal)
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure we can import ndimpute from the local source
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ndimpute import impute

def main():
    print("--- Example 07: Auto Distribution Selection ---\n")

    # --- Case 1: Left Censoring (Lognormal Data) ---
    print("1. Testing Left Censoring (True Dist: Lognormal)")

    np.random.seed(101)
    # Generate skewed lognormal data
    true_vals_log = np.random.lognormal(mean=2.0, sigma=1.2, size=50)

    # Censor values < 5.0 (approx 20-30%)
    lod = 5.0
    values_log = []
    for v in true_vals_log:
        if v < lod:
            values_log.append(f"<{lod}")
        else:
            values_log.append(v)

    print(f"   Input Data: {len(values_log)} samples, Left Censored at {lod}")

    # Run Imputation with Auto
    df_log = impute(values_log, dist='auto')

    best_dist_1 = df_log.attrs.get('best_dist')
    score_1 = df_log.attrs.get('fit_score')

    print(f"   -> Auto Selected: '{best_dist_1}'")
    print(f"   -> Fit Score (R^2): {score_1:.4f}")

    # Verify
    if best_dist_1 == 'lognormal':
        print("   [SUCCESS] Correctly identified Lognormal.")
    else:
        print("   [WARNING] Unexpected distribution selection.")

    print("-" * 40 + "\n")

    # --- Case 2: Mixed Censoring (Normal Data) ---
    print("2. Testing Mixed Censoring (True Dist: Normal)")

    # Generate Normal data
    true_vals_norm = np.random.normal(loc=100, scale=15, size=50)

    # Apply Mixed Censoring
    # Left: < 80
    # Right: > 120
    status_mixed = [] # -1, 0, 1
    values_mixed = []

    for v in true_vals_norm:
        if v < 80:
            status_mixed.append(-1) # Left
            values_mixed.append(80)
        elif v > 120:
            status_mixed.append(1) # Right
            values_mixed.append(120)
        else:
            status_mixed.append(0) # Obs
            values_mixed.append(v)

    print(f"   Input Data: {len(values_mixed)} samples")
    print(f"   Left Limit: 80, Right Limit: 120")

    # Run Imputation with Auto
    # Note: explicit status requires censoring_type='mixed' (now default if not specified)
    # or it will infer from integer status.
    df_norm = impute(values_mixed, status=status_mixed, dist='auto')

    best_dist_2 = df_norm.attrs.get('best_dist')
    score_2 = df_norm.attrs.get('fit_score')

    print(f"   -> Auto Selected: '{best_dist_2}'")
    print(f"   -> Fit Score (R^2): {score_2:.4f}")

    # Verify
    if best_dist_2 == 'normal':
        print("   [SUCCESS] Correctly identified Normal.")
    else:
        print("   [WARNING] Unexpected distribution selection.")

    print("\nDone.")

if __name__ == "__main__":
    main()

```
