"""
Example 03: Mixed Censoring

This example demonstrates handling complex data with both Upper and Lower bounds.
We use "Auto-Detect" with both '<' and '>' markers.

Method: Parametric Mixed (Lognormal).
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure we can import ndimpute from the local source
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ndimpute import impute

def main():
    print("--- Example 03: Mixed Censoring (Auto-Detect) ---\n")

    # 1. Create Synthetic Data (Lognormal)
    np.random.seed(99)
    true_vals = np.random.lognormal(mean=3.5, sigma=1.0, size=50) # Median ~33

    lower_limit = 10.0
    upper_limit = 100.0

    raw_data = []

    for val in true_vals:
        if val < lower_limit:
            raw_data.append(f"<{lower_limit}")
        elif val > upper_limit:
            raw_data.append(f">{upper_limit}")
        else:
            raw_data.append(f"{val:.2f}")

    print(f"Input Data Sample (Mixed Markers):")
    print(raw_data[:10])
    print("")

    # 2. Parametric Imputation
    # Auto-detects mixed censoring type
    print("Running Parametric Mixed Imputation...")
    df = impute(raw_data, method='parametric', dist='lognormal')

    # 3. View Specific Points
    print("\n--- Results ---")

    # Find a Left Censored Point
    left_mask = (df['censoring_status'] == -1)
    if left_mask.any():
        idx = left_mask.idxmax()
        print(f"Left Censored (Index {idx}):")
        print(f"  Input:   {raw_data[idx]}")
        print(f"  Imputed: {df.loc[idx, 'imputed_value']:.4f}")

    # Find a Right Censored Point
    right_mask = (df['censoring_status'] == 1)
    if right_mask.any():
        idx = right_mask.idxmax()
        print(f"Right Censored (Index {idx}):")
        print(f"  Input:   {raw_data[idx]}")
        print(f"  Imputed: {df.loc[idx, 'imputed_value']:.4f}")

if __name__ == "__main__":
    main()
