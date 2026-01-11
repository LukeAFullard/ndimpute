"""
Example 02: Reliability Engineering (Right Censoring)

This example demonstrates handling time-to-failure data where some units survived the test.
We use the "Auto-Detect" syntax with '>' indicators.

Method: Parametric Weibull Imputation.
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure we can import ndimpute from the local source
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ndimpute import impute

def main():
    print("--- Example 02: Reliability (Right Censoring) ---\n")

    # 1. Create Synthetic Data (Weibull)
    np.random.seed(101)
    N = 50
    true_times = 100 * np.random.weibull(2, size=N)

    # Simulate censorship at T=120
    cutoff = 120.0
    raw_data = []

    for t in true_times:
        if t > cutoff:
            raw_data.append(f">{cutoff}")
        else:
            raw_data.append(f"{t:.2f}")

    print(f"Input Data Sample (with '>' markers):")
    print(raw_data[:10])
    print("")

    # 2. Parametric Imputation
    # Auto-detects '>' as Right Censoring
    print("Running Parametric Weibull Imputation...")
    df = impute(raw_data, method='parametric', dist='weibull')

    # 3. Results
    censored_mask = df['censoring_status']
    avg_projected = df.loc[censored_mask, 'imputed_value'].mean()

    print("\n--- Results ---")
    print(f"Cutoff Time: {cutoff}")
    print(f"Average Projected Life for Survivors: {avg_projected:.2f}")

    print("\nDataFrame Output (Censored Subset):")
    print(df[censored_mask].head())

if __name__ == "__main__":
    main()
