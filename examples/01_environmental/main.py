import pandas as pd
import numpy as np
import sys
import os

# Add src to path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ndimpute import impute

def main():
    print("--- 01 Environmental Data Example (Left Censoring) ---")

    # 1. Define Data
    # Typical environmental dataset with non-detects (marked with '<')
    # Using the simplified String API (Auto-Detect)
    raw_data = [
        "<0.5", "0.7", "0.9", "<0.5", "1.2",
        "1.5", "<0.5", "2.1", "0.8", "1.1",
        "0.6", "<0.5", "1.8", "2.5", "0.9"
    ]

    print(f"\nRaw Data ({len(raw_data)} samples):\n{raw_data}")

    # 2. Impute using Robust ROS (default)
    # The library automatically detects '<' as left-censoring
    # and defaults to lognormal distribution for environmental data.
    df_ros = impute(raw_data, dist='lognormal', random_state=42)

    print("\n--- Imputation Results (First 5 Rows) ---")
    print(df_ros[['original_value', 'censoring_status', 'imputed_value']].head())

    # 3. Compare with Substitution (1/2 LOD)
    # Explicitly requesting substitution method
    df_sub = impute(raw_data, method='substitution', strategy='half')

    # 4. Calculate Statistics
    mean_ros = df_ros['imputed_value'].mean()
    mean_sub = df_sub['imputed_value'].mean()

    print("\n--- Statistical Comparison ---")
    print(f"Mean (Robust ROS):      {mean_ros:.4f}")
    print(f"Mean (Substitution):    {mean_sub:.4f}")

    print("\nNote: ROS provides a statistically more rigorous estimate of the mean")
    print("by fitting the distribution of the observed data, whereas 1/2 substitution")
    print("is an arbitrary heuristic that often biases results.")

if __name__ == "__main__":
    main()
