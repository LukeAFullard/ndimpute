import pandas as pd
import numpy as np
import os
import sys

# Ensure we can import ndimpute
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ndimpute import impute

def validate_left_ros(filepath):
    print(f"--- Validating Left ROS: {filepath} ---")
    if not os.path.exists(filepath):
        print("Skipping: File not found (Run R script first)")
        return

    df_bench = pd.read_csv(filepath)
    values = df_bench['original_value'].values
    # In R script I exported TRUE/1 for censored.
    # Python API expects boolean for Left
    status = df_bench['censoring_status'].astype(bool).values

    # Run Python Imputation
    df_py = impute(values, status, method='ros', censoring_type='left')

    # Comparison
    # Compare imputed values for censored data
    mask_cens = status

    r_vals = df_bench.loc[mask_cens, 'r_imputed'].values
    py_vals = df_py.loc[mask_cens, 'imputed_value'].values

    # Calculate difference
    # Note: ROS implementations may differ slightly in plotting positions (Hazen vs Weibull etc)
    # We expect high correlation but possibly not identical floating point.
    mae = np.mean(np.abs(r_vals - py_vals))
    corr = np.corrcoef(r_vals, py_vals)[0, 1]

    print(f"MAE: {mae:.4f}")
    print(f"Correlation: {corr:.4f}")

    if mae < 0.5: # Loose tolerance due to plotting position diffs
        print("PASS")
    else:
        print("WARN: Difference might be significant. Check plotting positions.")

def validate_right_parametric(filepath):
    print(f"--- Validating Right Parametric: {filepath} ---")
    if not os.path.exists(filepath):
        print("Skipping: File not found")
        return

    df_bench = pd.read_csv(filepath)
    values = df_bench['original_value'].values
    status = df_bench['censoring_status'].astype(bool).values

    # Run Python Imputation
    df_py = impute(values, status, method='parametric', censoring_type='right')

    mask_cens = status
    r_vals = df_bench.loc[mask_cens, 'r_imputed'].values
    py_vals = df_py.loc[mask_cens, 'imputed_value'].values

    mae = np.mean(np.abs(r_vals - py_vals))

    print(f"MAE: {mae:.4f}")

    if mae < 0.1: # Parametric should match closely if fitting algorithms align
        print("PASS")
    else:
        print("FAIL: Significant difference in parametric fit.")

if __name__ == "__main__":
    # Paths to look for (assumed in same dir)
    base_dir = os.path.dirname(__file__)

    validate_left_ros(os.path.join(base_dir, "benchmark_left_ros_lognormal.csv"))
    validate_right_parametric(os.path.join(base_dir, "benchmark_right_parametric_weibull.csv"))
