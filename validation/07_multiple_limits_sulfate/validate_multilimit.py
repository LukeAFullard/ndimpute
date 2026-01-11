import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import probplot
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ndimpute import impute

def run_validation():
    print("Running Validation: 07 Multiple Detection Limits (Sulfate-like)")

    # 1. Create Synthetic Data with Multiple Limits
    # Mimics environmental data where LOD changes over time or method.
    # Observed: [2, 4, 6, 8, 10]
    # Censored: [<1, <1, <5, <5]
    # Sorted by Value: <1, <1, 2, 4, <5, <5, 6, 8, 10

    values = np.array([1.0, 1.0, 5.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    status = np.array([True, True, True, True, False, False, False, False, False]) # True=Censored

    print("Input Data:")
    df_in = pd.DataFrame({'Value': values, 'Censored': status})
    print(df_in)

    # 2. Run ROS
    df_ros = impute(values, status, method='ros', censoring_type='left')

    # 3. Analyze Results
    print("\n--- Imputation Results ---")
    print(df_ros[['original_value', 'censoring_status', 'imputed_value']])

    # Check 1: Guardrails
    # Imputed values for <1 should be <= 1
    # Imputed values for <5 should be <= 5

    failures = []

    # Check < 1.0
    mask_1 = (df_ros['original_value'] == 1.0) & (df_ros['censoring_status'])
    if not np.all(df_ros.loc[mask_1, 'imputed_value'] <= 1.0):
        failures.append("Imputations for <1.0 exceed limit.")

    # Check < 5.0
    mask_5 = (df_ros['original_value'] == 5.0) & (df_ros['censoring_status'])
    if not np.all(df_ros.loc[mask_5, 'imputed_value'] <= 5.0):
        failures.append("Imputations for <5.0 exceed limit.")

    # Check 2: Relative Order (The "Simple" logic check)
    # Our simplified logic sorts by Value.
    # So <1 (Obs=1) comes first.
    # 2 (Obs=2) comes next.
    # <5 (Obs=5) comes after 2 and 4!
    # This implies that the regression interprets <5 as a point with Rank higher than 2.
    # This creates a risk: if the regression line is increasing, the Z-score for <5 will be higher than for 2.
    # But the *value* of <5 must be imputed.
    # If Z(<5) > Z(2), then Predicted(<5) > Predicted(2) approx 2.0.
    # So we expect Imputed(<5) to be around 2~5.

    imputed_1 = df_ros.loc[mask_1, 'imputed_value'].mean()
    imputed_5 = df_ros.loc[mask_5, 'imputed_value'].mean()
    obs_2 = 2.0

    print(f"\nMean Imputed (<1): {imputed_1:.4f}")
    print(f"Observed (2):      {obs_2:.4f}")
    print(f"Mean Imputed (<5): {imputed_5:.4f}")

    if imputed_1 < obs_2 and imputed_5 > obs_2:
        print("\n[INFO] Ordering preserved: <1  <  2.0  <  <5")
    else:
        print("\n[WARN] Ordering issue: simplified sorting might have distorted ranks.")

    if not failures:
        print("[PASS] Guardrails respected.")
    else:
        print(f"[FAIL] {failures}")

    # 4. Visualization
    plt.figure(figsize=(10, 6))

    # Plot Observed
    mask_obs = ~df_ros['censoring_status']
    plt.scatter(df_ros.loc[mask_obs, 'imputed_value'], df_ros.loc[mask_obs, 'imputed_value'],
                color='green', label='Observed', marker='o')

    # Plot Imputed
    mask_cens = df_ros['censoring_status']
    plt.scatter(df_ros.loc[mask_cens, 'original_value'], df_ros.loc[mask_cens, 'imputed_value'],
                color='red', label='Imputed (< Limit)', marker='x')

    # Limits line (x=y)
    max_val = 10.0
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Limit Line')

    plt.title("Multiple Detection Limits: Limit vs Imputed")
    plt.xlabel("Detection Limit / Observed Value")
    plt.ylabel("Imputed Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("multilimit_check.png")
    print("Saved multilimit_check.png")

if __name__ == "__main__":
    run_validation()
