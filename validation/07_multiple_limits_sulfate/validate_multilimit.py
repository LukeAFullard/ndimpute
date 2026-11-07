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

    # Check 2: Parity with NADA::ros (R, NADA 1.6-1.1).
    # NADA's unclamped $modeled for these data, mapped to the original row order.
    # Hirsch-Stedinger places the "<5" values in the lower part of the
    # distribution BELOW 5, which overlaps "<1" and the detect at 2 -- a "<5"
    # is not required to be imputed above 2.
    nada_modeled = np.array([0.843607, 1.329279, 1.329279, 2.479250, 2, 4, 6, 8, 10])
    df_nada = impute(values, status, method='ros', censoring_type='left', clamp=False)
    max_diff = np.max(np.abs(df_nada['imputed_value'].values - nada_modeled))
    print(f"\nMax |ndimpute(clamp=False) - NADA::ros|: {max_diff:.2e}")
    if max_diff > 1e-5:
        failures.append(f"Deviation from NADA::ros ({max_diff:.2e}).")

    imputed_1 = df_ros.loc[mask_1, 'imputed_value'].mean()
    imputed_5 = df_ros.loc[mask_5, 'imputed_value'].mean()
    print(f"Mean Imputed (<1): {imputed_1:.4f}")
    print(f"Mean Imputed (<5): {imputed_5:.4f}")

    if not failures:
        print("[PASS] Guardrails respected and NADA::ros reproduced.")
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
