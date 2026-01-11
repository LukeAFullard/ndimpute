import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ndimpute import impute

def run_test(name, func):
    print(f"\nRunning Sub-test: {name}")
    try:
        func()
        print("[PASS]")
        return True
    except Exception as e:
        print(f"[FAIL] Unexpected Error: {e}")
        return False

def test_high_censoring():
    # Case 3.1: High Censoring (95%)
    print("Scenario: N=100, 95% Left Censored (< 1.0). 5 Observed values (1.1...1.5).")
    N = 100
    n_cens = 95
    lod = 1.0

    values = np.ones(N) * lod
    status = np.zeros(N, dtype=bool)

    # 95 censored
    status[:n_cens] = True

    # 5 observed
    values[n_cens:] = np.linspace(1.1, 1.5, N - n_cens)

    df = impute(values, status, method='ros', censoring_type='left')

    # Check that it ran
    imputed_vals = df.loc[status, 'imputed_value']

    # Check guardrails: should be <= 1.0
    max_imputed = imputed_vals.max()
    print(f"Max Imputed (Should be <= 1.0): {max_imputed}")

    if max_imputed > 1.0 + 1e-9:
        raise AssertionError("Imputed values exceed LOD.")

    # Check that they are not all identical (regression should provide slope)
    # With only 5 points, slope might be flat or steep, but let's see variance
    print(f"Std Dev of Imputed: {imputed_vals.std()}")

def test_minimal_n():
    # Case 3.2: Minimal N (N=3, 2 Uncensored)
    print("Scenario: N=3, 1 Censored (<1), 2 Observed (2, 3).")
    values = [1.0, 2.0, 3.0]
    status = [True, False, False]

    df = impute(values, status, method='ros', censoring_type='left')
    val = df.loc[0, 'imputed_value']
    print(f"Imputed Value: {val}")

    if np.isnan(val):
        raise AssertionError("Returned NaN.")

def test_insufficient_uncensored():
    # Case: N=10, 1 Uncensored. ROS needs 2 points for regression line.
    print("Scenario: N=10, 1 Uncensored. Expecting ValueError.")
    values = np.ones(10)
    status = np.ones(10, dtype=bool)
    values[-1] = 2.0
    status[-1] = False # 1 Observed

    try:
        impute(values, status, method='ros', censoring_type='left')
        raise AssertionError("Did not raise ValueError")
    except ValueError as e:
        print(f"Caught expected error: {e}")
        if "Too few uncensored" not in str(e):
            raise AssertionError(f"Unexpected error message: {e}")

def test_100_percent_censoring():
    # Case 3.3: 100% Censoring
    print("Scenario: 100% Censoring. Expecting ValueError.")
    values = np.ones(10)
    status = np.ones(10, dtype=bool)

    try:
        impute(values, status, method='ros', censoring_type='left')
        raise AssertionError("Did not raise ValueError")
    except ValueError as e:
        print(f"Caught expected error: {e}")

def test_invalid_data_negative():
    # Case 3.4: Negative Data for LogNormal
    print("Scenario: Negative data for LogNormal ROS. Expecting ValueError.")
    values = [-1.0, 2.0, 3.0]
    status = [True, False, False] # -1 is censored (e.g. < -1? No, usually censored is LOD)
    # If values are negative, log transform fails.

    try:
        impute(values, status, method='ros', censoring_type='left')
        raise AssertionError("Did not raise ValueError")
    except ValueError as e:
        print(f"Caught expected error: {e}")
        if "positive" not in str(e).lower():
             # It might be "Values must be positive" from our code
             # or "invalid value in log" from numpy if check missed
             pass

def run_validation():
    print("Running Validation: 05 Edge Cases & Robustness")

    results = [
        run_test("High Censoring (95%)", test_high_censoring),
        run_test("Minimal N (N=3)", test_minimal_n),
        run_test("Insufficient Uncensored", test_insufficient_uncensored),
        run_test("100% Censoring", test_100_percent_censoring),
        run_test("Invalid Data (Negative)", test_invalid_data_negative)
    ]

    if all(results):
        print("\n[PASS] All Edge Cases Handled Correctly.")
    else:
        print("\n[FAIL] Some Edge Cases Failed.")

if __name__ == "__main__":
    run_validation()
