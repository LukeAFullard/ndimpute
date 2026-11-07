import pandas as pd
import numpy as np
import os
import glob
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ndimpute import impute

def run_suite():
    print("Running Comprehensive Validation Suite (Updated)")

    # Find all benchmarks
    base_dir = os.path.dirname(__file__)
    files = glob.glob(os.path.join(base_dir, "benchmark_*.csv"))

    if not files:
        print("No benchmark files found. Please run 'generate_r_suite.R' first.")
        return

    results = []

    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        # Parse filename: benchmark_lognormal_n50_cens0.5_single.csv
        parts = filename.replace("benchmark_", "").replace(".csv", "").split("_")
        dist = parts[0] # 'lognormal' or 'normal'
        # n = parts[1] (e.g. n50)
        # cens = parts[2] (e.g. cens0.5)
        limit_type = parts[3]

        # Load Data
        df = pd.read_csv(filepath)
        values = df['original_value'].values
        status = df['censoring_status'].astype(bool).values
        r_imputed = df['r_imputed'].values

        # Run ndimpute with correct distribution!
        try:
            # NADA-equivalent configuration: Hirsch-Stedinger positions, quantile
            # imputation, no clamping at the detection limit.
            df_py = impute(values, status, method='ros', censoring_type='left', dist=dist,
                           plotting_position='kaplan-meier', impute_type='quantile', clamp=False)
            py_imputed = df_py['imputed_value'].values

            # Compare (Censored only)
            mask = status
            if np.sum(mask) == 0:
                mae = 0.0
            else:
                mae = np.mean(np.abs(py_imputed[mask] - r_imputed[mask]))
            max_diff = float(np.max(np.abs(py_imputed - r_imputed)))

            results.append({
                'Scenario': filename,
                'Dist': dist,
                'N': len(values),
                'Limit': limit_type,
                'MAE': mae,
                'MaxAbsDiff': max_diff,
                'Status': 'PASS' if max_diff < 1e-8 else 'FAIL'
            })
            print(f"Processed {filename} (dist={dist}): MAE={mae:.2e}, max|diff|={max_diff:.2e}")

        except Exception as e:
            print(f"Failed {filename}: {e}")
            results.append({
                'Scenario': filename,
                'Status': f"ERROR: {e}"
            })

    # Summary
    df_res = pd.DataFrame(results)
    print("\n--- Summary ---")
    print(df_res.to_string())

    # Export results
    df_res.to_csv("suite_results.csv", index=False)
    print("\nSaved suite_results.csv")
    n_fail = int((df_res['Status'] != 'PASS').sum())
    print(f"\n[{'PASS' if n_fail == 0 else 'FAIL'}] {len(df_res) - n_fail}/{len(df_res)} "
          "datasets match NADA::ros to 1e-8.")

if __name__ == "__main__":
    run_suite()
