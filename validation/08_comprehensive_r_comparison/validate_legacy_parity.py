import pandas as pd
import numpy as np
import os
import glob
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ndimpute import impute

def run_parity_check():
    print("Running Legacy Parity Validation (plotting_position='simple')")

    base_dir = os.path.dirname(__file__)
    files = glob.glob(os.path.join(base_dir, "benchmark_*.csv"))

    if not files:
        print("No benchmarks found. Generating...")
        import simulate_r_data
        simulate_r_data.generate_benchmarks()
        files = glob.glob(os.path.join(base_dir, "benchmark_*.csv"))

    results = []

    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        parts = filename.replace("benchmark_", "").replace(".csv", "").split("_")
        dist = parts[0]
        limit_type = parts[3]

        df = pd.read_csv(filepath)
        values = df['original_value'].values
        status = df['censoring_status'].astype(bool).values
        r_imputed = df['r_imputed'].values

        try:
            # Explicitly request 'simple' plotting positions to match the benchmark generation logic
            df_py = impute(values, status, method='ros', censoring_type='left',
                           dist=dist, plotting_position='simple')

            py_imputed = df_py['imputed_value'].values

            mask = status
            if np.sum(mask) == 0:
                mae = 0.0
            else:
                mae = np.mean(np.abs(py_imputed[mask] - r_imputed[mask]))

            results.append({'Scenario': filename, 'Dist': dist, 'Limit': limit_type, 'MAE': mae})
            print(f"Processed {filename}: MAE={mae:.6f}")

        except Exception as e:
            print(f"Failed {filename}: {e}")

    df_res = pd.DataFrame(results)

    # Calculate Mean MAE by Scenario
    summary = df_res.groupby(['Dist', 'Limit'])['MAE'].mean().reset_index()
    print("\n--- Mean MAE (Legacy Mode) ---")
    print(summary)

    # Validation Check
    if summary['MAE'].max() < 0.001:
        print("\n[PASS] Legacy mode matches simple benchmarks perfectly.")
    else:
        print("\n[WARN] Legacy mode has some deviation.")

if __name__ == "__main__":
    run_parity_check()
