import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ndimpute import impute

def run_monte_carlo(n_simulations=1000, n_samples=50, censoring_level=0.3):
    print(f"Running Monte Carlo Simulation ({n_simulations} iterations)...")
    print(f"Distribution: Lognormal(mean=2, sigma=1), N={n_samples}, Censoring={censoring_level*100}%")

    results = []

    np.random.seed(42)

    for i in range(n_simulations):
        # 1. Generate Data (Lognormal)
        # Note: np.random.lognormal takes mean and sigma of the underlying normal distribution
        true_mean_log = 2.0
        true_sigma_log = 1.0

        # True Mean of Lognormal Distribution: exp(mu + sigma^2/2)
        true_mean_dist = np.exp(true_mean_log + (true_sigma_log**2)/2)

        true_values = np.random.lognormal(mean=true_mean_log, sigma=true_sigma_log, size=n_samples)

        # 2. Apply Censoring
        # Determine LOD based on quantile to achieve desired censoring %
        lod = np.quantile(true_values, censoring_level)

        status = true_values < lod # True if censored
        values = true_values.copy()
        values[status] = lod

        # 3. Method A: Substitution (LOD/2)
        df_sub = impute(values, status, method='substitution', censoring_type='left', strategy='half')
        mean_sub = df_sub['imputed_value'].mean()

        # 4. Method B: Robust ROS
        try:
            df_ros = impute(values, status, method='ros', censoring_type='left')
            mean_ros = df_ros['imputed_value'].mean()

            # Record Results
            results.append({
                'sim_id': i,
                'bias_sub': mean_sub - true_mean_dist,
                'bias_ros': mean_ros - true_mean_dist,
                'sq_err_sub': (mean_sub - true_mean_dist)**2,
                'sq_err_ros': (mean_ros - true_mean_dist)**2
            })
        except ValueError:
            # Skip if regression fails (e.g. too few uncensored points, unlikely with 30% cens)
            continue

    df_res = pd.DataFrame(results)

    # 5. Analysis
    rmse_sub = np.sqrt(df_res['sq_err_sub'].mean())
    rmse_ros = np.sqrt(df_res['sq_err_ros'].mean())

    avg_bias_sub = df_res['bias_sub'].mean()
    avg_bias_ros = df_res['bias_ros'].mean()

    print("\n--- Validation Results ---")
    print(f"Substitution (LOD/2) RMSE: {rmse_sub:.4f}")
    print(f"Robust ROS RMSE:           {rmse_ros:.4f}")
    print(f"Improvement Factor:        {rmse_sub / rmse_ros:.2f}x")

    print(f"\nSubstitution Mean Bias:    {avg_bias_sub:.4f}")
    print(f"Robust ROS Mean Bias:      {avg_bias_ros:.4f}")

    # 6. Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(df_res['bias_sub'], bins=30, alpha=0.5, label='Substitution (LOD/2)', color='red')
    plt.hist(df_res['bias_ros'], bins=30, alpha=0.5, label='Robust ROS', color='blue')
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f"Bias Distribution (N={n_simulations} Simulations)")
    plt.xlabel("Bias (Estimated Mean - True Mean)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("bias_distribution.png")
    print("Saved bias_distribution.png")

    # Success Criteria
    # ROS should have lower RMSE and Bias closer to 0
    if rmse_ros < rmse_sub:
        print("\n[PASS] ROS outperforms Substitution.")
    else:
        print("\n[FAIL] ROS did not outperform Substitution.")

if __name__ == "__main__":
    run_monte_carlo()
