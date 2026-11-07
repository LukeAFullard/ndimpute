import numpy as np
import pandas as pd
from ndimpute import impute
import sys

def main():
    # 1. Simulate Data
    np.random.seed(42)
    N = 1000
    mean_log = 4.0
    sigma_log = 1.0

    # True data
    true_values = np.random.lognormal(mean_log, sigma_log, N)

    # Censor lowest 50% for stronger effect
    limit = np.percentile(true_values, 50)
    obs_values = true_values.copy()
    status = np.zeros(N, dtype=bool)

    cens_mask = obs_values < limit
    obs_values[cens_mask] = limit
    status[cens_mask] = True # True = Censored

    print(f"Dataset: N={N}, Censoring Limit={limit:.4f}, Censored Fraction={np.mean(cens_mask):.1%}")
    print("-" * 60)

    # 2. Single Stochastic Imputation
    # Explicitly set censoring_type='left' to avoid Mixed warning/logic
    df_stochastic = impute(obs_values, status, method='ros', censoring_type='left', dist='lognormal',
                           impute_type='stochastic', random_state=123)
    vals_stochastic = df_stochastic['imputed_value'].values

    # 3. Bagging (Averaging 50 stochastic imputations)
    print("Running Bagging (50 iterations)...")
    bagged_accumulator = np.zeros(N)
    n_bags = 50

    for i in range(n_bags):
        df_temp = impute(obs_values, status, method='ros', censoring_type='left', dist='lognormal',
                         impute_type='stochastic', random_state=i)
        bagged_accumulator += df_temp['imputed_value'].values

    vals_bagged = bagged_accumulator / n_bags

    # 4. Compare Statistics (on Log Scale to verify Lognormal properties)
    log_true = np.log(true_values)
    log_stoch = np.log(vals_stochastic)
    log_bagged = np.log(vals_bagged)

    stats_log = pd.DataFrame({
        'True (Log)': log_true,
        'Stochastic (Log)': log_stoch,
        'Bagged (Log)': log_bagged
    })

    print("\nStatistical Comparison (Log Scale):")
    print(stats_log.describe().loc[['mean', 'std', 'min', 'max']])

    true_std = np.std(log_true)
    stoch_std = np.std(log_stoch)
    bag_std = np.std(log_bagged)

    print("\nVariance Analysis (Log Scale):")
    print(f"True Std Dev:      {true_std:.4f}")
    print(f"Stochastic Std:    {stoch_std:.4f} (Error: {((stoch_std-true_std)/true_std):.1%})")
    print(f"Bagged Std:        {bag_std:.4f}   (Error: {((bag_std-true_std)/true_std):.1%})")

    print("\nConclusion:")
    if bag_std < stoch_std:
        print(">> Bagging significantly UNDERESTIMATES the variance.")
        print("   Notice how the 'min' value for Bagged is much higher than True/Stochastic.")
        print("   The averaged values cluster around the regression line, losing the tail.")
    else:
        print(">> Variance preserved (unexpected).")

    # 5. Visualization
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))

        sns.kdeplot(log_true, label='True Distribution', color='black', linestyle='--')
        sns.kdeplot(log_stoch, label='Single Stochastic Imputation', color='blue', alpha=0.6)
        sns.kdeplot(log_bagged, label='Bagged (Avg) Imputation', color='red', alpha=0.6)

        plt.axvline(np.log(limit), color='gray', linestyle=':', label='Detection Limit')
        plt.title('Bagging vs Stochastic Imputation (Log Scale)')
        plt.xlabel('Log(Value)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_file = 'bagging_comparison.png'
        plt.savefig(output_file)
        print(f"\nPlot saved to {output_file}")

    except ImportError:
        print("\nMatplotlib/Seaborn not installed. Skipping plot.")

if __name__ == "__main__":
    main()
