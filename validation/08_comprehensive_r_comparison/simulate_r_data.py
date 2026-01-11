import numpy as np
import pandas as pd
from scipy.stats import norm, linregress
import os

# --- Reference ROS Implementation (Simulating NADA logic) ---
def reference_ros(values, is_censored, dist='lognormal'):
    """
    Manual implementation of NADA-style ROS for generating benchmarks.
    Uses Weibull Plotting Positions (rank / (N+1)).
    """
    df = pd.DataFrame({'val': values, 'cens': is_censored})
    n = len(df)

    # Sort
    df = df.sort_values('val')

    # Weibull Plotting Positions
    df['rank'] = np.arange(1, n + 1)
    df['pp'] = df['rank'] / (n + 1)
    df['z'] = norm.ppf(df['pp'])

    # Fit Regression
    # NADA fits on UNcensored data
    unc_mask = ~df['cens']

    if dist == 'lognormal':
        y = np.log(df.loc[unc_mask, 'val'])
    else:
        y = df.loc[unc_mask, 'val']

    x = df.loc[unc_mask, 'z']

    if len(x) < 2:
        return np.full(n, np.nan) # Cannot fit

    slope, intercept, _, _, _ = linregress(x, y)

    # Predict
    z_all = df['z']
    pred = intercept + slope * z_all

    if dist == 'lognormal':
        imputed_all = np.exp(pred)
    else:
        imputed_all = pred

    # NADA returns Observed values where observed, and Imputed where censored
    final_series = df['val'].copy()
    final_series[df['cens']] = imputed_all[df['cens']]

    # Important: NADA might NOT clamp at detection limit?
    # Usually ROS is extrapolated. We will NOT clamp here to simulate "pure" regression
    # output which ndimpute might improve upon with guardrails.

    return final_series.sort_index().values

# --- Dataset Generator ---
def generate_benchmarks():
    output_dir = os.path.dirname(__file__)
    print(f"Generating benchmarks in: {output_dir}")

    np.random.seed(42)

    dists = ['lognormal', 'normal']
    ns = [20, 50, 200]
    cens_levels = [0.2, 0.5, 0.8]
    limit_types = ['single', 'multiple']

    count = 0

    for d in dists:
        for n in ns:
            for c in cens_levels:
                for l in limit_types:
                    # 1. Generate True Data
                    if d == 'lognormal':
                        true_vals = np.random.lognormal(mean=2, sigma=1, size=n)
                    else:
                        true_vals = np.random.normal(loc=10, scale=2, size=n)

                    # 2. Apply Censoring
                    if l == 'single':
                        limit = np.quantile(true_vals, c)
                        is_cens = true_vals < limit
                        obs_vals = true_vals.copy()
                        obs_vals[is_cens] = limit
                    else:
                        # Multiple limits
                        target_q = np.quantile(true_vals, c)
                        # Limits around the target
                        potential_limits = [target_q * 0.8, target_q, target_q * 1.2]
                        assigned_limits = np.random.choice(potential_limits, size=n)
                        is_cens = true_vals < assigned_limits
                        obs_vals = true_vals.copy()
                        obs_vals[is_cens] = assigned_limits[is_cens]

                    # Skip if too few observed
                    if np.sum(~is_cens) < 2:
                        continue

                    # 3. Run Reference ROS
                    r_imputed = reference_ros(obs_vals, is_cens, dist=d)

                    # 4. Export
                    filename = f"benchmark_{d}_n{n}_cens{c}_{l}.csv"
                    filepath = os.path.join(output_dir, filename)

                    df_out = pd.DataFrame({
                        'original_value': obs_vals,
                        'censoring_status': is_cens,
                        'r_imputed': r_imputed
                    })

                    df_out.to_csv(filepath, index=False)
                    count += 1

    print(f"Generated {count} benchmark files.")

if __name__ == "__main__":
    generate_benchmarks()
