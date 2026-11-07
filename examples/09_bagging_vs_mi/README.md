# Example 09: Bagging vs Multiple Imputation

This example answers the question: **"Is bagging useful at all to get a better result when imputing?"**

## Scenario
We have a simulated lognormal dataset ($N=1000$) with heavy Left Censoring (50% of data < LOD). We compare two approaches:
1.  **Single Stochastic Imputation:** A single run of Robust ROS with `impute_type='stochastic'`.
2.  **Bagged Imputation:** The average of 50 independent stochastic imputation runs.

## The Short Answer

It depends on what you define as "better":
*   **Yes**, if you want the most stable, accurate point estimate for each missing value (minimizing Root Mean Squared Error).
*   **No**, if you want to preserve the correct statistical variance and distribution shape for downstream analysis (e.g., hypothesis testing, calculating confidence intervals, or risk assessment).

### What is "Bagging"?
In this context, "Bagging" (Bootstrap Aggregating) means running a stochastic imputation method multiple times and **averaging** the results into a single dataset:
$$ \hat{y}_{bagged} = \frac{1}{M} \sum_{m=1}^{M} \hat{y}_{m} $$

This is distinct from **Multiple Imputation (MI)**, where you analyze $M$ datasets separately and pool the parameters.

### Pros and Cons
*   **Pros of Bagging:** Stability, Lower RMSE, Simplicity (single dataset).
*   **Cons of Bagging:** **Variance Collapse** (underestimates std dev), Distributional Bias (loses tails), Invalid Inference (p-values too low).

## Code (`main.py`)

```python
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

if __name__ == "__main__":
    main()
```

## Output

```text
Dataset: N=1000, Censoring Limit=55.9972, Censored Fraction=50.0%
------------------------------------------------------------
Running Bagging (50 iterations)...

Statistical Comparison (Log Scale):
      True (Log)  Stochastic (Log)  Bagged (Log)
mean    4.019332          3.974781      4.052376
std     0.979216          1.032828      0.865335
min     0.758733          0.086857      3.074028
max     7.852731          7.852731      7.852731

Variance Analysis (Log Scale):
True Std Dev:      0.9787
Stochastic Std:    1.0323 (Error: 5.5%)
Bagged Std:        0.8649   (Error: -11.6%)

Conclusion:
>> Bagging significantly UNDERESTIMATES the variance.
   Notice how the 'min' value for Bagged is much higher than True/Stochastic.
   The averaged values cluster around the regression line, losing the tail.
```

## Interpretation

The output clearly demonstrates the "Variance Collapse":

1.  **Standard Deviation:** The Bagged data has a standard deviation of **0.86**, significantly lower than the True value of **0.98**. The Single Stochastic imputation (1.03) is much closer.
2.  **Minimum Value:** The Bagged data has a minimum log-value of **3.07**, whereas the True minimum is **0.76**. This means Bagging completely failed to reconstruct the lower tail of the distribution.
3.  **Bias:** By averaging, the values converge toward the conditional mean (the regression line), ignoring the random residual noise inherent in the population.

**Recommendation:** Do not use bagging if you intend to calculate percentiles, confidence intervals, or p-values. Use Single Stochastic Imputation for visualization/simple stats, or proper Multiple Imputation for rigorous inference.

> **Note on Multiple Imputation.** Repeating `impute(..., impute_type='stochastic')` with different seeds does **not**, by itself, give *proper* multiple imputations in the sense of Rubin (1987): every run conditions on the same fitted parameters, so the between-imputation variance omits parameter uncertainty and Rubin's-rules standard errors will be too small. For proper MI, resample the data (bootstrap) before each imputation, or draw the model parameters from their sampling distribution, and then impute.

## References

1.  **Little, R. J. A., & Rubin, D. B. (2019).** *Statistical Analysis with Missing Data*. Wiley.
2.  **Van Buuren, S. (2018).** *Flexible Imputation of Missing Data*. CRC Press.
