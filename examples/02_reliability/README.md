# Example 02: Reliability Engineering (Right Censoring)

This example demonstrates how to use `ndimpute` for reliability analysis, where test units often survive past the end of the observation period (Right Censoring).

## Scenario
We simulate a reliability test of 50 components. The test stops at $T=120$. Any component still working is marked as `">120.0"`.

We use **Parametric Imputation** assuming a Weibull distribution, which is standard in reliability engineering. This calculates the expected total lifetime $E[T | T > 120]$ for the survivors.

## Code (`main.py`)
The script generates Weibull-distributed failure times, formats survivors with the `>` prefix, and imputes the expected lifetimes.


```python
"""
Example 02: Reliability Engineering (Right Censoring)

This example demonstrates handling time-to-failure data where some units survived the test.
We use the "Auto-Detect" syntax with '>' indicators.

Method: Parametric Weibull Imputation.
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure we can import ndimpute from the local source
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ndimpute import impute

def main():
    print("--- Example 02: Reliability (Right Censoring) ---\n")

    # 1. Create Synthetic Data (Weibull)
    np.random.seed(101)
    N = 50
    true_times = 100 * np.random.weibull(2, size=N)

    # Simulate censorship at T=120
    cutoff = 120.0
    raw_data = []

    for t in true_times:
        if t > cutoff:
            raw_data.append(f">{cutoff}")
        else:
            raw_data.append(f"{t:.2f}")

    print(f"Input Data Sample (with '>' markers):")
    print(raw_data[:10])
    print("")

    # 2. Parametric Imputation
    # Auto-detects '>' as Right Censoring
    print("Running Parametric Weibull Imputation...")
    df = impute(raw_data, method='parametric', dist='weibull')

    # 3. Results
    censored_mask = df['censoring_status']
    avg_projected = df.loc[censored_mask, 'imputed_value'].mean()

    print("\n--- Results ---")
    print(f"Cutoff Time: {cutoff}")
    print(f"Average Projected Life for Survivors: {avg_projected:.2f}")

    print("\nDataFrame Output (Censored Subset):")
    print(df[censored_mask].head())

if __name__ == "__main__":
    main()

```

## Output

```text
--- Example 02: Reliability (Right Censoring) ---

Input Data Sample (with '>' markers):
['85.23', '91.95', '17.00', '43.38', '107.52', '>120.0', '60.55', '>120.0', '113.07', '45.90']

Running Parametric Weibull Imputation...

--- Results ---
Cutoff Time: 120.0
Average Projected Life for Survivors: 150.99

DataFrame Output (Censored Subset):
    imputed_value  original_value  censoring_status  is_imputed
5      150.994672           120.0              True        True
7      150.994672           120.0              True        True
13     150.994672           120.0              True        True
...
```

## Interpretation
The model estimates that the components surviving past 120 units of time will, on average, last until ~151 units. This value is derived analytically from the fitted Weibull parameters (Shape/Scale) and represents the conditional expectation.
