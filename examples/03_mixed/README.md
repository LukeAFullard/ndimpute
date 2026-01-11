# Example 03: Mixed Censoring

This example demonstrates how to handle messy real-world datasets that contain both Left Censoring (lower detection limits) and Right Censoring (saturation limits).

## Scenario
A sensor measures pressure but has a limited operating range:
*   It cannot detect below 10 psi (outputs `"<10.0"`).
*   It saturates above 100 psi (outputs `">100.0"`).

We use **Parametric Mixed Imputation** (Generalized Likelihood) to fit a single Lognormal distribution to the observed, left-censored, and right-censored data simultaneously.

## Code (`main.py`)
The script generates synthetic data and formats it with both `<` and `>` markers. `ndimpute` automatically infers `censoring_type='mixed'`.


```python
"""
Example 03: Mixed Censoring

This example demonstrates handling complex data with both Upper and Lower bounds.
We use "Auto-Detect" with both '<' and '>' markers.

Method: Parametric Mixed (Lognormal).
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure we can import ndimpute from the local source
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ndimpute import impute

def main():
    print("--- Example 03: Mixed Censoring (Auto-Detect) ---\n")

    # 1. Create Synthetic Data (Lognormal)
    np.random.seed(99)
    true_vals = np.random.lognormal(mean=3.5, sigma=1.0, size=50) # Median ~33

    lower_limit = 10.0
    upper_limit = 100.0

    raw_data = []

    for val in true_vals:
        if val < lower_limit:
            raw_data.append(f"<{lower_limit}")
        elif val > upper_limit:
            raw_data.append(f">{upper_limit}")
        else:
            raw_data.append(f"{val:.2f}")

    print(f"Input Data Sample (Mixed Markers):")
    print(raw_data[:10])
    print("")

    # 2. Parametric Imputation
    # Auto-detects mixed censoring type
    print("Running Parametric Mixed Imputation...")
    df = impute(raw_data, method='parametric', dist='lognormal')

    # 3. View Specific Points
    print("\n--- Results ---")

    # Find a Left Censored Point
    left_mask = (df['censoring_status'] == -1)
    if left_mask.any():
        idx = left_mask.idxmax()
        print(f"Left Censored (Index {idx}):")
        print(f"  Input:   {raw_data[idx]}")
        print(f"  Imputed: {df.loc[idx, 'imputed_value']:.4f}")

    # Find a Right Censored Point
    right_mask = (df['censoring_status'] == 1)
    if right_mask.any():
        idx = right_mask.idxmax()
        print(f"Right Censored (Index {idx}):")
        print(f"  Input:   {raw_data[idx]}")
        print(f"  Imputed: {df.loc[idx, 'imputed_value']:.4f}")

if __name__ == "__main__":
    main()

```

## Output

```text
--- Example 03: Mixed Censoring (Auto-Detect) ---

Input Data Sample (Mixed Markers):
['28.72', '>100.0', '43.96', '>100.0', '28.37', '30.91', '70.47', '75.61', '29.58', '<10.0']

Running Parametric Mixed Imputation...

--- Results ---
Left Censored (Index 9):
  Input:   <10.0
  Imputed: 7.0636
Right Censored (Index 1):
  Input:   >100.0
  Imputed: 163.7200
```

## Interpretation
*   For the left-censored value (`<10.0`), the imputed value is ~7.06, consistent with the tail of the distribution below 10.
*   For the right-censored value (`>100.0`), the imputed value is ~163.72, representing the expected true pressure given saturation.
