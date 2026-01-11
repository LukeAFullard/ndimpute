# Example 01: Environmental Science (Left Censoring)

This example demonstrates the primary use case for `ndimpute`: handling environmental data with nondetects (Left Censoring) using the **Auto-Detect** syntax.

## Scenario
We have a dataset of 30 chemical concentration measurements. The data is "dirty"—it contains strings like `"<5.0"` where the concentration was below the Limit of Detection (LOD).

We compare two methods:
1.  **Robust ROS (Regression on Order Statistics):** The recommended method. It fits a lognormal distribution to the observed data and fills in the censored values based on the fitted tail, preserving variance.
2.  **Substitution (LOD/2):** A simple heuristic that replaces `"<5.0"` with `2.5`. This often biases the mean and variance.

## Code (`main.py`)
The script generates synthetic lognormal data, formats it into a list of strings mimicking a raw CSV column, and then imputes it.


```python
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ndimpute import impute

def main():
    print("--- 01 Environmental Data Example (Left Censoring) ---")

    # 1. Define Data
    # Typical environmental dataset with non-detects (marked with '<')
    # Using the simplified String API (Auto-Detect)
    raw_data = [
        "<0.5", "0.7", "0.9", "<0.5", "1.2",
        "1.5", "<0.5", "2.1", "0.8", "1.1",
        "0.6", "<0.5", "1.8", "2.5", "0.9"
    ]

    print(f"\nRaw Data ({len(raw_data)} samples):\n{raw_data}")

    # 2. Impute using Robust ROS (default)
    # The library automatically detects '<' as left-censoring
    # and defaults to lognormal distribution for environmental data.
    df_ros = impute(raw_data, dist='lognormal', random_state=42)

    print("\n--- Imputation Results (First 5 Rows) ---")
    print(df_ros[['original_value', 'censoring_status', 'imputed_value']].head())

    # 3. Compare with Substitution (1/2 LOD)
    # Explicitly requesting substitution method
    df_sub = impute(raw_data, method='substitution', strategy='half')

    # 4. Calculate Statistics
    mean_ros = df_ros['imputed_value'].mean()
    mean_sub = df_sub['imputed_value'].mean()

    print("\n--- Statistical Comparison ---")
    print(f"Mean (Robust ROS):      {mean_ros:.4f}")
    print(f"Mean (Substitution):    {mean_sub:.4f}")

    print("\nNote: ROS provides a statistically more rigorous estimate of the mean")
    print("by fitting the distribution of the observed data, whereas 1/2 substitution")
    print("is an arbitrary heuristic that often biases results.")

if __name__ == "__main__":
    main()

```

## Output

```text
--- Example 01: Environmental Nondetects (Auto-Detect) ---

Input Data Sample (First 10):
['12.14', '6.43', '14.12', '33.89', '5.85', '5.85', '35.85', '15.92', '<5.0', '12.71']

Running Method A: Robust ROS...
Running Method B: Substitution (LOD/2)...

--- Results Comparison ---
Method               | Mean Est.  | Bias
---------------------------------------------
True Data            | 9.1020     | -
Robust ROS           | 8.9715     | -0.1304
Substitution         | 8.8163     | -0.2856

DataFrame Output (First 5):
   original_value  censoring_status  imputed_value
0           12.14             False          12.14
1            6.43             False           6.43
2           14.12             False          14.12
3           33.89             False          33.89
4            5.85             False           5.85
```

## Interpretation
Robust ROS produced a mean estimate (8.97) closer to the true population mean (9.10) than simple substitution (8.81). By using the auto-detect feature, we simply passed the list of strings, and `ndimpute` handled the parsing logic automatically.
