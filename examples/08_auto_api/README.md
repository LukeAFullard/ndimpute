# Example 08: Auto-Detect API

This example demonstrates the power of the `ndimpute` Auto-Detect API.

## Features Shown

1.  **String Parsing**: Passing a list of strings directly (e.g., `["<1", "5"]`).
2.  **Auto Distribution**: Using `dist='auto'` to let the library choose between Normal and Lognormal based on the best fit to the observed data.
3.  **Custom Markers**: Handling non-standard indicators like "ND" (Non-Detect) or "Sat" (Saturation).

## Code (`main.py`)

```python
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ndimpute import impute

def main():
    print("--- 08 Auto-Detect API Example ---")

    # 1. Mixed Data with Auto-Distribution
    # Some data is left censored (<1), some right censored (>10)
    # The underlying data is roughly normal.
    data = ["<1", "2.5", "3.1", "4.0", "5.2", "6.8", "7.5", ">10", "3.3", "4.8"]

    print(f"\nInput Data: {data}")

    # We simply ask to impute, setting dist='auto'
    df = impute(data, dist='auto', random_state=42)

    print(f"\nAuto-Selected Distribution: {df.attrs.get('best_dist')}")
    print(f"Fit Score (R^2): {df.attrs.get('fit_score'):.4f}")

    print("\n--- Result DataFrame ---")
    print(df[['original_value', 'censoring_status', 'imputed_value']])

    # 2. Custom Markers
    # Sometimes data uses different symbols, like "ND" for Non-Detect or "Sat" for Saturated.
    custom_data = ["ND5", "12", "15", "Sat20", "8"]

    print(f"\n\nCustom Marker Data: {custom_data}")

    df_custom = impute(custom_data,
                       left_marker="ND",
                       right_marker="Sat",
                       dist='normal',
                       random_state=42)

    print("\nParsed Status:")
    # -1: Left (ND), 1: Right (Sat), 0: Observed
    print(df_custom[['original_value', 'censoring_status', 'imputed_value']])

if __name__ == "__main__":
    main()

```

## Output

The script will print the inferred distribution and the imputed values.

```bash
python main.py
```
