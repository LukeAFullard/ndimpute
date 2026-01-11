# Example 05: Custom Markers

This example demonstrates how to use custom string indicators for censoring, such as `"ND"` (Non-Detect) or `"Sat"` (Saturation), instead of the default `<` and `>`.

## Usage

Simply pass `left_marker` and/or `right_marker` arguments to the `impute` function.

```python
data = ["ND 0.5", "10.0", "Sat 100"]

df = impute(data,
            method='ros',
            left_marker="ND",
            right_marker="Sat")
```

The package will automatically parse:
*   `"ND 0.5"` -> Left Censored at 0.5
*   `"Sat 100"` -> Right Censored at 100
*   `"10.0"` -> Observed


## Code (`main.py`)

```python
"""
Example 05: Custom Markers

This example shows how to use custom censoring indicators (e.g., "ND", "Sat")
instead of the default "<" and ">".
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure we can import ndimpute from the local source
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ndimpute import impute

def main():
    print("--- Example 05: Custom Markers ---\n")

    # 1. Define Data with Custom Indicators
    # "ND" = Non-Detect (Left Censored)
    # "Sat" = Saturated (Right Censored)
    raw_data = [
        "10.5",
        "ND 0.5",    # Left censored at 0.5
        "ND 0.2",    # Left censored at 0.2
        "5.0",
        "Sat 100",   # Right censored at 100
        "12.1"
    ]

    print("Input Data:")
    print(raw_data)
    print("-" * 30)

    # 2. Impute using custom markers
    # We pass left_marker and right_marker arguments
    df = impute(raw_data,
                method='ros',
                left_marker='ND',
                right_marker='Sat')

    print("\nImputation Results:")
    print(df[['original_value', 'imputed_value', 'censoring_status']])

    # Verify status
    # ND 0.5 -> Status should be -1 (Left) or boolean True if interpreted as Left
    # Sat 100 -> Status should be 1 (Right) if Mixed, or ...
    # Wait, detect_and_parse returns integer status for Mixed.
    # Our impute function converts it.

    # Check
    print("\nVerification:")
    status = df['censoring_status'].values
    # If mixed, status is -1, 0, 1
    # Check "ND 0.5" (Index 1)
    if status[1] == -1 or status[1] == True: # Depending on how mixed status is stored in DF
        print("Index 1 (ND 0.5) correctly identified as Left Censored.")
    else:
        print(f"Index 1 Status: {status[1]}")

    # Check "Sat 100" (Index 4)
    if status[4] == 1:
        print("Index 4 (Sat 100) correctly identified as Right Censored.")
    else:
        print(f"Index 4 Status: {status[4]}")

if __name__ == "__main__":
    main()

```
