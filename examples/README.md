# User Examples

This directory contains comprehensive examples demonstrating how to use `ndimpute`. Each example is self-contained in its own folder with a script and a README explaining the results.

## Basic Code Examples

### 1. Left Censoring (Environmental)
Most common for concentration data with detection limits (e.g., `<0.5`).

```python
import ndimpute

# Data with detection limits
data = ["<0.5", "1.2", "3.4", "<0.5", "5.1", "2.3"]

# Impute using Robust ROS (default method)
df = ndimpute.impute(data)

print(df[['original_value', 'imputed_value']])
```

### 2. Right Censoring (Reliability)
Common for survival analysis or saturation limits (e.g., `>100`).

```python
import ndimpute

# Data with survival times (censored at 100)
data = ["12", "45", ">100", "88", ">100"]

# Impute using Parametric Weibull (recommended for reliability)
df = ndimpute.impute(data, method='parametric', dist='weibull')

print(df)
```

### 3. Interval Censoring
When values are known only to be within a range.

```python
import numpy as np
import ndimpute

# Intervals: [Left, Right]
# [2, 5] -> value between 2 and 5
# [10, 10] -> exact observation
intervals = np.array([
    [2, 5],
    [10, 10],
    [0, 1]
])

# Impute using Interval ROS
df = ndimpute.impute(intervals, censoring_type='interval')

print(df)
```

## Available Examples

*   **[01_environmental](01_environmental/)**: **Left Censoring.** Handling nondetects in environmental data (e.g., `<0.5`). Compares Robust ROS vs Substitution.
*   **[02_reliability](02_reliability/)**: **Right Censoring.** Estimating component lifetime when some units survive the test (e.g., `>120`).
*   **[03_mixed](03_mixed/)**: **Mixed Censoring.** Handling complex data with both lower detection limits and upper saturation limits.
*   **[04_visualization](04_visualization/)**: **Visualization.** How to plot histograms and CDFs of the imputed data.
*   **[05_custom_markers](05_custom_markers/)**: **Custom Markers.** Using custom indicators like "ND" or "Sat" instead of "<" and ">".
*   **[06_interval](06_interval/)**: **Interval Censoring.** Handling data where values are only known to lie within a specific range (e.g., `[5, 10]`).
*   **[07_auto_dist](07_auto_dist/)**: **Auto Distribution.** Automatically selecting the best distribution (Normal vs Lognormal) for ROS imputation.

## Running Examples

You can run the script inside each folder:

```bash
python examples/01_environmental/main.py
python examples/02_reliability/main.py
# etc.
```

All examples use the **Auto-Detect** syntax, where you simply pass a list of strings (e.g., `["<0.5", "10.0"]`) and `ndimpute` automatically handles the censoring logic.
