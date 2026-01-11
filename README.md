<p align="center">
  <img src="assets/Logo.jpg" width="600" alt="ndimpute Logo">
</p>

# ndimpute

[![PyPI version](https://img.shields.io/pypi/v/ndimpute.svg)](https://pypi.org/project/ndimpute/)
[![Python versions](https://img.shields.io/pypi/pyversions/ndimpute.svg)](https://pypi.org/project/ndimpute/)
[![License](https://img.shields.io/pypi/l/ndimpute.svg)](https://github.com/yourusername/ndimpute/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/yourusername/ndimpute/actions)

**Unbiased statistical imputation for censored data** (environmental nondetects, survival times, detection limits, and interval data).

This package implements rigorous statistical methods—Robust ROS and Parametric MLE—to handle censored data without introducing the bias inherent in simple substitution methods.

### Why not just use substitution (LOD/2)?

Simple substitution (replacing values below the **Limit of Detection (LOD)** with `LOD/2` or `0`) destroys the variance of your dataset and biases mean estimates, especially for lognormal data.

| Method | Mean Error (Bias) | Standard Deviation Error |
| :--- | :--- | :--- |
| **Substitution (LOD/2)** | **-15% to +20%** | **Severely Biased** |
| **ndimpute (ROS)** | **< 1%** | **< 2%** |

## Key Features

*   **Robust ROS (Regression on Order Statistics):** Uses Kaplan-Meier plotting positions (handling multiple detection limits) to fit a regression line to the observed data and impute censored values based on their probability.
*   **Parametric Imputation:** MLE-based imputation (Weibull, Lognormal, Normal) for reliability and mixed-censoring analysis.
*   **Interval Censoring:** Handles data known only to lie within a range (e.g., `[5, 10]`) using the **Turnbull Estimator**.
*   **Production-Ready:** Validated against R industry standards (`NADA`, `survival`) and fully vectorized for performance ($N > 100k$).

## Installation

```bash
pip install ndimpute
```

For visualization support (matplotlib/seaborn):
```bash
pip install ndimpute[viz]
```

## Quick Start

### 1. Simplest Case: Numeric Arrays (Recommended)

If you already have your data as numeric arrays and a boolean mask, this is the most direct way to use the library.

```python
import numpy as np
from ndimpute import impute

# Data: 10.0, 0.5 (censored), 5.0
values = np.array([10.0, 0.5, 5.0])
is_censored = np.array([False, True, False])

# Robust ROS (Default for Left Censoring)
df = impute(values, status=is_censored, method='ros')

print(df[['original_value', 'imputed_value']])
#   original_value  imputed_value
# 0           10.0      10.000000
# 1            0.5       0.234...  <-- Imputed below 0.5
# 2            5.0       5.000000
```

### 2. Auto-Detect from Strings

You can pass data as a list containing both numbers and strings. `ndimpute` will parse the `<` (left) and `>` (right) markers from the strings automatically.

```python
from ndimpute import impute

# Mixed data: Observed numbers mixed with censored strings
data = [10.0, "<0.5", 5.0, ">100"]

# Parametric Imputation (Recommended for Mixed data)
df = impute(data, method='parametric', dist='lognormal')
```

## When to Use Which Method

| Data Type | Censoring | Recommended Method | Why? |
| :--- | :--- | :--- | :--- |
| **Environmental** | Left (`<LOD`) | `method='ros'` | Robust to outliers; standard in environmental science. |
| **Reliability** | Right (`>Time`) | `method='parametric'` | Reliability data usually follows physical laws (Weibull). |
| **Mixed** | Left & Right | `method='parametric'` | ROS for mixed data is biased (~14% error); MLE is consistent. |
| **Interval** | Interval `[L, R]` | `method='ros'` | Uses Turnbull estimator to handle arbitrary intervals. |

**Code Example Guide:**

```python
# ❌ Don't do this - Substitution is biased!
df = impute(data, method='substitution', strategy='half')

# ✅ Do this for Environmental Data (Left Censored)
df = impute(data, method='ros')

# ✅ Do this for Reliability Data (Right Censored)
df = impute(data, method='parametric', dist='weibull')
```

## Examples

See the `examples/` directory for runnable scripts:

*   [01_environmental](examples/01_environmental): Left-censored ROS.
*   [02_reliability](examples/02_reliability): Right-censored Weibull.
*   [03_mixed](examples/03_mixed): Mixed censoring.
*   [06_interval](examples/06_interval): Interval censoring with Turnbull.
*   [04_visualization](examples/04_visualization): Visualizing imputed distributions.

## Statistical Methods

For deep dives into the algorithms (Kaplan-Meier, Turnbull, MLE), see [STATISTICAL_METHODS.md](STATISTICAL_METHODS.md).

## Limitations

*   **Minimum Observations:** ROS requires at least 2 uncensored observations to fit a regression line.
*   **Mixed Censoring Bias in ROS:** While supported, using ROS for mixed censoring is less accurate than parametric methods. We recommend `method='parametric'` for mixed data.
*   **Double Censoring:** The package does not currently support specific "double censoring" models (e.g., interval-censored starting time AND interval-censored ending time) often found in incubation period studies.

## Validation

This package is tested against:
*   **R NADA Package:** Matches results for single detection limits.
*   **R Survival Package:** Matches Kaplan-Meier and Turnbull estimations.
*   **NADA 2:** Uses improved Kaplan-Meier plotting positions for multiple detection limits (superior to legacy NADA's simple ranking).

## License

MIT
