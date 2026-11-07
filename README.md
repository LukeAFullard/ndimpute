<p align="center">
  <img src="https://raw.githubusercontent.com/LukeAFullard/ndimpute/main/assets/Logo.jpg" width="600" alt="ndimpute Logo">
</p>

# ndimpute

[![PyPI version](https://img.shields.io/pypi/v/ndimpute.svg)](https://pypi.org/project/ndimpute/)
[![License](https://img.shields.io/pypi/l/ndimpute.svg)](https://github.com/LukeAFullard/ndimpute/blob/main/LICENSE)

**Unbiased statistical imputation for censored data** (environmental nondetects, survival times, detection limits, and interval data).

This package implements rigorous statistical methods—Robust ROS and Parametric MLE—to handle censored data without introducing the bias inherent in simple substitution methods.

### Why not just use substitution (LOD/2)?

Simple substitution (replacing values below the **Limit of Detection (LOD)** with `LOD/2` or `0`) destroys the variance of your dataset and biases mean estimates, especially for lognormal data.

Monte Carlo, lognormal data, n = 200, ~60 % censored, three detection limits ([`validation/12_monte_carlo`](https://github.com/LukeAFullard/ndimpute/blob/main/validation/12_monte_carlo)):

| Method | Bias in mean of logs | Bias in SD of logs |
| :--- | :--- | :--- |
| **Substitution (LOD/2)** | +0.197 | −0.209 |
| **ndimpute ROS (default)** | +0.003 | −0.016 |
| **ndimpute parametric (stochastic)** | −0.002 | +0.000 |

(True values: 0 and 1.)

## Key Features

*   **Robust ROS (Regression on Order Statistics):** Hirsch–Stedinger plotting positions for any number of detection limits. Reproduces `NADA::ros` exactly (validated in R).
*   **Parametric MLE imputation:** Lognormal, Normal and Weibull fits to the full censored likelihood for left, right, mixed and interval censoring. Validated against `survival::survreg`. Conditional means and truncated draws are numerically stable in the far tails.
*   **LWPTrends-compatible mixed censoring:** the `Impute.lower` + `Impute.upper` procedure, with an exact compatibility mode (`lwp_compat=True`).
*   **Interval censoring:** parametric MLE, or Turnbull-NPMLE-based ROS (exploratory).
*   **Safe input handling:** parses `"<0.5"` / `">10"` strings and LWPTrends `CenType` codes (`'lt'`, `'gt'`, `'not'`); passes missing values through; raises on ambiguous or conflicting censoring information instead of guessing.

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

# Ten samples; three are non-detects reported at the LOD of 0.5
values = np.array([0.5, 1.2, 0.5, 2.3, 3.1, 0.5, 4.8, 7.5, 1.9, 12.0])
is_censored = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=bool)

# Robust ROS (default). A boolean / 0-1 status means LEFT censoring ("<LOD").
df = impute(values, status=is_censored, method='ros')

print(df[['original_value', 'imputed_value']].round(3))
#    original_value  imputed_value
# 0             0.5          0.254   <-- imputed from the fitted lognormal line
# 1             1.2          1.200
# 2             0.5          0.446
# ...
# 5             0.5          0.500   <-- line predicts > LOD; capped (clamp=True)
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
| **Environmental** | Left (`<LOD`) | `method='ros'` | Standard in environmental science (Helsel 2012); robust to distributional misfit of the detects. |
| **Reliability** | Right (`>Time`) | `method='parametric'` | Survival data usually follow a parametric law (e.g. Weibull). |
| **Mixed** | Left & Right | `method='parametric'` | One coherent likelihood for both tails. Use `method='ros', lwp_compat=True` to reproduce LWPTrends. |
| **Interval** | Interval `[L, R]` | `method='parametric'` | Exact interval-censored likelihood. |

**Key options**

| Option | Values |
| :--- | :--- |
| `dist` | `'lognormal'` (default), `'normal'`, `'weibull'` (parametric), `'auto'` |
| `impute_type` | ROS: `'quantile'` (default, NADA), `'stochastic'`, `'mean'`. Parametric: `'mean'` (default), `'stochastic'` |
| `random_state` | Seed for stochastic imputation |
| `clamp` | ROS: cap imputations at their detection limit (default `True`; `False` reproduces NADA exactly) |
| `mixed_strategy` | Mixed ROS: `'sequential'` (default, LWPTrends) or `'turnbull'` (experimental) |
| `lwp_compat` | Mixed ROS: reproduce LWPTrends v2502 exactly, including its fall-back rules |

`df.attrs` records the method, censoring type, distribution and fit details for provenance.

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

See the [examples/](https://github.com/LukeAFullard/ndimpute/tree/main/examples) directory for runnable scripts:

## Statistical Methods

For deep dives into the algorithms (Kaplan-Meier, Turnbull, MLE), see [STATISTICAL_METHODS.md](https://github.com/LukeAFullard/ndimpute/blob/main/STATISTICAL_METHODS.md).

## Limitations

*   **Minimum data:** ROS needs at least 2 distinct detects (a warning is issued below 5, and when more than 80 % of values are censored). Parametric fits need at least 2 exact or finite-interval values.
*   **Uncertainty:** imputations condition on fitted parameters. Repeating stochastic imputation with different seeds is *not* proper multiple imputation, and standard errors computed from imputed data are too small. See [STATISTICAL_METHODS.md §7](https://github.com/LukeAFullard/ndimpute/blob/main/STATISTICAL_METHODS.md).
*   **Mixed ROS** is a sequential approximation (it is the LWPTrends procedure); for inference prefer `method='parametric'`. The Turnbull-based ROS variants are exploratory.
*   **Double censoring** (e.g. interval-censored start *and* end times) is not supported.

## Validation

*   **R `NADA::ros` (1.6-1.1):** identical plotting positions and imputations (difference < 1e-10) on 60 datasets, including multiple detection limits and normal-scale ROS ([`validation/11_r_parity`](https://github.com/LukeAFullard/ndimpute/blob/main/validation/11_r_parity)).
*   **R `survival::survreg`:** identical MLE parameters and log-likelihoods for Weibull, lognormal and normal under right, left, mixed and interval censoring; conditional means agree with numerical integration in R.
*   **Monte Carlo:** bias and RMSE of every method over 1,000 replicates per scenario ([`validation/12_monte_carlo`](https://github.com/LukeAFullard/ndimpute/blob/main/validation/12_monte_carlo)).
*   The R reference data are committed; `pytest` checks parity without needing R.

## License

MIT
