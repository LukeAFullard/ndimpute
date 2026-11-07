# Changelog

## 0.1.1 — statistical audit fixes

This release fixes defects found in a statistical and implementation audit. **Several defaults change results.** Re-run any analyses made with earlier versions.

### Critical: censoring direction was silently inverted

* **`">"` strings with `censoring_type='mixed'` were imputed as left-censored.** For example, `">4"` was imputed as 2.0. Markers are now always parsed to −1/0/1 before any conversion.
* **A 0/1 integer status with no `censoring_type` was read as right-censoring.** It now means left-censoring (1 = "<L"), as documented. A status containing −1 is mixed. Any other value, or a status that conflicts with the requested `censoring_type`, raises `ValueError`.
* **A boolean status with no `censoring_type` was routed to the Turnbull-based mixed ROS.** It now uses left ROS, as documented.

### High

* **ROS plotting positions are now Hirsch–Stedinger, matching `NADA::ros` exactly.** The previous `'kaplan-meier'` option evaluated $P(X < x)$ with `scipy.stats.ecdf`, which shifted detects down by about one rank and biased the fitted line (in simulation: intercept +0.07 vs ≈0, slope 1.17 vs 1.06 with correct positions), with larger errors for multiple detection limits.
* **Parametric stochastic draws could fall on the wrong side of the censoring limit** in the tails (e.g. ">7" imputed as 6.3), because of clipped inverse-CDF sampling. Draws and conditional means now use stable truncated-distribution routines (`_truncated.py`) and always respect their bounds.
* **Interval and mixed ROS could return `-inf`, `0` or the bound itself** for observations far into the tails. Fixed by the same routines.
* **The validation did not compare against R.** `validation/01_textbook_tce/truth.csv` was not NADA output; the case 08 "R" benchmarks were produced in Python; and the R scripts misaligned NADA's sorted `$modeled`. Added `validation/11_r_parity` with genuine `NADA::ros` and `survival::survreg` output, checked in `tests/test_r_parity.py`.

### Medium

* **ROS `impute_type`:**
  * The new `'quantile'` mode (deterministic NADA positions) is the default for left and right ROS, and is unaffected by `random_state`.
  * `'stochastic'` is now always random. Previously it silently became deterministic when `random_state` was `None`.
  * `'mean'` now returns the conditional mean on the original scale. Previously it back-transformed the log-scale mean, which is biased low.
* **Interval ROS `'mean'` is now the original-scale conditional mean** for lognormal data.
* **Mixed ROS default is now the LWPTrends sequential procedure** (`mixed_strategy='sequential'`):
  * Left step: ROS, as in LWPTrends `Impute.lower`.
  * Right step: MLE fit and a conditional draw, as in `Impute.upper`. The right step uses the same distribution as `dist`; set `right_dist='weibull'` for LWPTrends' choice.
  * `lwp_compat=True` reproduces LWPTrends v2502 including its fall-back rules.
  * The Turnbull approach is still available with `mixed_strategy='turnbull'` and is labelled experimental.
* **`plotting_position='simple'` no longer breaks ties arbitrarily:** censored values now rank below detects of equal value. It warns when there are multiple detection limits, where it is strongly biased.
* **New fit diagnostics:**
  * Errors on a non-positive ROS slope and on detects with no spread.
  * Warnings at > 80 % censoring, with fewer than 5 detects, for detection limits above the largest detect, and when more than half of the imputations need capping.
* **Turnbull NPMLE now uses the proper innermost intervals** (Peto/Turnbull) with a sparse EM algorithm and a tight convergence tolerance. It reproduces Kaplan–Meier and reverse Kaplan–Meier, and its memory use scales with the data rather than N².

### New features

* `method='parametric'` for **left-censored** and **interval-censored** data.
* `dist='auto'` for parametric fits: chooses the distribution with the largest maximised log-likelihood on the original scale.
* `clamp` option for ROS. The default `True` caps imputations at their detection limit; `False` reproduces NADA exactly.
* Missing values (NaN, None, `""`, `"NA"`) are passed through as NaN and excluded from fitting.
* Status may be given as LWPTrends `CenType` strings (`'lt'`, `'gt'`, `'not'`).
* The parser accepts `"5."` and rejects boolean values.
* Unknown keyword arguments raise `TypeError` instead of being silently ignored.
* `df.attrs` records method, censoring type, distribution, impute type and fit details.
* For interval data, `censoring_status` is now `'observed'` for exact rows, and `is_imputed` is `False` for them.

### Removed

* The silent fallback from Turnbull mixed ROS to the old "left ROS, then reverse ROS" heuristic. A failed fit now raises an error.
