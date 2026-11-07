# Validation 11: Parity with R (NADA, survival)

**Purpose.** Verify `ndimpute` against genuine R output, not a re-implementation.

## Contents

| File | Produced by | Contents |
| :--- | :--- | :--- |
| `generate_r_parity.R` | — | Generates all reference data (seeded). |
| `data/nada_ros.csv` | `NADA::ros` 1.6-1.1 | 60 datasets: lognormal (`forwardT="log"`) and normal (`forwardT=NULL`); n = 12, 30, 100; single and multiple detection limits. Plotting positions `r_pp` and `r_modeled`, **mapped back from NADA's sorted order to the original row order**. |
| `data/survreg_fits.csv` | `survival::survreg(Surv(lo, hi, type="interval2") ~ 1)` | 36 datasets: Weibull, lognormal and Gaussian × right, left, mixed and interval censoring. Parameters and log-likelihood. |
| `data/survreg_cond_means.csv` | R `integrate()` | $E[X \mid lo < X < hi]$ under each survreg fit, for every censored observation. |
| `data/lwptrends_impute_lower.csv` | LWPTrends v2502 `Impute.lower` | *Optional:* generated only when the path to `LWPTrends_v2502.r` is passed. |
| `data/sessionInfo.txt` | R | Versions used. |

## Reproduce

```bash
Rscript generate_r_parity.R                       # NADA + survival
Rscript generate_r_parity.R path/to/LWPTrends_v2502.r   # + LWPTrends Impute.lower
python -m pytest ../../tests/test_r_parity.py
```

NADA 1.6-1.1 is archived on CRAN: `R CMD INSTALL NADA_1.6-1.1.tar.gz` (its only dependency is `survival`).

## Results (`tests/test_r_parity.py`)

| Check | Tolerance | Result |
| :--- | :--- | :--- |
| Hirsch–Stedinger plotting positions vs `NADA::hc.ppoints` (60 datasets) | abs 1e-12 | PASS |
| ROS imputations vs `NADA::ros$modeled` (`impute_type='quantile'`, `clamp=False`) | rel/abs 1e-10 | PASS |
| MLE parameters vs `survreg` (36 datasets) | rel 2e-4 | PASS |
| Log-likelihood (original scale) vs `survreg` | rel 1e-3 | PASS |
| Conditional means vs R numerical integration | rel 5e-4 | PASS |
| LWPTrends v2502 `Impute.lower` vs `lwp_compat=True` (30 datasets; 10 use its "≤ 10 detects → L/2" fallback) | rel 1e-10 | PASS |

## Note on earlier "R comparison" validations

Validation cases 01 and 08 previously did not compare against R:

* `01_textbook_tce/truth.csv` was not NADA output. It has been regenerated with R, and ndimpute now matches it exactly.
* The case 08 R scripts wrote NADA's sorted `$modeled` next to unsorted data, and the case 08 benchmarks were produced by a Python stand-in (`simulate_r_data.py`), not by R.

This case replaces both.
