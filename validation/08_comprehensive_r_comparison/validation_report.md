# Validation: Comprehensive R Comparison (08)

## 1. Test Description
A parameter sweep comparing `ndimpute` ROS with **`NADA::ros`** (R, NADA 1.6-1.1) across:
- Distributions: lognormal (`forwardT="log"`) and normal (`forwardT=NULL`, `reverseT=NULL`)
- Sample sizes: N = 20, 50, 200
- Censoring levels: 20 %, 50 %, 80 %
- Detection limits: single, or three limits assigned at random

**Category:** Tier 4, parity with the reference implementation.

## 2. History (why this report was rewritten)
Earlier versions of this case did not compare against R:
* The benchmarks were produced by `simulate_r_data.py`, a Python rank-based stand-in, and not by NADA. That script and `validate_legacy_parity.py` have been removed.
* `generate_r_suite.R` wrote NADA's `$modeled` (which is in **sorted** order) next to the unsorted data.
* For the normal distribution it passed `forwardT="identity"` while leaving `reverseT="exp"`, so R exponentiated the normal-scale fit.
* Datasets in which NADA drops censored values above the largest detect were not handled, which also misaligns rows.

All four problems are fixed. Datasets NADA would truncate are now skipped.

## 3. Success Criteria
- [x] With the NADA-equivalent configuration (`plotting_position='kaplan-meier'`, `impute_type='quantile'`, `clamp=False`), every imputed value matches `NADA::ros` to within 1e-8.

## 4. Reproduce
```bash
Rscript generate_r_suite.R      # writes benchmark_*.csv (36 scenarios; 31 usable)
python validate_suite.py        # writes suite_results.csv
python analyze_results.py       # heatmap_mae.png, boxplot_mae.png
```

## 5. Results
31 of the 36 scenarios were generated. The 5 skipped are normal, multiple-limit scenarios in which a limit exceeds the largest detect, so NADA would drop rows.

| Distribution | Limits | Datasets | Max abs. difference vs NADA |
| :--- | :--- | ---: | ---: |
| lognormal | multiple | 9 | 8.7e-14 |
| lognormal | single | 9 | 5.3e-14 |
| normal | multiple | 4 | 5.3e-14 |
| normal | single | 9 | 9.8e-14 |

**Console output (tail):**
```text
[PASS] 31/31 datasets match NADA::ros to 1e-8.
```

## 6. Interpretation & Conclusion
`ndimpute`'s ROS is numerically identical to `NADA::ros`, to floating-point precision, for single and multiple detection limits on both the log and the original scale. The default `clamp=True` differs from NADA only where NADA's line predicts a value above its own detection limit, in which case `ndimpute` caps it at the limit. See also `validation/11_r_parity`, which covers the plotting positions themselves and the parametric (survreg) fits.

**Pass/Fail Status:**
- [x] **PASS**
- [ ] **FAIL**
