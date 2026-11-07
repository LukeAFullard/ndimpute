# Validation Summary

**Re-run:** 2026-11-06, ndimpute 0.1.1 (after the statistical audit; see `CHANGELOG.md`).
**Reference software:** R with NADA 1.6-1.1, survival, and LWPTrends v2502 (`Impute.lower`).

| Case | Scenario | Status | Key finding (re-run) |
| :--- | :--- | :--- | :--- |
| 01 | Textbook TCE | PASS | Matches genuine `NADA::ros` output exactly (the old `truth.csv` was not NADA output; regenerated). |
| 02 | Monte Carlo, left, n = 50 | PASS | Mean bias: ROS −0.075 (−0.6 %, ≈ 1 MC s.e.); LOD/2 −0.151 (−1.2 %). |
| 03 | Weibull reliability (right) | PASS | Parametric conditional mean within −0.07 % of the analytical value. |
| 04 | Mixed stress (Weibull data) | PASS | Parametric +1.6 %. Mixed ROS +14 % with the default lognormal assumption; +1.5 % with `right_dist='weibull'`. The bias is from misspecification, not the procedure. |
| 05 | Edge cases | PASS | 95 % censoring, N = 3, insufficient detects, 100 % censoring and negative data are all handled or rejected with clear errors. |
| 06 | Comparative right | PASS | Parametric −0.05; substitution −16.8; reverse lognormal ROS on Weibull data +37 (misspecified). |
| 07 | Multiple limits | PASS | Matches `NADA::ros`; the old "`<5` must exceed 2" expectation was an artefact of the old plotting positions. |
| 08 | NADA sweep (36 scenarios) | PASS | 31/31 usable datasets match `NADA::ros` to < 1e-13 (now real R output; script bugs fixed). |
| 09 | Interval (Turnbull ROS) | PASS | 100 % of imputations inside their intervals; mean bias −0.1 %. |
| 10 | Auto-detect API | PASS | All 4 cases pass. |
| 11 | R parity | PASS | NADA (60 datasets, 1e-10), survreg (36 fits, 2e-4), conditional means vs `integrate()`, LWPTrends `Impute.lower` (30 datasets, 1e-10). |
| 12 | Monte Carlo study | — | Bias/RMSE of every method; see `12_monte_carlo/SUMMARY.md`. |

## Limitations (current)
* Imputations condition on fitted parameters; stochastic imputation is not proper multiple imputation.
* Results depend on the assumed distribution (cases 04 and 06); choose it deliberately or use `dist='auto'`.
* Mixed ROS is a sequential approximation, and the Turnbull variants are exploratory; prefer `method='parametric'` for inference on mixed or interval data.
