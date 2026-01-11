# Validation: Edge Cases & Stability (05)

## 1. Test Description
**What is being tested:**
Robustness of the package against extreme inputs, including high censoring rates, minimal sample sizes, full censoring, and invalid data types.

**Category:**
Edge Cases, Stability, Input Validation.

## 2. Rationale
**Why this test is important:**
Data quality in the wild is often poor. A robust library must not crash with cryptic errors or return physically impossible results (e.g., NaNs, infinite values, or values violating detection limits) when faced with edge cases. It must raise informative exceptions when statistical assumptions are violated (e.g., fitting a regression line with only 1 point).

## 3. Success Criteria
**Expected Outcome for Pass:**
- [x] **High Censoring (95%):** ROS runs successfully and respects LOD guardrails.
- [x] **Minimal N:** ROS runs successfully with $N_{unc}=2$.
- [x] **Insufficient Data:** Raises `ValueError` for $N_{unc}<2$.
- [x] **100% Censoring:** Raises `ValueError`.
- [x] **Invalid Data:** Raises `ValueError` for negative inputs to Lognormal models.

## 4. Data Generation
**Data Characteristics:**
- **High Censoring:** $N=100$, 95 Censored.
- **Minimal N:** $N=3$, 1 Censored.
- **Insufficient:** $N=10$, 9 Censored.
- **Full Censoring:** $N=10$, 10 Censored.
- **Negative:** Values include -1.0.

## 5. Validation Code
See `validate_edges.py`.

## 6. Results Output
**Console/Text Output:**
```text
Running Validation: 05 Edge Cases & Robustness

Running Sub-test: High Censoring (95%)
Scenario: N=100, 95% Left Censored (< 1.0). 5 Observed values (1.1...1.5).
Max Imputed (Should be <= 1.0): 1.0
Std Dev of Imputed: 0.20675544814859903
[PASS]

Running Sub-test: Minimal N (N=3)
Scenario: N=3, 1 Censored (<1), 2 Observed (2, 3).
Imputed Value: 1.0
[PASS]

Running Sub-test: Insufficient Uncensored
Scenario: N=10, 1 Uncensored. Expecting ValueError.
Caught expected error: Too few uncensored observations to fit regression.
[PASS]

Running Sub-test: 100% Censoring
Scenario: 100% Censoring. Expecting ValueError.
Caught expected error: Too few uncensored observations to fit regression.
[PASS]

Running Sub-test: Invalid Data (Negative)
Scenario: Negative data for LogNormal ROS. Expecting ValueError.
Caught expected error: Values must be positive for lognormal distribution.
[PASS]

[PASS] All Edge Cases Handled Correctly.
```

## 7. Visual Evidence
*N/A - This validation relies on Exception handling and assertion checks rather than plots.*

## 8. Interpretation & Conclusion
**Analysis:**
The `ndimpute` package demonstrates robust error handling and stability. It correctly identifies invalid statistical conditions (insufficient data points, invalid value domains) and raises informative errors. It gracefully handles extreme censoring (95%) without crashing or violating physical constraints (thanks to LOD guardrails).

**Pass/Fail Status:**
- [x] **PASS**
- [ ] **FAIL**
