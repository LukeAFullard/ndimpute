# Validation: Synthetic Lognormal Monte Carlo (02)

## 1. Test Description
**What is being tested:**
Statistical robustness of `method='ros'` vs `method='substitution'` (LOD/2).

**Category:**
Statistical Performance, Tier 2 (Monte Carlo).

## 2. Rationale
**Why this test is important:**
Simple substitution (like LOD/2) is known to be biased for lognormal data because it alters the variance structure. ROS is designed to recover the parameters of the underlying distribution. This test empirically proves (via simulation) that `ndimpute`'s ROS implementation provides a more accurate estimate of the population mean than standard substitution methods.

## 3. Success Criteria
**Expected Outcome for Pass:**
- [x] **Lower Error:** ROS must have a lower Root Mean Square Error (RMSE) than Substitution.
- [x] **Lower Bias:** ROS must have a mean bias closer to 0.

## 4. Data Generation
**Data Characteristics:**
- **Distribution:** Lognormal ($\mu=2, \sigma=1$).
- **Sample Size (N):** 50
- **Censoring:** 30% (Left).
- **Simulations:** 1000 iterations.

## 5. Validation Code
See `validate_mc.py`.

## 6. Results Output
**Console/Text Output:**
```text
Running Monte Carlo Simulation (1000 iterations)...
Distribution: Lognormal(mean=2, sigma=1), N=50, Censoring=30.0%

--- Validation Results ---
Substitution (LOD/2) RMSE: 2.2498
Robust ROS RMSE:           2.2185
Improvement Factor:        1.01x

Substitution Mean Bias:    -0.1508
Robust ROS Mean Bias:      -0.0016
Saved bias_distribution.png

[PASS] ROS outperforms Substitution.
```

## 7. Visual Evidence
**Bias Distribution:**
![Bias Distribution](bias_distribution.png)
*[Caption: Histogram of bias for 1000 simulations. The ROS distribution (Blue) is centered almost perfectly at 0, whereas Substitution (Red) shows a clear negative shift.]*

## 8. Interpretation & Conclusion
**Analysis:**
The Monte Carlo simulation confirms the statistical superiority of the ROS method implemented in `ndimpute`.
- **Bias:** Substitution (LOD/2) consistently underestimates the true mean (Bias $\approx -0.15$), whereas ROS is unbiased (Bias $\approx 0.00$).
- **RMSE:** ROS provides a slight improvement in overall error reduction.

This validation proves that for lognormal data with moderate censoring (30%), `ndimpute`'s ROS implementation correctly recovers the population mean, fulfilling the statistical requirement for robust environmental data analysis.

**Pass/Fail Status:**
- [x] **PASS**
- [ ] **FAIL**
