# Monte Carlo summary (1,000 replicates per scenario)

Population: lognormal(μ = 0, σ = 1). Bias = mean(estimate) − truth. Full table: `mc_results.csv`.

## Left censoring: n = 200, three detection limits, ~60 % censored

| Method | Bias, mean of logs | Bias, SD of logs | Bias, arithmetic mean |
| :--- | ---: | ---: | ---: |
| ROS, Hirsch–Stedinger (default, clamp) | +0.003 | −0.016 | +0.000 |
| ROS, Hirsch–Stedinger (NADA, no clamp) | +0.003 | −0.016 | +0.000 |
| ROS, simple ranking | **+0.541** | **−0.338** | **+0.523** |
| Parametric, conditional mean | +0.119 | **−0.219** | +0.002 |
| Parametric, stochastic | −0.002 | +0.000 | +0.002 |
| Substitution DL/2 | **+0.197** | **−0.209** | +0.091 |

## Mixed censoring ("<0.5", ">4"), n = 100

| Method | Bias, mean of logs | Bias, SD of logs | Bias, arithmetic mean |
| :--- | ---: | ---: | ---: |
| Mixed ROS sequential (default; right step lognormal) | +0.044 | −0.080 | −0.008 |
| Mixed ROS sequential, Weibull right step (LWPTrends) | +0.029 | −0.112 | **−0.132** |
| Mixed ROS Turnbull (experimental) | +0.004 | −0.006 | +0.017 |
| Parametric, conditional mean | +0.034 | −0.051 | +0.017 |
| Parametric, stochastic | +0.001 | +0.002 | +0.026 |

(True arithmetic mean: 1.649.)

## Conclusions

* Hirsch–Stedinger ROS is close to unbiased for single and multiple detection limits. Clamping changes nothing material.
* Simple-ranking ROS should not be used with multiple detection limits; ndimpute warns when it is.
* Conditional-mean imputation shrinks the SD (≈ −0.22 at 60 % censoring). Use stochastic imputation when spread or percentiles matter.
* For mixed data the parametric stochastic method is best overall. LWPTrends' Weibull right step under-estimates the mean of lognormal data by about 8 %. Matching the right-step distribution to `dist` (the ndimpute default) removes most of that bias.
* The Turnbull variant performs well here but has no published theory; it remains labelled experimental.
* At n ≤ 30 every method has RMSE several times its bias; small censored samples carry limited information whatever the method.
