# Statistical Methods

This document describes exactly what `ndimpute` computes, which published method each option implements, and how it was verified. Numerical claims refer to the validation cases in [`validation/`](validation/).

## 0. Notation and censoring codes

| Censoring | Meaning | `status` |
| :--- | :--- | :--- |
| Observed | value $x$ known | `0` / `False` |
| Left | $X < L$ ("<L", non-detect) | `-1` / `True` (left) / `'lt'` |
| Right | $X > R$ (">R") | `1` / `True` with `censoring_type='right'` / `'gt'` |
| Interval | $l \le X \le r$ | `(N, 2)` bounds with `censoring_type='interval'` |

A boolean or 0/1 status with no `censoring_type` is **left** censoring (environmental convention). A status containing `-1` is **mixed**. Mismatches between requested type and data raise an error rather than being silently reinterpreted.

---

## 1. Left censoring: Robust Regression on Order Statistics (ROS)

**Reference method:** `NADA::ros` (Lee & Helsel 2005; Helsel 2012, ch. 6), as used by LWPTrends `Impute.lower`.
**Verified:** identical to `NADA::ros` `$modeled` (max abs. difference < 1e-10) on 60 datasets — lognormal and normal, $n = 12, 30, 100$, single and multiple detection limits (`validation/11_r_parity`, `tests/test_r_parity.py`).

### 1.1 Plotting positions (Hirsch–Stedinger, default `plotting_position='kaplan-meier'`)

Hirsch & Stedinger (1987), in the formulation of Helsel & Cohn (1988). With detection limits $L_1 < \dots < L_m$:

* $A_j$ = number of detects in $[L_j, L_{j+1})$,
* $B_j$ = number of observations $\le L_j$, excluding detects equal to $L_j$,
* $C_j$ = number of values censored at $L_j$,
* $P_j = P_{j+1} + \dfrac{A_j}{A_j + B_j}(1 - P_{j+1})$, $P_{m+1} = 0$ — the probability of exceeding $L_j$.

If any detect lies below $L_1$, a pseudo-limit below all data is added (NADA uses 0, which silently drops negative detects on the untransformed scale; `ndimpute` uses $-\infty$, identical for non-negative data).

* Detect of rank $r$ within $[L_j, L_{j+1})$: $p = (1-P_j) + (P_j - P_{j+1})\,\dfrac{r}{A_j+1}$
* Value censored at $L_j$, rank $r$ among the $C_j$: $p = (1-P_j)\,\dfrac{r}{C_j+1}$

This is a Kaplan–Meier-type estimator of the CDF at each limit and is valid for any number of detection limits. (Versions ≤ 0.2 evaluated `scipy.stats.ecdf` at $P(X < x)$ instead of these positions, shifting detects down by about one rank; that has been replaced.)

`plotting_position='simple'` uses rank$/(n+1)$ over all observations (censored values ranked below tied detects). It is only valid for a single detection limit below all detects and is strongly biased otherwise (+0.54 in the log-mean at 60 % censoring with multiple limits, `validation/12_monte_carlo`); a warning is issued in that case.

### 1.2 Regression and imputation

OLS of the transformed detects ($\ln x$ for `dist='lognormal'`, $x$ for `'normal'`) on $z = \Phi^{-1}(p)$ gives intercept $\hat\mu$ and slope $\hat\sigma$. Detected values are always returned unchanged. A non-positive slope, or detects with no spread, raises an error.

`impute_type`:

* `'quantile'` (default) — each censored value is the line's prediction at its own plotting position: $\hat x = T^{-1}(\hat\mu + \hat\sigma z)$. Deterministic; this is NADA.
* `'stochastic'` — $p \sim U(0, 1-P_j)$, then as above. Seed with `random_state`.
* `'mean'` — the conditional mean of the fitted distribution below the limit's position $z_j = \Phi^{-1}(1-P_j)$, **on the original scale**. For lognormal:
  $E[X \mid Z < z_j] = e^{\hat\mu + \hat\sigma^2/2}\,\Phi(z_j - \hat\sigma)/\Phi(z_j)$.
  (Earlier versions back-transformed the log-scale mean, which is biased low by Jensen's inequality.) Assigning one value to every non-detect at a limit shrinks the variance, so this mode is not recommended for SDs or percentiles.

### 1.3 Clamping (deviation from NADA)

NADA does not constrain imputations, so a "<L" can be imputed above $L$ when the fitted line is steep relative to the detection limit. With `clamp=True` (default) `ndimpute` caps each imputation at its own limit; `clamp=False` reproduces NADA exactly. In the Monte Carlo study the two differ by < 0.006 in every summary statistic. If more than half of the non-detects need capping a warning is issued, because the model fits the data poorly.

### 1.4 Diagnostics and limits

Warnings are issued when > 80 % of values are censored (as NADA does), when there are fewer than 5 detects, and when a detection limit exceeds the largest detect (NADA drops those values). ROS needs at least 2 distinct detects.

---

## 2. Right censoring

### 2.1 Reverse ROS (`method='ros'`)

The data are mapped to a left-censored problem by a transformation that preserves the distribution family ($y = 1/x$ for lognormal since $\ln y = -\ln x$; $y = -x$ for normal), imputed with §1, and mapped back. For reliability / survival data, parametric MLE (§4) is usually more appropriate.

### 2.2 Parametric (`method='parametric'`): see §4.

---

## 3. Mixed (left + right) censoring

### 3.1 `method='parametric'` (recommended for inference): see §4.

### 3.2 `method='ros'`, `mixed_strategy='sequential'` (default) — the LWPTrends procedure

1. **Left step** (LWPTrends `Impute.lower`): non-detects are imputed by ROS (§1); right-censored values take part as if detected at their limit.
2. **Right step** (LWPTrends `Impute.upper`): a distribution is fitted by MLE (equivalent to `survival::survreg`) to the step-1 values with ">R" values right-censored, and each ">R" is replaced by a draw from the fitted distribution conditional on $X > R$ (`right_impute_type='stochastic'`) or its conditional mean (`'mean'`).

LWPTrends always uses a Weibull in step 2. `ndimpute` defaults to the same family as the ROS step (`right_dist=None` → `dist`), because a Weibull right tail under-estimates the arithmetic mean of lognormal data by about 8 % (n = 100; `validation/12_monte_carlo`), compared with under 1 % when the family matches.

`lwp_compat=True` reproduces LWPTrends v2502 exactly, including its fall-back rules:

* "<L" with $L \ge$ the largest non-"<" value is reset to half that value before ROS;
* ROS is only used when there are more than 10 uncensored values; otherwise "<L" → $L/2$;
* with $n < 24$, ">R" → $1.1R$;
* otherwise a Weibull draw, with the conditioning point capped at the fitted 99.5th percentile;
* no clamping.

LWPTrends draws from 10,000 simulated Weibull values above the limit; `ndimpute` draws exactly from the truncated distribution.

### 3.3 `mixed_strategy='turnbull'` (experimental)

The data are recast as intervals ([0 or $-\infty$, L], [x, x], [R, ∞)) and imputed with Turnbull ROS (§5.2). This is not an established published estimator and is retained for exploration only.

---

## 4. Parametric maximum-likelihood imputation (`method='parametric'`)

Available for left, right, mixed and interval censoring with `dist` in `'lognormal'`, `'normal'`, `'weibull'`.

1. **Fit.** Maximise the full censored likelihood
   $$L(\theta) = \prod_{\text{obs}} f(x_i)\prod_{\text{left}} F(L_i)\prod_{\text{right}} S(R_i)\prod_{\text{interval}}[F(r_i) - F(l_i)]$$
   with `scipy.stats.CensoredData`. **Verified** against `survival::survreg` (intercept-only) for all three distributions under right, left, mixed and interval censoring (36 datasets): parameters agree to a relative tolerance of 2e-4, and so do the log-likelihoods on the original scale.
2. **Impute.** Each censored observation with bounds $[lo, hi]$ receives
   * `impute_type='mean'` (default): $E[X \mid lo \le X \le hi;\hat\theta]$, or
   * `impute_type='stochastic'`: a draw from the fitted distribution truncated to $[lo, hi]$.

   Closed forms: normal and lognormal through truncated-normal moments (lognormal: $e^{\mu+\sigma^2/2}[\Phi(b-\sigma)-\Phi(a-\sigma)]/[\Phi(b)-\Phi(a)]$); Weibull through regularised incomplete gamma functions. All are evaluated in log-probability space (`_truncated.py`), so values stay finite and inside their bounds however far into the tail the limit lies. Weibull draws use the exact transformation $Y=(X/\lambda)^k \sim \text{Exp}(1)$. **Verified** against numerical integration in R (`integrate()`), relative tolerance 5e-4, and against quadrature in `tests/test_audit_fixes.py`.

`dist='auto'` chooses the distribution with the largest maximised log-likelihood on the original data scale (lognormal includes the Jacobian). All candidates have two parameters, so this is equivalent to choosing by AIC.

Conditional-mean imputation gives every value censored at the same limit the same value, which under-states the spread (SD of the log values biased by about −0.22 at 60 % censoring, `validation/12_monte_carlo`). Use `'stochastic'` when spread, percentiles or distribution shape matter.

---

## 5. Interval censoring

### 5.1 Parametric (`method='parametric'`, recommended): §4 with interval bounds.

### 5.2 Turnbull ROS (`method='ros'`)

1. **Turnbull NPMLE** (Turnbull 1976): mass is placed on the *innermost intervals* (Peto 1973) — a left endpoint followed directly by a right endpoint in the sorted endpoints, with ties ordered left-first for closed intervals — and estimated with the self-consistency EM algorithm (sparse; tolerance 1e-10). Verified to reproduce Kaplan–Meier for right-censored data and reverse Kaplan–Meier for left-censored data.
2. **Probability plot:** a weighted least-squares line of the innermost-interval midpoint (geometric midpoint for lognormal) on $\Phi^{-1}$(mid-cumulative probability), with Turnbull masses as weights. Intervals with infinite or zero midpoints are excluded from the fit.
3. **Impute:** the line's $(\hat\mu, \hat\sigma)$ define a distribution, and each observation receives a truncated draw (`'stochastic'`, default) or conditional mean (`'mean'`) within its own bounds.

This is a reasonable semi-parametric generalisation of ROS, but it has no published sampling theory. Prefer §5.1 for inference.

---

## 6. Substitution

`LOD/2`, `0`, `LOD`, or `LOD × multiplier` (and `C`, `C × multiplier` for right censoring). Provided for comparison only. Substitution is biased (Helsel 2006); see `validation/12_monte_carlo`.

---

## 7. Uncertainty

All imputations condition on point estimates of the model parameters. Stochastic imputations reproduce the spread of the data but not the uncertainty in the parameters, so repeating them with different seeds is **not** proper multiple imputation (Rubin 1987), and standard errors computed from imputed data will be too small. For inference, bootstrap the whole procedure, or use censored-data methods directly (e.g. MLE, Kaplan–Meier, `cenken`).

---

## 8. Recommendations

| Censoring | Goal | Recommended |
| :--- | :--- | :--- |
| Left, any number of limits | Descriptive statistics, trend inputs | `method='ros'` (defaults) |
| Left | Distribution shape / percentiles | `method='ros', impute_type='stochastic'` or `method='parametric', impute_type='stochastic'` |
| Right (reliability) | Any | `method='parametric', dist='weibull'` (or `'auto'`) |
| Mixed | Inference | `method='parametric'` |
| Mixed | Reproduce LWPTrends | `method='ros', lwp_compat=True` |
| Interval | Any | `method='parametric'` |

## References

* Helsel, D. R. (2006). Fabricating data: how substituting values for nondetects can ruin results. *Chemosphere* 65, 2434–2439.
* Helsel, D. R. (2012). *Statistics for Censored Environmental Data Using Minitab and R*, 2nd ed. Wiley.
* Helsel, D. R. & Cohn, T. A. (1988). Estimation of descriptive statistics for multiply censored water quality data. *Water Resources Research* 24(12), 1997–2004.
* Hirsch, R. M. & Stedinger, J. R. (1987). Plotting positions for historical floods and their precision. *Water Resources Research* 23(4), 715–727.
* Lee, L. & Helsel, D. R. (2005). Statistical analysis of water-quality data containing multiple detection limits: S-language software for regression on order statistics. *Computers & Geosciences* 31, 1241–1248. (R package NADA.)
* Peto, R. (1973). Experimental survival curves for interval-censored data. *Applied Statistics* 22, 86–91.
* Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.
* Snelder, T., Fraser, C. & Larned, S. LWPTrends R functions (v2502), LWP Ltd.
* Turnbull, B. W. (1976). The empirical distribution function with arbitrarily grouped, censored and truncated data. *JRSS B* 38, 290–295.
