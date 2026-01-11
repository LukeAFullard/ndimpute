# Statistical Methods

This document details the statistical algorithms and methodologies implemented in `ndimpute`.

## 1. Left Censoring (Environmental Data)

**Typical Use Case:** Chemical concentrations below a Limit of Detection (LOD).
**Data Structure:** Observations are either fully observed values $x_i$ or censored values known only to be in the interval $(-\infty, L_i]$.

### 1.1 Robust Regression on Order Statistics (ROS)

Robust ROS is a semi-parametric method. It assumes the data follows a distribution (typically Lognormal) only for the purpose of imputing the censored values, while keeping the observed values exactly as measured.

**Algorithm:**
1.  **Plotting Positions:** Calculate the non-exceedance probability $p_i$ for each observation.
    *   **Simple Ranking:** For a single detection limit, $p_i = \text{rank}(x_i) / (N+1)$.
    *   **Kaplan-Meier (Hirsch-Stedinger):** For multiple detection limits, `ndimpute` uses the Kaplan-Meier product-limit estimator (via `scipy.stats.ecdf`) to estimate $p_i$. This correctly accounts for the fact that a value censored at $<5$ provides less information than one censored at $<1$.

        **Why Kaplan-Meier?**
        Standard "Simple Ranking" (rank/(N+1)) works only if there is a single detection limit. When multiple limits exist (e.g. `<1` and `<5`), simple ranking fails to account for the overlapping information. Kaplan-Meier (or Hirsch-Stedinger for left-censoring) adjusts the "effective sample size" at each step.
        For example, if we have observations `[<1, 3, <5, 7]`.
        - Simple ranking might assign ranks 1, 2, 3, 4.
        - KM recognizes that the `<5` value implies the value *could* be anywhere below 5, potentially overlapping with `<1` or `3`.
        The resulting plotting positions are more accurate, leading to better regression estimates.

2.  **Regression:**
    *   Fit a linear regression of Observed Values (or log(Values)) against their theoretical quantiles (Z-scores):
        $$ \ln(x_{obs}) = \text{slope} \cdot \Phi^{-1}(p_{obs}) + \text{intercept} $$
3.  **Imputation:**
    *   For censored observations, predict the value using the regression line and the censored probability:
        $$ \hat{x}_{cens} = \exp(\text{slope} \cdot \Phi^{-1}(p_{cens}) + \text{intercept}) $$
    *   **Stochastic vs. Mean:**
        *   **Stochastic (Default):** Censored values are distributed evenly across the tail $[0, P(X < L)]$ to preserve the variance of the population.
        *   **Conditional Mean:** Imputes the single expected value $E[X | X < L]$. This collapses variance and is generally **not** recommended for standard descriptive statistics.

**Sample Size Considerations (N=30 to 60):**
For smaller sample sizes ($N < 50$), the choice of plotting position logic is critical.
*   **Legacy methods (NADA):** Often used simple ranking. This works for single limits but introduces bias with multiple limits.
*   **ndimpute approach:** Uses Kaplan-Meier logic by default. This is asymptotically unbiased and performs better for multiple limits. For $N=30$, this may yield slightly different results than legacy calculators but is statistically superior.

---

## 2. Right Censoring (Reliability/Survival)

**Typical Use Case:** Time-to-failure data where units survive past the end of the test.
**Data Structure:** Observations are either failure times $t_i$ or censoring times $C_i$ where the unit survived longer ($T > C_i$).

### 2.1 Parametric Conditional Mean Imputation

This method fits a full parametric distribution (Weibull, Lognormal, or Normal) to the data using Maximum Likelihood Estimation (MLE), accounting for the censoring.

**Algorithm:**
1.  **Fit:** Estimate distribution parameters ($\theta$) using `scipy.stats.CensoredData` and MLE.
2.  **Imputation:** Replace each censored value $C_i$ with the expected residual life given it has survived to $C_i$:
    $$ \hat{t}_i = E[T | T > C_i; \theta] = \frac{\int_{C_i}^{\infty} t \cdot f(t; \theta) dt}{1 - F(C_i; \theta)} $$

    *   For **Weibull**, this relates to the upper incomplete gamma function.
    *   For **Lognormal/Normal**, this uses the Inverse Mills Ratio.

**Why Parametric?**
Reliability data often strictly follows physical laws (e.g., Weibull for weakest-link failure), making parametric imputation more powerful than regression-based methods for right-censored data.

---

## 3. Mixed Censoring

**Typical Use Case:** Data with both lower detection limits and upper saturation limits (e.g., sensors that max out).

### 3.1 Parametric (Generalized Likelihood)
Fits a distribution by maximizing the likelihood function across three groups:
$$ L(\theta) = \prod_{obs} f(x_i) \cdot \prod_{left} F(L_i) \cdot \prod_{right} (1 - F(R_i)) $$
Imputation is then performed using the conditional expectation for the respective tail (Left or Right).

---

## 4. Interval Censoring

**Typical Use Case:** An event occurred between two inspection times $(t_1, t_2]$.

### 4.1 Turnbull Estimator + ROS
When exact values are unknown but intervals are bounded:
1.  **Turnbull Estimator:** A non-parametric Maximum Likelihood Estimator (NPMLE) that generalizes the Kaplan-Meier curve for interval-censored data. It iteratively solves for probability masses assigned to overlapping intervals.
2.  **Imputation:** The estimated cumulative distribution function (CDF) from Turnbull is used to assign plotting positions, which are then fed into the ROS regression logic.

---

## 5. Summary of Recommendations

| Sample Size | Censoring Type | Recommended Method | Notes |
| :--- | :--- | :--- | :--- |
| **Small (N < 30)** | Left | **ROS (Kaplan-Meier)** | Robust to outliers; KM handles limits better than simple ranking. |
| **Medium (N=30-100)** | Left | **ROS** | Standard environmental practice. |
| **Any** | Right | **Parametric** | Survival models are usually parametric (Weibull). |
| **Any** | Mixed | **Parametric** | Simplest way to unify disjoint likelihoods. |
