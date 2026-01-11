# Validation: [Test Case Name]

## 1. Test Description
**What is being tested:**
[Brief description of the specific scenario, feature, or edge case being validated. E.g., "Robust ROS Imputation on Lognormal Data with 30% Left Censoring."]

**Category:**
[E.g., Left Censoring (ROS), Right Censoring (Parametric), Mixed Censoring, Substitution Strategy, Edge Case]

## 2. Rationale
**Why this test is important:**
[Explanation of why this validation is necessary. E.g., "To ensure the ROS implementation correctly recovers the mean and variance of the underlying distribution despite heavy censoring, matching NADA/R results."]

## 3. Success Criteria
**Expected Outcome for Pass:**
[Specific, measurable criteria that must be met to consider the test a pass.]
- [ ] **Statistical Accuracy:** [E.g., Imputed mean within 5% of true mean]
- [ ] **Distributional Fit:** [E.g., Imputed data passes Normality test on log-scale (Shapiro-Wilk p > 0.05)]
- [ ] **Benchmark Parity:** [E.g., Imputed values within 1% of R's NADA output]
- [ ] **Order Preservation:** [E.g., Imputed values < Detection Limit]
- [ ] **Execution:** [E.g., Runs without errors or warnings]

## 4. Data Generation
**Data Characteristics:**
- **Distribution:** [E.g., Lognormal(mean=2, sigma=1)]
- **Sample Size (N):** [E.g., 50]
- **Censoring Type:** [E.g., Left, Single Detection Limit]
- **Censoring Level:** [E.g., 30% of data < 2.0]
- **Imputation Method:** [E.g., ROS]

## 5. Validation Code
**Step-by-Step Implementation:**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ndimpute import impute

# 1. Generate Synthetic Data
# [Code to generate the data]
# np.random.seed(42)
# true_values = ...
# status = ...
# values = ...

# 2. Run Imputation
# df_result = impute(values, status, method='ros', censoring_type='left')

# 3. Calculate Metrics
# impute_mean = df_result['imputed_value'].mean()
# true_mean = np.mean(true_values)
# bias = impute_mean - true_mean
# print(f"Bias: {bias}")

# 4. Plot Comparison
# fig, ax = plt.subplots()
# sns.kdeplot(true_values, label='True', ax=ax)
# sns.kdeplot(df_result['imputed_value'], label='Imputed', ax=ax)
# ax.legend()
# fig.savefig("distribution_comparison.png")

# 5. Scatter Plot (if applicable)
# fig2, ax2 = plt.subplots()
# plt.scatter(df_result['original_value'], df_result['imputed_value'])
# fig2.savefig("imputation_scatter.png")
```

## 6. Results Output
**Console/Text Output:**
```text
[Paste the actual text output, dictionary, or log from the code execution here]
```

## 7. Visual Evidence
**Distribution Comparison:**
![Distribution Plot](path/to/distribution_comparison.png)
*[Caption: Comparison of the Kernel Density Estimate (KDE) of the true data distribution vs. the distribution after imputation.]*

**Imputation Scatter/QQ Plot:**
![Scatter Plot](path/to/imputation_scatter.png)
*[Caption: Scatter plot showing observed vs imputed values, or a QQ plot demonstrating fit.]*

## 8. Interpretation & Conclusion
**Analysis:**
[Detailed interpretation of the results. Did the imputation recover the parameters? Was the bias acceptable?]

**Benchmark Comparison (if applicable):**
[How did this compare to the R baseline? Were there discrepancies?]

**Pass/Fail Status:**
- [ ] **PASS**
- [ ] **FAIL**

**Notes:**
[Any additional observations or follow-up actions required.]
