# R Script for Validation Case 01: Textbook TCE
# Based on Helsel's examples (Small N, Single Limit)

# Check for NADA
if (!requireNamespace("NADA", quietly = TRUE)) {
  stop("Package 'NADA' is needed for this function to work. Please install it.")
}

library(NADA)

# 1. Define Data
# Synthetic "Textbook" data mimicking TCE concentrations
# 12 samples, some < 1.0
obs <- c(1.0, 1.0, 1.0, 2.4, 3.5, 5.1, 8.0, 1.0, 1.0, 12.0, 1.0, 4.2)
cens <- c(TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE)
# TRUE means Censored (< 1.0)
# FALSE means Observed

# 2. Run NADA ROS
ros_model <- NADA::ros(obs, cens)

# 3. Extract Modeled Values
# NADA stores the filled-in values in $modeled
r_imputed <- ros_model$modeled

# 4. Export Truth
df <- data.frame(
  original_value = obs,
  censoring_status = cens, # TRUE (Censored) / FALSE (Observed)
  r_imputed = r_imputed
)

write.csv(df, "truth.csv", row.names = FALSE)
cat("Exported validation/01_textbook_tce/truth.csv\n")
