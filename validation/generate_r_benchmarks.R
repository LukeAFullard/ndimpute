# R Script to Generate Benchmark Data for ndimpute Validation
# Requires packages: NADA, survival, data.table

# Install dependencies if missing (uncomment to run)
# install.packages(c("NADA", "survival", "data.table"))

library(NADA)
library(survival)
library(data.table)

set.seed(42)

# --- Helper: Export Function ---
export_benchmark <- function(name, values, status, r_imputed, r_method) {
  df <- data.frame(
    original_value = values,
    censoring_status = status, # In Python format: True/1 if censored
    r_imputed = r_imputed,
    method = r_method
  )
  filename <- paste0("benchmark_", name, ".csv")
  write.csv(df, filename, row.names = FALSE)
  cat(paste("Exported:", filename, "\n"))
}

# ==============================================================================
# Scenario 1: Left Censoring (ROS) - Synthetic Lognormal
# ==============================================================================
n <- 50
true_vals <- rlnorm(n, meanlog = 2, sdlog = 0.5)
lod <- quantile(true_vals, 0.3) # 30% censoring
status_left <- true_vals < lod
obs_vals <- true_vals
obs_vals[status_left] <- lod # Replace with LOD

# Run NADA::ros
# cens = TRUE if censored
ros_model <- NADA::ros(obs_vals, status_left)
# Modeled values (imputed)
r_imputed_ros <- as.numeric(predict(ros_model, newdata=ros_model$modeled))
# Note: NADA predict might behave differently depending on version.
# ros_model$modeled usually contains the filled-in values.
r_imputed_ros <- ros_model$modeled

export_benchmark("left_ros_lognormal", obs_vals, status_left, r_imputed_ros, "NADA::ros")


# ==============================================================================
# Scenario 2: Right Censoring (Parametric) - Synthetic Weibull
# ==============================================================================
n_right <- 100
true_right <- rweibull(n_right, shape = 2, scale = 50)
cens_time <- 60
status_right <- true_right > cens_time
obs_right <- pmin(true_right, cens_time)

# Run survival::survreg
# Surv object: status=1 is Event (Uncensored), 0 is Censored.
# Our status_right is TRUE (1) if Censored. So we flip.
surv_obj <- Surv(time = obs_right, event = !status_right)
fit <- survreg(surv_obj ~ 1, dist = "weibull")

# Predict Mean (response)
# Note: predict() in survreg gives unconditional mean or linear predictor.
# We need conditional mean E[T | T > C].
# R's predict(type="response") gives unconditional mean E[T].
# We might need to manually integrate in R to get exact parity,
# OR we rely on Python to match the unconditional mean for validation if we change the Python logic.
# BUT ndimpute implements CONDITIONAL mean.
# So to validate, we should calculate conditional mean in R explicitly.

shape <- 1 / fit$scale
scale <- exp(coef(fit))

# Function for E[T | T > C] for Weibull
conditional_mean <- function(c, shape, scale) {
  # survival function S(c)
  s_c <- pweibull(c, shape, scale, lower.tail = FALSE)
  # integral t * f(t) from c to Inf
  # Integrate t * dweibull(t)
  int_val <- integrate(function(x) x * dweibull(x, shape, scale), lower = c, upper = Inf)$value

  return(c + (int_val / s_c) - c) # Logic check: integral is E[T restricted].
  # Actually E[T|T>c] = (Integral_c_inf x f(x) dx) / S(c)
  return(int_val / s_c)
}

r_imputed_param <- obs_right
for(i in 1:length(obs_right)) {
  if(status_right[i]) {
    r_imputed_param[i] <- conditional_mean(obs_right[i], shape, scale)
  }
}

export_benchmark("right_parametric_weibull", obs_right, status_right, r_imputed_param, "survival::survreg_conditional")


# ==============================================================================
# Scenario 3: Mixed Censoring (Heuristic)
# ==============================================================================
# There is no direct R package function for the "Sequential ROS" heuristic we implemented.
# We will generate data but leave the R_imputed column empty or approximate.
# This serves as a dataset for Python consistency testing (Regression Testing).
# We can use "impute left then right" logic manually here.

# Generate Data
n_mixed <- 50
mixed_true <- rlnorm(n_mixed, 2, 0.5)
lod_mixed <- 3
right_mixed <- 15

vals_mixed <- mixed_true
stat_mixed <- rep(0, n_mixed) # 0=Obs

# Left
is_left <- vals_mixed < lod_mixed
vals_mixed[is_left] <- lod_mixed
stat_mixed[is_left] <- -1

# Right
is_right <- vals_mixed > right_mixed
vals_mixed[is_right] <- right_mixed
stat_mixed[is_right] <- 1

# Export for Python to run
# Status codes need to be preserved
export_benchmark("mixed_heuristic_input", vals_mixed, stat_mixed, rep(NA, n_mixed), "Python_Heuristic_Check")
