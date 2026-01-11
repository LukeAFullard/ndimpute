# R Script to Generate Comprehensive Benchmark Suite for ndimpute
# Validates against NADA::ros

if (!requireNamespace("NADA", quietly = TRUE)) {
  stop("Package 'NADA' is needed. Please install it: install.packages('NADA')")
}

library(NADA)

set.seed(123)

output_dir <- "."

generate_dataset <- function(dist, n, cens_level, limit_type) {

  # 1. Generate True Values
  if (dist == "lognormal") {
    # Meanlog=2, Sdlog=1 -> Median=7.4, Mean=12.2
    true_vals <- rlnorm(n, meanlog = 2, sdlog = 1)
  } else if (dist == "normal") {
    # Mean=10, Sd=2
    true_vals <- rnorm(n, mean = 10, sd = 2)
  }

  # 2. Define Limits
  obs_vals <- true_vals
  cens_status <- rep(FALSE, n)

  if (limit_type == "single") {
    # Find quantile for censoring
    limit <- quantile(true_vals, cens_level)
    is_cens <- true_vals < limit
    obs_vals[is_cens] <- limit
    cens_status[is_cens] <- TRUE

  } else if (limit_type == "multiple") {
    # Create 3 limits around the desired censoring level
    # e.g., if target is 50%, limits might be at 30%, 50%, 70% quantiles randomly assigned
    # Or simpler: interleave limits.
    # Let's assign random limits from a set [L1, L2, L3] to all points,
    # then censor if Value < AssignedLimit.
    # We tune limits to achieve roughly 'cens_level' total censoring.

    # Target quantile
    target_q <- quantile(true_vals, cens_level)
    limits <- c(target_q * 0.5, target_q, target_q * 1.5)

    # Assign a random limit to each observation
    assigned_limits <- sample(limits, n, replace = TRUE)

    is_cens <- true_vals < assigned_limits
    obs_vals[is_cens] <- assigned_limits[is_cens]
    cens_status[is_cens] <- TRUE
  }

  # 3. Run NADA ROS
  # NADA::ros expects (obs, censored=TRUE)
  # Handle Normal vs Lognormal. NADA ros() by default logs data.
  # forwardT="log" is default. forwardT="identity" for Normal?
  # NADA documentation says: "forwardT: transformation to apply... default is 'log'"

  transform <- if (dist == "lognormal") "log" else "identity"

  # Catch errors (e.g. if too few uncensored)
  tryCatch({
    ros_model <- NADA::ros(obs_vals, cens_status, forwardT = transform)
    r_imputed <- ros_model$modeled

    # 4. Export
    filename <- paste0("benchmark_", dist, "_n", n, "_cens", cens_level, "_", limit_type, ".csv")
    filepath <- file.path(output_dir, filename)

    df <- data.frame(
      original_value = obs_vals,
      censoring_status = cens_status,
      r_imputed = r_imputed
    )

    write.csv(df, filepath, row.names = FALSE)
    cat(paste("Generated:", filename, "\n"))

  }, error = function(e) {
    cat(paste("Skipping scenario:", dist, n, cens_level, limit_type, "-", e$message, "\n"))
  })
}

# --- Execution Loop ---

dists <- c("lognormal", "normal")
ns <- c(20, 50, 200)
cens_levels <- c(0.2, 0.5, 0.8)
limit_types <- c("single", "multiple")

for (d in dists) {
  for (n in ns) {
    for (c in cens_levels) {
      for (l in limit_types) {
        generate_dataset(d, n, c, l)
      }
    }
  }
}
