# Genuine R reference outputs for ndimpute.
#
# Requires: R, NADA (CRAN archive 1.6-1.1), survival.
# Optional: path to LWPTrends_v2502.r as first command-line argument, to also
#           record the output of LWPTrends' Impute.lower().
#
# Usage (from this directory):
#   Rscript generate_r_parity.R [path/to/LWPTrends_v2502.r]
#
# IMPORTANT: NADA::ros() returns $modeled in SORTED order. It is mapped back
# to the original row order here (as LWPTrends does) before writing.

suppressPackageStartupMessages({ library(NADA); library(survival) })
set.seed(20261106)
args <- commandArgs(trailingOnly = TRUE)
out_dir <- "data"
dir.create(out_dir, showWarnings = FALSE)

# ------------------------------------------------------------------
# 1. NADA::ros
# ------------------------------------------------------------------
ros_rows <- list()
k <- 0
for (dist in c("lognormal", "normal")) {
  for (n in c(12, 30, 100)) {
    for (lim in c("single", "multiple")) {
      for (rep in 1:5) {
        repeat {
          x <- if (dist == "lognormal") rlnorm(n, 1, 1) else rnorm(n, 10, 2)
          if (lim == "single") {
            L <- rep(quantile(x, runif(1, 0.2, 0.5)), n)
          } else {
            q <- quantile(x, c(0.15, 0.35, 0.55))
            L <- sample(q, n, replace = TRUE)
          }
          cens <- x < L
          obs <- ifelse(cens, L, x)
          # Need >= 3 detects and no limit above the largest detect (NADA drops those)
          if (sum(!cens) >= 3 && sum(cens) >= 1 && max(obs[cens]) <= max(obs[!cens])) break
        }
        ft <- if (dist == "lognormal") "log" else NULL
        rt <- if (dist == "lognormal") "exp" else NULL
        r <- suppressWarnings(ros(obs, cens, forwardT = ft, reverseT = rt))
        ix <- order(obs)
        modeled <- numeric(n); modeled[ix] <- r$modeled
        pp <- numeric(n); pp[ix] <- r$pp
        k <- k + 1
        ros_rows[[k]] <- data.frame(dataset = k, dist = dist, n = n, limits = lim,
                                    value = obs, censored = cens, r_pp = pp, r_modeled = modeled)
      }
    }
  }
}
ros_df <- do.call(rbind, ros_rows)
write.csv(ros_df, file.path(out_dir, "nada_ros.csv"), row.names = FALSE)
cat("Wrote", k, "NADA::ros datasets\n")

# ------------------------------------------------------------------
# 2. (optional) LWPTrends Impute.lower -- deterministic left step
# ------------------------------------------------------------------
if (length(args) >= 1 && file.exists(args[1])) {
  exprs <- parse(args[1])
  for (e in exprs) {
    if (is.call(e) && identical(e[[1]], as.name("<-")) && is.name(e[[2]]) &&
        as.character(e[[2]]) %in% c("Impute.lower")) eval(e, envir = globalenv())
  }
  lwp_rows <- list(); j <- 0
  for (id in unique(ros_df$dataset[ros_df$dist == "lognormal"])) {
    d <- ros_df[ros_df$dataset == id, ]
    x <- data.frame(RawValue = d$value, CenType = ifelse(d$censored, "lt", "not"),
                    Censored = d$censored, myDate = seq_len(nrow(d)))
    invisible(capture.output(out <- suppressWarnings(Impute.lower(x))))
    j <- j + 1
    lwp_rows[[j]] <- data.frame(dataset = id, value = d$value, censored = d$censored,
                                lwp_i1 = out$i1Values[order(out$myDate)],
                                lwp_note = out$LeftImpute[1])
  }
  write.csv(do.call(rbind, lwp_rows), file.path(out_dir, "lwptrends_impute_lower.csv"), row.names = FALSE)
  cat("Wrote", j, "LWPTrends Impute.lower datasets\n")
}

# ------------------------------------------------------------------
# 3. survival::survreg intercept-only MLE + conditional means by integrate()
# ------------------------------------------------------------------
cond_mean <- function(dens, lo, hi) {
  num <- integrate(function(t) t * dens(t), lo, hi, rel.tol = 1e-10)$value
  den <- integrate(dens, lo, hi, rel.tol = 1e-10)$value
  num / den
}
fit_rows <- list(); mean_rows <- list(); k <- 0
for (dist in c("weibull", "lognormal", "gaussian")) {
  for (ctype in c("right", "left", "mixed", "interval")) {
    for (rep in 1:3) {
      n <- 80
      x <- switch(dist, weibull = rweibull(n, 2, 10), lognormal = rlnorm(n, 2, 0.6),
                  gaussian = rnorm(n, 10, 3))
      lo <- x; hi <- x
      ql <- quantile(x, 0.25); qr <- quantile(x, 0.8)
      if (ctype %in% c("left", "mixed")) { m <- x < ql; lo[m] <- NA; hi[m] <- ql }
      if (ctype %in% c("right", "mixed")) { m <- x > qr; lo[m] <- qr; hi[m] <- NA }
      if (ctype == "interval") {
        m <- runif(n) < 0.5
        lo[m] <- x[m] - runif(sum(m), 0.5, 3)
        hi[m] <- x[m] + runif(sum(m), 0.5, 3)
        if (dist != "gaussian") lo[m] <- pmax(lo[m], 0.01)
      }
      fit <- survreg(Surv(lo, hi, type = "interval2") ~ 1, dist = dist,
                     control = survreg.control(rel.tolerance = 1e-12, maxiter = 200))
      b <- unname(coef(fit)); s <- fit$scale
      if (dist == "weibull") {
        p1 <- 1 / s; p2 <- exp(b); dens <- function(t) dweibull(t, p1, p2); lower_open <- 0
      } else if (dist == "lognormal") {
        p1 <- b; p2 <- s; dens <- function(t) dlnorm(t, p1, p2); lower_open <- 0
      } else {
        p1 <- b; p2 <- s; dens <- function(t) dnorm(t, p1, p2); lower_open <- -Inf
      }
      k <- k + 1
      fit_rows[[k]] <- data.frame(dataset = k, dist = dist, ctype = ctype, p1 = p1, p2 = p2,
                                  loglik = fit$loglik[1])
      l2 <- ifelse(is.na(lo), lower_open, lo); h2 <- ifelse(is.na(hi), Inf, hi)
      cm <- rep(NA_real_, n)
      for (i in which(l2 != h2)) cm[i] <- cond_mean(dens, l2[i], h2[i])
      mean_rows[[k]] <- data.frame(dataset = k, dist = dist, ctype = ctype,
                                   lo = l2, hi = h2, r_cond_mean = cm)
    }
  }
}
write.csv(do.call(rbind, fit_rows), file.path(out_dir, "survreg_fits.csv"), row.names = FALSE)
write.csv(do.call(rbind, mean_rows), file.path(out_dir, "survreg_cond_means.csv"), row.names = FALSE)
cat("Wrote", k, "survreg datasets\n")
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
