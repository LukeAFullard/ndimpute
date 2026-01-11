import numpy as np
from scipy.stats import norm, linregress
from ._turnbull import turnbull_em, predict_turnbull

def impute_interval_ros(left, right, dist='lognormal', impute_type='stochastic', random_state=None, return_fit=False):
    """
    Imputes interval-censored data using ROS with plotting positions derived
    from the Turnbull Estimator.

    Args:
        left (array): Lower bounds.
        right (array): Upper bounds.
        dist (str): 'lognormal' or 'normal'.
        impute_type (str): 'stochastic' (default) or 'mean'.
            - 'mean': Imputes the conditional mean of the interval (reduces variance).
            - 'stochastic': Randomly samples from the truncated distribution within the interval.
              This preserves variance for large datasets.
        random_state (int, np.random.Generator, optional): Seed or generator for stochastic imputation.
        return_fit (bool): If True, returns (imputed_values, r_squared).
    """
    left = np.array(left)
    right = np.array(right)

    if dist not in ['normal', 'lognormal']:
        raise ValueError(f"Unknown distribution '{dist}'. Supported: 'normal', 'lognormal'.")

    # 1. Turnbull Estimator
    intervals, probs = turnbull_em(left, right)

    if len(probs) == 0:
        raise ValueError("Turnbull estimator failed to find valid intervals.")

    # 2. Calculate Plotting Positions
    # Fit regression to Turnbull CDF points
    if dist == 'lognormal':
        # Geometric mean for lognormal midpoint
        # For interval (L, R), mid = sqrt(L*R).
        # We must ensure L > 0. If L=0, geometric mean is 0.

        mids = np.sqrt(intervals[:, 0] * intervals[:, 1])
    else:
        mids = np.mean(intervals, axis=1)

    # Cumulative Probabilities
    cdf_vals = np.cumsum(probs)
    pp_turnbull = cdf_vals - (probs / 2.0)

    # Handle P=0 or P=1 edge cases if any
    pp_turnbull = np.clip(pp_turnbull, 1e-9, 1 - 1e-9)

    z_turnbull = norm.ppf(pp_turnbull)

    if dist == 'lognormal':
        # Handle non-positive midpoints if any (though Turnbull intervals should be within obs range)
        # If mid <= 0, we can't take log.
        valid_mids = mids > 0
        if not np.any(valid_mids):
             raise ValueError("Intervals must be positive for lognormal distribution.")

        y_fit = np.log(mids[valid_mids])
        z_fit = z_turnbull[valid_mids]
        weights = probs[valid_mids]
    else:
        # Filter infinite midpoints (e.g., from -inf lower bound in mixed censoring)
        valid_mids = np.isfinite(mids)
        if not np.any(valid_mids):
             # Should be rare unless all data is fully censored to infinity
             raise ValueError("No finite intervals available for normal distribution fit.")

        y_fit = mids[valid_mids]
        z_fit = z_turnbull[valid_mids]
        weights = probs[valid_mids]

    # Ensure we have enough points with weight > 0
    mask_w = weights > 1e-9
    y_fit = y_fit[mask_w]
    z_fit = z_fit[mask_w]
    weights = weights[mask_w]

    if len(y_fit) < 2:
        # Not enough points to fit regression line
        raise ValueError("Turnbull yielded insufficient intervals (points) to fit regression.")

    # Weighted OLS
    w_mean_x = np.average(z_fit, weights=weights)
    w_mean_y = np.average(y_fit, weights=weights)

    numerator = np.sum(weights * (z_fit - w_mean_x) * (y_fit - w_mean_y))
    denominator = np.sum(weights * (z_fit - w_mean_x)**2)

    # Calculate R-squared for weighted regression
    # R^2 = (S_zy / sqrt(S_zz * S_yy))^2
    # numerator is S_zy
    # denominator is S_zz
    s_yy = np.sum(weights * (y_fit - w_mean_y)**2)

    if denominator < 1e-12 or s_yy < 1e-12:
        # Variance of Z or Y is zero.
        slope = 0.0
        r_squared = 0.0
    else:
        slope = numerator / denominator
        r_squared = (numerator / np.sqrt(denominator * s_yy))**2

    intercept = w_mean_y - slope * w_mean_x

    # 3. Impute
    mu_model = intercept
    sigma_model = slope

    imputed = np.zeros_like(left, dtype=float)

    # Pre-generate random noise for stochastic
    if impute_type == 'stochastic':
        rng = np.random.default_rng(random_state)
        # Use uniform noise U[0, 1] to sample from truncated CDF
        u_noise = rng.uniform(0, 1, size=len(left))

    for i in range(len(left)):
        l_i, r_i = left[i], right[i]

        # Transform bounds to Z-space
        # If sigma is 0, we can't divide.
        if abs(sigma_model) < 1e-12:
             if dist == 'lognormal':
                imputed[i] = np.exp(mu_model)
             else:
                imputed[i] = mu_model
             continue

        if dist == 'lognormal':
            with np.errstate(divide='ignore'):
                z_l = (np.log(l_i) - mu_model) / sigma_model if l_i > 0 else -np.inf
                z_r = (np.log(r_i) - mu_model) / sigma_model if not np.isinf(r_i) else np.inf
        else:
            z_l = (l_i - mu_model) / sigma_model
            z_r = (r_i - mu_model) / sigma_model if not np.isinf(r_i) else np.inf

        # Ensure z_l < z_r
        if sigma_model < 0:
             z_l, z_r = z_r, z_l

        # Calculate Z value
        Phi_a = norm.cdf(z_l)
        Phi_b = norm.cdf(z_r) if not np.isinf(z_r) else 1.0

        if impute_type == 'mean':
            phi_a = norm.pdf(z_l)
            phi_b = norm.pdf(z_r) if not np.isinf(z_r) else 0.0
            denom = Phi_b - Phi_a

            if denom < 1e-9:
                # Interval is extremely far in tail or tiny (e.g. singleton).
                e_z = (z_l + z_r) / 2 if not np.isinf(z_r) and not np.isinf(z_l) else z_l
            else:
                e_z = (phi_a - phi_b) / denom

            z_final = e_z

        else: # stochastic
            # Sample Z from Truncated Normal

            # Clip Phi values slightly to avoid inf
            Phi_a = np.clip(Phi_a, 1e-15, 1 - 1e-15)
            Phi_b = np.clip(Phi_b, 1e-15, 1 - 1e-15)

            # If interval is tiny (singleton), Phi_a ~ Phi_b.
            if (Phi_b - Phi_a) < 1e-9:
                 z_final = z_l
            else:
                # Map U[0,1] to U[Phi_a, Phi_b]
                u_mapped = Phi_a + u_noise[i] * (Phi_b - Phi_a)
                z_final = norm.ppf(u_mapped)

        # Back transform
        pred_val = mu_model + sigma_model * z_final

        if dist == 'lognormal':
            imputed[i] = np.exp(pred_val)
        else:
            imputed[i] = pred_val

        # Clamp to bounds to ensure numerical precision didn't violate constraints
        # Especially important for stochastic sampling near edges
        if not np.isinf(l_i):
            imputed[i] = max(imputed[i], l_i)
        if not np.isinf(r_i):
            imputed[i] = min(imputed[i], r_i)

    # Explicitly preserve exact observations to avoid floating point drift
    # where left == right
    mask_exact = (left == right)
    imputed[mask_exact] = left[mask_exact]

    if return_fit:
        return imputed, r_squared
    return imputed
