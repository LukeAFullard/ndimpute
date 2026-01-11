import numpy as np
from scipy.stats import weibull_min, norm, lognorm, CensoredData
from scipy.special import gamma, gammaincc, gammainc

def impute_right_conditional(values, is_censored, dist='lognormal', impute_type='mean', random_state=None):
    """
    Imputes right-censored data using Conditional Mean Imputation (Vectorized) or Stochastic Imputation.

    Args:
        values (array): Data values.
        is_censored (bool array): True if value is censored.
        dist (str): Distribution ('lognormal', 'normal', 'weibull').
        impute_type (str): 'mean' (default) or 'stochastic'.
        random_state (int, optional): Seed for reproducibility.
    """
    data = np.array(values)
    cens = np.array(is_censored, dtype=bool)

    if not np.any(cens):
        return data.copy()

    rng = np.random.default_rng(random_state) if impute_type == 'stochastic' else None

    # Dispatch to specific distribution implementation
    if dist == 'weibull': # Explicit Weibull
        return _impute_right_weibull(data, cens, impute_type, rng)
    elif dist == 'lognormal':
        return _impute_right_lognormal(data, cens, impute_type, rng)
    elif dist == 'normal':
        return _impute_right_normal(data, cens, impute_type, rng)
    else:
        raise ValueError(f"Unknown distribution '{dist}' for parametric imputation.")

def impute_mixed_parametric(values, status, dist='lognormal', impute_type='mean', random_state=None):
    """
    Imputes mixed-censored data using Conditional Mean Imputation or Stochastic Imputation.

    Args:
        values (array): Data values.
        status (array): Censoring status (-1: Left, 0: Obs, 1: Right).
        dist (str): Distribution ('lognormal', 'normal', 'weibull').
        impute_type (str): 'mean' (default) or 'stochastic'.
        random_state (int, optional): Seed for reproducibility.
    """
    data = np.array(values)
    status = np.array(status, dtype=int)

    # Masks
    mask_obs = (status == 0)
    mask_left = (status == -1)
    mask_right = (status == 1)

    rng = np.random.default_rng(random_state) if impute_type == 'stochastic' else None

    if dist == 'weibull':
         return _impute_mixed_weibull(data, mask_obs, mask_left, mask_right, impute_type, rng)
    elif dist == 'lognormal':
         return _impute_mixed_lognormal(data, mask_obs, mask_left, mask_right, impute_type, rng)
    elif dist == 'normal':
         return _impute_mixed_normal(data, mask_obs, mask_left, mask_right, impute_type, rng)
    else:
         raise ValueError(f"Unknown distribution '{dist}' for parametric imputation.")

# --- Internal Implementations ---

def _impute_right_weibull(data, cens, impute_type='mean', rng=None):
    """
    Imputes right-censored data using Weibull distribution.

    Args:
        data (array): All data points.
        cens (bool array): True if right-censored.
        impute_type (str): 'mean' or 'stochastic'.
        rng (np.random.Generator, optional): Random number generator.

    Returns:
        array: Imputed data.
    """
    uncensored_vals = data[~cens]
    censored_vals = data[cens]
    cd = CensoredData(uncensored=uncensored_vals, right=censored_vals)
    shape, loc, scale = weibull_min.fit(cd, floc=0)

    imputed = data.copy()
    C = data[cens]

    if impute_type == 'stochastic':
        # Sample from truncated Weibull > C
        # CDF(x) = 1 - exp(-(x/scale)^shape)
        # We need to sample U ~ Uniform(CDF(C), 1)
        # Then x = CDF_inv(U) = scale * (-ln(1-U))^(1/shape)

        cdf_C = 1.0 - np.exp(-(C / scale) ** shape)
        # Ensure cdf_C is strictly < 1.0
        cdf_C = np.minimum(cdf_C, 1.0 - 1e-9)

        u = rng.uniform(low=cdf_C, high=1.0)
        # Avoid log(0) if u is exactly 1 (which shouldn't happen with uniform[low, 1.0) usually, but safe guard)
        u = np.minimum(u, 1.0 - 1e-15)

        sampled_val = scale * (-np.log(1.0 - u)) ** (1.0 / shape)
        imputed[cens] = sampled_val
    else:
        u_c = (C / scale) ** shape
        mean_unconditional = scale * gamma(1 + 1.0/shape)
        integral_upper = mean_unconditional * gammaincc(1.0 + 1.0/shape, u_c)
        S_C = np.exp(-u_c)
        valid_mask = S_C > 1e-15
        expected_val = C.copy()
        expected_val[valid_mask] = integral_upper[valid_mask] / S_C[valid_mask]
        expected_val[~valid_mask] = C[~valid_mask] # Fallback
        imputed[cens] = expected_val

    return imputed

def _impute_mixed_weibull(data, mask_obs, mask_left, mask_right, impute_type='mean', rng=None):
    """
    Imputes mixed-censored data using Weibull distribution.

    Args:
        data (array): All data points.
        mask_obs (bool array): True if observed.
        mask_left (bool array): True if left-censored.
        mask_right (bool array): True if right-censored.
        impute_type (str): 'mean' or 'stochastic'.
        rng (np.random.Generator, optional): Random number generator.

    Returns:
        array: Imputed data.
    """
    cd = CensoredData(
        uncensored=data[mask_obs],
        left=data[mask_left],
        right=data[mask_right]
    )
    shape, loc, scale = weibull_min.fit(cd, floc=0)

    imputed = data.copy()
    mean_unconditional = scale * gamma(1 + 1.0/shape)

    if np.any(mask_left):
        L = data[mask_left]
        if impute_type == 'stochastic':
            # Sample from truncated Weibull < L
            # U ~ Uniform(0, CDF(L))
            cdf_L = 1.0 - np.exp(-(L / scale) ** shape)
            cdf_L = np.maximum(cdf_L, 1e-9) # Avoid 0 range

            u = rng.uniform(low=0.0, high=cdf_L)
            sampled_val = scale * (-np.log(1.0 - u)) ** (1.0 / shape)
            imputed[mask_left] = sampled_val
        else:
            u_L = (L / scale) ** shape
            F_L = 1.0 - np.exp(-u_L)
            integral_lower = mean_unconditional * gammainc(1.0 + 1.0/shape, u_L)
            valid_mask = F_L > 1e-15
            vals = L.copy()
            vals[valid_mask] = integral_lower[valid_mask] / F_L[valid_mask]
            vals[~valid_mask] = L[~valid_mask] / 2.0
            imputed[mask_left] = vals

    if np.any(mask_right):
        R = data[mask_right]
        if impute_type == 'stochastic':
            # Sample from truncated Weibull > R
            cdf_R = 1.0 - np.exp(-(R / scale) ** shape)
            cdf_R = np.minimum(cdf_R, 1.0 - 1e-9)

            u = rng.uniform(low=cdf_R, high=1.0)
            u = np.minimum(u, 1.0 - 1e-15)

            sampled_val = scale * (-np.log(1.0 - u)) ** (1.0 / shape)
            imputed[mask_right] = sampled_val
        else:
            u_R = (R / scale) ** shape
            S_R = np.exp(-u_R)
            integral_upper = mean_unconditional * gammaincc(1.0 + 1.0/shape, u_R)
            valid_mask = S_R > 1e-15
            vals = R.copy()
            vals[valid_mask] = integral_upper[valid_mask] / S_R[valid_mask]
            vals[~valid_mask] = R[~valid_mask] # Fallback
            imputed[mask_right] = vals

    return imputed

def _impute_mixed_normal(data, mask_obs, mask_left, mask_right, impute_type='mean', rng=None):
    """
    Imputes mixed-censored data using Normal distribution.

    Args:
        data (array): All data points.
        mask_obs (bool array): True if observed.
        mask_left (bool array): True if left-censored.
        mask_right (bool array): True if right-censored.
        impute_type (str): 'mean' or 'stochastic'.
        rng (np.random.Generator, optional): Random number generator.

    Returns:
        array: Imputed data.
    """
    # Fit Normal
    cd = CensoredData(
        uncensored=data[mask_obs],
        left=data[mask_left],
        right=data[mask_right]
    )
    mu, std = norm.fit(cd)

    imputed = data.copy()

    # E[X | X < L]
    if np.any(mask_left):
        L = data[mask_left]
        if impute_type == 'stochastic':
            # Truncated Normal < L
            # U ~ Uniform(0, CDF(L))
            z_L = (L - mu) / std
            cdf_L = norm.cdf(z_L)
            cdf_L = np.maximum(cdf_L, 1e-9)

            u = rng.uniform(low=0.0, high=cdf_L)
            z_sampled = norm.ppf(u)
            imputed[mask_left] = mu + std * z_sampled
        else:
            z = (L - mu) / std
            pdf_z = norm.pdf(z)
            cdf_z = norm.cdf(z)
            valid = cdf_z > 1e-15
            impute_vals = L.copy()
            impute_vals[valid] = mu - std * (pdf_z[valid] / cdf_z[valid])
            impute_vals[~valid] = L[~valid]
            imputed[mask_left] = impute_vals

    # E[X | X > R]
    if np.any(mask_right):
        R = data[mask_right]
        if impute_type == 'stochastic':
            # Truncated Normal > R
            # U ~ Uniform(CDF(R), 1)
            z_R = (R - mu) / std
            cdf_R = norm.cdf(z_R)
            cdf_R = np.minimum(cdf_R, 1.0 - 1e-9)

            u = rng.uniform(low=cdf_R, high=1.0)
            u = np.minimum(u, 1.0 - 1e-15)
            z_sampled = norm.ppf(u)
            imputed[mask_right] = mu + std * z_sampled
        else:
            z = (R - mu) / std
            pdf_z = norm.pdf(z)
            sf_z = 1.0 - norm.cdf(z)
            valid = sf_z > 1e-15
            impute_vals = R.copy()
            impute_vals[valid] = mu + std * (pdf_z[valid] / sf_z[valid])
            impute_vals[~valid] = R[~valid]
            imputed[mask_right] = impute_vals

    return imputed

def _impute_right_normal(data, cens, impute_type='mean', rng=None):
    """
    Imputes right-censored data using Normal distribution.

    Args:
        data (array): All data points.
        cens (bool array): True if right-censored.
        impute_type (str): 'mean' or 'stochastic'.
        rng (np.random.Generator, optional): Random number generator.

    Returns:
        array: Imputed data.
    """
    # Specialized for Right Only (can reuse mixed logic)
    mask_obs = ~cens
    mask_left = np.zeros(len(data), dtype=bool)
    mask_right = cens
    return _impute_mixed_normal(data, mask_obs, mask_left, mask_right, impute_type, rng)

def _impute_mixed_lognormal(data, mask_obs, mask_left, mask_right, impute_type='mean', rng=None):
    """
    Imputes mixed-censored data using LogNormal distribution.

    Args:
        data (array): All data points.
        mask_obs (bool array): True if observed.
        mask_left (bool array): True if left-censored.
        mask_right (bool array): True if right-censored.
        impute_type (str): 'mean' or 'stochastic'.
        rng (np.random.Generator, optional): Random number generator.

    Returns:
        array: Imputed data.
    """
    if (data <= 0).any():
        raise ValueError("Values must be positive for lognormal distribution.")

    log_data = np.log(data)

    # We can reuse the normal implementation by operating on log data,
    # then exponentiating the result for stochastic mode.
    # For mean mode, we need the specific lognormal conditional expectation formula.

    if impute_type == 'stochastic':
        # Fit Normal on log data
        cd = CensoredData(
            uncensored=log_data[mask_obs],
            left=log_data[mask_left],
            right=log_data[mask_right]
        )
        mu, std = norm.fit(cd)

        imputed = data.copy()

        if np.any(mask_left):
            L = data[mask_left]
            ln_L = np.log(L)
            z_L = (ln_L - mu) / std
            cdf_L = norm.cdf(z_L)
            cdf_L = np.maximum(cdf_L, 1e-9)

            u = rng.uniform(low=0.0, high=cdf_L)
            z_sampled = norm.ppf(u)
            imputed[mask_left] = np.exp(mu + std * z_sampled)

        if np.any(mask_right):
            R = data[mask_right]
            ln_R = np.log(R)
            z_R = (ln_R - mu) / std
            cdf_R = norm.cdf(z_R)
            cdf_R = np.minimum(cdf_R, 1.0 - 1e-9)

            u = rng.uniform(low=cdf_R, high=1.0)
            u = np.minimum(u, 1.0 - 1e-15)
            z_sampled = norm.ppf(u)
            imputed[mask_right] = np.exp(mu + std * z_sampled)

        return imputed

    else:
        # Mean mode requires the specific integral formula implemented previously
        # Fit on log data
        cd = CensoredData(
            uncensored=log_data[mask_obs],
            left=log_data[mask_left],
            right=log_data[mask_right]
        )
        mu, std = norm.fit(cd)

        imputed = data.copy()
        mean_unconditional = np.exp(mu + 0.5 * std**2)

        if np.any(mask_left):
            L = data[mask_left]
            ln_L = np.log(L)
            alpha = (ln_L - mu) / std

            Phi_alpha = norm.cdf(alpha)
            Phi_shifted = norm.cdf(alpha - std)

            valid = Phi_alpha > 1e-15
            vals = L.copy()
            vals[valid] = mean_unconditional * (Phi_shifted[valid] / Phi_alpha[valid])
            vals[~valid] = L[~valid] # Fallback

            imputed[mask_left] = vals

        if np.any(mask_right):
            R = data[mask_right]
            ln_R = np.log(R)
            alpha = (ln_R - mu) / std

            Sf_alpha = 1.0 - norm.cdf(alpha)
            Sf_shifted = 1.0 - norm.cdf(alpha - std)

            valid = Sf_alpha > 1e-15
            vals = R.copy()
            vals[valid] = mean_unconditional * (Sf_shifted[valid] / Sf_alpha[valid])
            vals[~valid] = R[~valid] # Fallback

            imputed[mask_right] = vals

        return imputed

def _impute_right_lognormal(data, cens, impute_type='mean', rng=None):
    """
    Imputes right-censored data using LogNormal distribution.

    Args:
        data (array): All data points.
        cens (bool array): True if right-censored.
        impute_type (str): 'mean' or 'stochastic'.
        rng (np.random.Generator, optional): Random number generator.

    Returns:
        array: Imputed data.
    """
    mask_obs = ~cens
    mask_left = np.zeros(len(data), dtype=bool)
    mask_right = cens
    return _impute_mixed_lognormal(data, mask_obs, mask_left, mask_right, impute_type, rng)
