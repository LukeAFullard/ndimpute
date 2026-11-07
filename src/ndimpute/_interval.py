"""
Imputation for interval-censored data.

Two approaches are provided:

* ``impute_interval_parametric`` (recommended): maximum-likelihood fit of a
  parametric distribution to the interval-censored likelihood, then
  conditional means / truncated draws within each interval.
* ``impute_interval_ros``: a semi-parametric probability-plot method. The
  Turnbull NPMLE supplies cumulative probabilities for its innermost
  intervals; a weighted least-squares line is fitted through
  (normal quantile of the mid-cumulative probability, interval midpoint), and
  the line's (mu, sigma) are used to impute within each observation's bounds.
  This generalises ROS to interval data but, unlike ROS for left-censored
  data, it is not an established published estimator; treat it as
  exploratory and prefer the parametric method for inference.
"""

import numpy as np
from scipy.stats import norm

from ._turnbull import turnbull_em
from ._truncated import (
    norm_trunc_mean, norm_trunc_sample, lognorm_trunc_mean, lognorm_trunc_sample,
)
from ._parametric import impute_parametric_bounds


def _validate_bounds(left, right, dist):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("left and right bounds must have the same shape.")
    if np.any(np.isnan(left)) or np.any(np.isnan(right)):
        raise ValueError("Interval bounds contain NaN.")
    if np.any(left > right):
        raise ValueError("Interval lower bounds must not exceed upper bounds.")
    if dist in ('lognormal', 'weibull'):
        if np.any(right <= 0):
            raise ValueError(f"Upper bounds must be positive for {dist} distribution.")
        exact = left == right
        if np.any(left[exact] <= 0):
            raise ValueError(f"Exact values must be positive for {dist} distribution.")
        left = np.where(left < 0, 0.0, left)  # (-inf, R] == [0, R] for a positive variable
    return left, right


def impute_interval_parametric(left, right, dist='lognormal', impute_type='mean', random_state=None,
                               return_fit=False):
    """
    Parametric MLE imputation for interval-censored data.

    Args:
        left, right (array): Bounds; left == right for exact values, left = -inf
            (or 0 for positive distributions) for left-censored, right = inf for
            right-censored.
        dist (str): 'lognormal', 'normal' or 'weibull'.
        impute_type (str): 'mean' or 'stochastic'.
        random_state: Seed for 'stochastic'.
        return_fit (bool): If True, returns (imputed, fit_dict).
    """
    left, right = _validate_bounds(left, right, dist)
    return impute_parametric_bounds(left, right, dist=dist, impute_type=impute_type,
                                    random_state=random_state, return_fit=return_fit)


def impute_interval_ros(left, right, dist='lognormal', impute_type='stochastic', random_state=None,
                        return_fit=False):
    """
    Turnbull-based probability-plot imputation for interval-censored data.

    Args:
        left (array): Lower bounds.
        right (array): Upper bounds.
        dist (str): 'lognormal' or 'normal'.
        impute_type (str): 'stochastic' (default; random draw within the
            interval from the fitted distribution) or 'mean' (conditional mean
            within the interval, on the original scale).
        random_state (int, np.random.Generator, optional): Seed for 'stochastic'.
        return_fit (bool): If True, returns (imputed_values, r_squared).
    """
    if dist not in ('normal', 'lognormal'):
        raise ValueError(f"Unknown distribution '{dist}'. Supported: 'normal', 'lognormal'.")
    if impute_type not in ('stochastic', 'mean'):
        raise ValueError(f"Unknown impute_type '{impute_type}' for interval ROS. Options: 'stochastic', 'mean'.")
    left, right = _validate_bounds(left, right, dist)

    # 1. Turnbull NPMLE
    intervals, probs = turnbull_em(left, right)

    # 2. Plotting positions: mid-cumulative probability of each innermost interval
    cdf_vals = np.cumsum(probs)
    pp = np.clip(cdf_vals - probs / 2.0, 1e-12, 1 - 1e-12)
    z = norm.ppf(pp)

    s, e = intervals[:, 0], intervals[:, 1]
    with np.errstate(invalid='ignore', divide='ignore'):
        if dist == 'lognormal':
            mids = np.sqrt(s * e)  # geometric midpoint
            y = np.log(mids)
        else:
            mids = 0.5 * (s + e)
            y = mids
    valid = np.isfinite(y) & (probs > 1e-12)
    if np.sum(valid) < 2:
        raise ValueError("Turnbull yielded insufficient finite intervals (points) to fit regression.")

    y_fit, z_fit, w = y[valid], z[valid], probs[valid]

    # 3. Weighted least squares
    zm = np.average(z_fit, weights=w)
    ym = np.average(y_fit, weights=w)
    s_zy = np.sum(w * (z_fit - zm) * (y_fit - ym))
    s_zz = np.sum(w * (z_fit - zm) ** 2)
    s_yy = np.sum(w * (y_fit - ym) ** 2)
    if s_zz <= 0 or s_yy <= 0:
        raise ValueError("Degenerate probability plot (no spread); interval ROS cannot be fitted.")
    slope = s_zy / s_zz
    intercept = ym - slope * zm
    r_squared = s_zy ** 2 / (s_zz * s_yy)
    if not slope > 0:
        raise ValueError("Interval ROS regression has non-positive slope; the model is invalid for these data.")

    # 4. Impute within each observation's bounds using (mu, sigma) = (intercept, slope)
    imputed = left.copy()
    cens = left != right
    if np.any(cens):
        lo, hi = left[cens], right[cens]
        if impute_type == 'stochastic':
            rng = np.random.default_rng(random_state)
            u = rng.uniform(size=int(np.sum(cens)))
            if dist == 'lognormal':
                imputed[cens] = lognorm_trunc_sample(intercept, slope, lo, hi, u)
            else:
                imputed[cens] = norm_trunc_sample(intercept, slope, lo, hi, u)
        else:
            if dist == 'lognormal':
                imputed[cens] = lognorm_trunc_mean(intercept, slope, lo, hi)
            else:
                imputed[cens] = norm_trunc_mean(intercept, slope, lo, hi)

    if return_fit:
        return imputed, r_squared
    return imputed
