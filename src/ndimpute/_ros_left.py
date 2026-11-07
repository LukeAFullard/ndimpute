"""
Robust Regression on Order Statistics (ROS) for left-censored data.

Reference implementation: ``NADA::ros`` (Lee & Helsel 2005; Helsel 2012,
*Statistics for Censored Environmental Data*, 2nd ed., ch. 6), which uses the
Hirsch & Stedinger (1987) plotting positions in the reformulation of Helsel &
Cohn (1988). This is also the ROS used by LWPTrends (``Impute.lower``).

With ``plotting_position='kaplan-meier'``, ``impute_type='quantile'`` and
``clamp=False`` this reproduces ``NADA::ros`` ``$modeled`` exactly (see
``validation/11_r_parity``). The default ``clamp=True`` additionally caps each
imputation at its own detection limit, which NADA does not do.
"""

import warnings

import numpy as np
from scipy.stats import norm, linregress

from ._truncated import norm_trunc_mean, lognorm_trunc_mean

_HS_ALIASES = ("kaplan-meier", "hirsch-stedinger", "hc", "ecdf")
_SIMPLE_ALIASES = ("simple", "weibull")
_IMPUTE_TYPES = ("quantile", "stochastic", "mean")


def cohn_numbers(values, is_censored):
    """
    Helsel-Cohn (1988) quantities for the Hirsch-Stedinger plotting positions.

    Port of ``NADA:::cohn``. For each detection limit L_j (sorted ascending):

    * A_j: number of detected values in [L_j, L_{j+1})
    * B_j: number of values (detected or censored) <= L_j, excluding detects equal to L_j
    * C_j: number of values censored at L_j
    * P_j: probability of exceeding L_j, P_j = P_{j+1} + A_j/(A_j+B_j) (1 - P_{j+1})

    If any detect lies below the lowest limit a pseudo-limit is added below all
    data. NADA uses 0 for this pseudo-limit, which silently drops negative
    detects (only possible on the untransformed/normal scale); ``-inf`` is used
    here, which is identical for non-negative data.
    """
    values = np.asarray(values, dtype=float)
    is_censored = np.asarray(is_censored, dtype=bool)
    uncen = values[~is_censored]
    cen = values[is_censored]

    limits = np.unique(cen)
    if np.any(uncen < limits[0]):
        limits = np.concatenate([[-np.inf], limits])

    m = len(limits)
    A = np.zeros(m)
    B = np.zeros(m)
    C = np.zeros(m)
    P = np.zeros(m)
    for i in range(m - 1, -1, -1):
        lim = limits[i]
        upper = limits[i + 1] if i < m - 1 else np.inf
        A[i] = np.sum((uncen >= lim) & (uncen < upper))
        B[i] = np.sum(values <= lim) - np.sum(uncen == lim)
        C[i] = np.sum(cen == lim)
        p_next = P[i + 1] if i < m - 1 else 0.0
        denom = A[i] + B[i]
        P[i] = p_next + (A[i] / denom) * (1.0 - p_next) if denom > 0 else p_next
    return limits, A, B, C, P


def hirsch_stedinger_ppoints(values, is_censored):
    """
    Hirsch-Stedinger plotting positions (port of ``NADA::hc.ppoints``).

    Returns:
        tuple: (pp, pe_limit)
            pp: plotting position for every observation (original order).
                Detects in [L_j, L_{j+1}) get (1-P_j) + (P_j-P_{j+1}) r/(A_j+1);
                values censored at L_j get (1-P_j) r/(C_j+1), r = rank within group.
            pe_limit: for every censored observation, (1 - P_j), the estimated
                probability of lying below its own detection limit.
    """
    values = np.asarray(values, dtype=float)
    is_censored = np.asarray(is_censored, dtype=bool)
    n = len(values)
    pp = np.full(n, np.nan)
    pe_limit = np.full(n, np.nan)

    limits, A, B, C, P = cohn_numbers(values, is_censored)
    m = len(limits)
    # Stable sort so tie-breaking matches R's order() on sorted input
    order = np.argsort(values, kind="stable")
    v_sorted = values[order]
    c_sorted = is_censored[order]

    for i in range(m):
        lim = limits[i]
        upper = limits[i + 1] if i < m - 1 else np.inf
        p_next = P[i + 1] if i < m - 1 else 0.0

        sel_u = (~c_sorted) & (v_sorted >= lim) & (v_sorted < upper)
        idx_u = order[sel_u]
        if len(idx_u):
            r = np.arange(1, len(idx_u) + 1)
            pp[idx_u] = (1.0 - P[i]) + (P[i] - p_next) * r / (len(idx_u) + 1)

        sel_c = c_sorted & (v_sorted == lim)
        idx_c = order[sel_c]
        if len(idx_c):
            r = np.arange(1, len(idx_c) + 1)
            pp[idx_c] = (1.0 - P[i]) * r / (len(idx_c) + 1)
            pe_limit[idx_c] = 1.0 - P[i]

    return pp, pe_limit


def weibull_ppoints(values, is_censored):
    """
    Legacy 'simple' plotting positions: rank/(n+1) over all observations,
    with censored values ranked below detects of equal value.

    Only appropriate for a single detection limit lying below every detect.
    """
    values = np.asarray(values, dtype=float)
    is_censored = np.asarray(is_censored, dtype=bool)
    n = len(values)
    # lexsort: last key is primary. Censored (<L) sorts before a detect at L.
    order = np.lexsort((~is_censored, values))
    pp = np.empty(n)
    pp[order] = np.arange(1, n + 1) / (n + 1)

    # Probability below each limit, chosen so that the quantile positions of
    # the C values censored at that limit are pe * r / (C + 1) = rank / (n + 1).
    pe_limit = np.full(n, np.nan)
    for lim in np.unique(values[is_censored]):
        sel = is_censored & (values == lim)
        pe_limit[sel] = np.max(pp[sel]) + 1.0 / (n + 1)
    return pp, pe_limit


def impute_ros_left(values, is_censored, dist='lognormal', plotting_position='kaplan-meier',
                    return_fit=False, impute_type='quantile', random_state=None,
                    clamp=True, **kwargs):
    """
    Impute left-censored data ("<L") with Robust ROS.

    Detected values are returned unchanged. A straight line is fitted by OLS
    to (normal quantile of plotting position, transformed detect); censored
    values are then predicted from that line.

    Args:
        values (array): Detected values, or the detection limit for censored values.
        is_censored (bool array): True where the value is "<L".
        dist (str): 'lognormal' (regress log values; the NADA default) or 'normal'.
        plotting_position (str):
            - 'kaplan-meier' (default; aliases 'hirsch-stedinger', 'hc'):
              Hirsch-Stedinger / Helsel-Cohn positions, identical to NADA. Correct
              for multiple detection limits.
            - 'simple' (alias 'weibull'): rank/(n+1). Legacy; only valid for a
              single detection limit below all detects. A warning is issued otherwise.
        return_fit (bool): If True, return (imputed, r_squared).
        impute_type (str):
            - 'quantile' (default): censored values at limit L_j receive the
              deterministic NADA positions (1-P_j) r/(C_j+1). Reproduces NADA.
            - 'stochastic': random positions p ~ U(0, 1-P_j) (needs ``random_state``
              for reproducibility).
            - 'mean': conditional mean of the fitted distribution below the
              limit's plotting position, on the ORIGINAL scale, i.e.
              E[X | Z < z_j] with z_j = Phi^-1(1-P_j).
        random_state (int or Generator, optional): Seed for 'stochastic'.
        clamp (bool): If True (default), imputed values are capped at their
            detection limit, so no "<L" value is imputed above L. NADA::ros does
            not cap; use clamp=False to reproduce NADA exactly.

    Returns:
        ndarray (or tuple with r_squared if return_fit=True).
    """
    values = np.asarray(values, dtype=float)
    is_censored = np.asarray(is_censored, dtype=bool)
    n = len(values)

    if impute_type not in _IMPUTE_TYPES:
        raise ValueError(f"Unknown impute_type '{impute_type}'. Options: {_IMPUTE_TYPES}.")
    if dist not in ('lognormal', 'normal'):
        raise ValueError(f"Unknown distribution '{dist}' for ROS. Options: 'lognormal', 'normal'.")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contain NaN or infinite entries.")

    n_unc = int(np.sum(~is_censored))
    n_cen = n - n_unc

    if n_unc < 2:
        raise ValueError("Too few uncensored observations to fit regression (need at least 2).")
    if dist == 'lognormal' and np.any(values <= 0):
        raise ValueError("Values (including detection limits) must be positive for lognormal distribution.")

    def _transform(x):
        return np.log(x) if dist == 'lognormal' else x

    def _inverse(y):
        return np.exp(y) if dist == 'lognormal' else y

    if n_cen == 0:
        # Nothing to impute. Fit is still reported for distribution selection.
        pp = (np.argsort(np.argsort(values, kind="stable")) + 1) / (n + 1)
        slope, intercept, r_value, _, _ = linregress(norm.ppf(pp), _transform(values))
        return (values.copy(), r_value ** 2) if return_fit else values.copy()

    if n_cen / n > 0.8:
        warnings.warn(f"{n_cen / n:.0%} of observations are censored (>80%); ROS results are tenuous.",
                      UserWarning, stacklevel=2)
    if n_unc < 5:
        warnings.warn(f"Only {n_unc} detected values; the ROS regression is poorly determined.",
                      UserWarning, stacklevel=2)
    if np.any(values[is_censored] > np.max(values[~is_censored])):
        warnings.warn("Some detection limits exceed the largest detected value; their imputations "
                      "are poorly constrained (NADA drops such values).", UserWarning, stacklevel=2)

    # --- Plotting positions ---
    if plotting_position in _HS_ALIASES:
        pp, pe_limit = hirsch_stedinger_ppoints(values, is_censored)
    elif plotting_position in _SIMPLE_ALIASES:
        lims = np.unique(values[is_censored])
        if len(lims) > 1 or np.any(values[~is_censored] < lims.max()):
            warnings.warn("plotting_position='simple' is only valid for a single detection limit lying "
                          "below all detects; it is biased here. Use 'kaplan-meier' (Hirsch-Stedinger).",
                          UserWarning, stacklevel=2)
        pp, pe_limit = weibull_ppoints(values, is_censored)
    else:
        raise ValueError(f"Unknown plotting_position '{plotting_position}'.")

    # --- Regression on detects ---
    unc = ~is_censored
    z_unc = norm.ppf(pp[unc])
    y_unc = _transform(values[unc])
    if np.ptp(z_unc) == 0 or np.ptp(y_unc) == 0:
        raise ValueError("Detected values are all identical; ROS regression is undefined.")
    slope, intercept, r_value, _, _ = linregress(z_unc, y_unc)
    r_squared = r_value ** 2
    if not slope > 0:
        raise ValueError("ROS regression has non-positive slope; the probability-plot model is invalid "
                         "for these data.")

    # --- Imputation ---
    cen_idx = np.where(is_censored)[0]
    if impute_type == 'quantile':
        z_cen = norm.ppf(pp[cen_idx])
        imputed = _inverse(intercept + slope * z_cen)
    elif impute_type == 'stochastic':
        rng = np.random.default_rng(random_state)
        u = rng.uniform(size=len(cen_idx))
        p = np.clip(u * pe_limit[cen_idx], np.finfo(float).tiny, None)
        imputed = _inverse(intercept + slope * norm.ppf(p))
    else:  # 'mean' -- E[X | Z < z_limit] on the original scale
        z_lim = norm.ppf(pe_limit[cen_idx])
        y_lim = intercept + slope * z_lim  # transformed-scale value at the limit's position
        if dist == 'lognormal':
            imputed = lognorm_trunc_mean(intercept, slope, 0.0, np.exp(y_lim))
        else:
            imputed = norm_trunc_mean(intercept, slope, -np.inf, y_lim)

    above = imputed > values[cen_idx]
    if clamp:
        if np.mean(above) > 0.5:
            warnings.warn(f"{int(np.sum(above))} of {len(above)} censored values were predicted above their "
                          "detection limit and capped at it; the ROS model fits these data poorly.",
                          UserWarning, stacklevel=2)
        imputed = np.minimum(imputed, values[cen_idx])
    elif np.any(imputed > values[cen_idx]):
        warnings.warn(f"{int(np.sum(imputed > values[cen_idx]))} censored value(s) were imputed above their "
                      "own detection limit by the ROS line (as NADA::ros does).", UserWarning, stacklevel=2)

    result = values.copy()
    result[cen_idx] = imputed

    if return_fit:
        return result, r_squared
    return result
