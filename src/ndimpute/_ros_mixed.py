"""
ROS-style imputation for mixed (left + right) censored data.

Two strategies:

``'sequential'`` (default) -- the LWPTrends procedure (Snelder, Fraser &
Larned, LWP Ltd; functions ``Impute.lower`` and ``Impute.upper``):

  1. Left-censored values ("<L") are imputed with Robust ROS (``NADA::ros``,
     Hirsch-Stedinger plotting positions). Right-censored values take part in
     this step as if they were detected at their limit.
  2. Right-censored values (">R") are imputed by fitting a distribution by
     maximum likelihood (``survival::survreg``) to the step-1 values, with ">R"
     values right-censored, and drawing each imputation from the fitted
     distribution conditional on exceeding its limit. LWPTrends always uses a
     Weibull here; ndimpute defaults to the same family as the ROS step
     (``right_dist=None``) and uses Weibull when ``lwp_compat=True``.

``'turnbull'`` (experimental) -- mixed data are recast as intervals
([0 or -inf, L], [x, x], [R, inf]) and imputed with ``impute_interval_ros``.
This is not an established published estimator; prefer ``'sequential'`` or
``method='parametric'``.
"""

import warnings

import numpy as np

from ._ros_left import impute_ros_left
from ._interval import impute_interval_ros
from ._parametric import impute_mixed_parametric, fit_parametric
from ._truncated import weibull_trunc_sample

_STRATEGIES = ('sequential', 'turnbull')


def impute_ros_mixed(values, status, dist='lognormal', plotting_position='kaplan-meier',
                     return_fit=False, mixed_strategy='sequential', impute_type=None,
                     random_state=None, right_dist=None, right_impute_type='stochastic',
                     lwp_compat=False, clamp=True, **kwargs):
    """
    Impute mixed-censored data with an ROS-based strategy.

    Args:
        values (array): Observed values or censoring limits.
        status (array): -1 left-censored, 0 observed, 1 right-censored.
        dist (str): Distribution for the left-censored ROS step ('lognormal' or 'normal').
        plotting_position (str): ROS plotting positions (see ``impute_ros_left``).
        return_fit (bool): If True, return (imputed, r_squared) where r_squared is
            that of the ROS regression (left step, or Turnbull fit).
        mixed_strategy (str): 'sequential' (default, LWPTrends) or 'turnbull'.
        impute_type (str, optional): Imputation mode for the ROS step.
            'sequential': 'quantile' (default), 'stochastic' or 'mean'.
            'turnbull': 'stochastic' (default) or 'mean'.
        random_state: Seed or Generator for all random draws.
        right_dist (str, optional): Distribution for the right-censored step of
            'sequential': 'weibull' (as in LWPTrends), 'lognormal' or 'normal'.
            Defaults to ``dist`` (one distributional assumption for both tails),
            or to 'weibull' when ``lwp_compat=True``. Monte Carlo (validation/12)
            shows a Weibull right tail under-estimates the mean of lognormal
            data by ~8%; matching ``dist`` reduces this to <1%.
        right_impute_type (str): 'stochastic' (default, LWPTrends) or 'mean'.
        lwp_compat (bool): Reproduce the LWPTrends v2502 fall-back rules exactly:
            left-censored limits >= the largest detect are reset to half that detect;
            ROS is only used with more than 10 detects (otherwise "<L" -> L/2);
            right-censored values are multiplied by 1.1 when n < 24; and the
            Weibull conditioning point is capped at the fitted 99.5th percentile.
        clamp (bool): Cap ROS imputations at their detection limit (default True;
            LWPTrends/NADA do not cap -- set False, or lwp_compat=True, to match).
    """
    values = np.asarray(values, dtype=float)
    status = np.asarray(status, dtype=int)
    if not np.all(np.isin(status, (-1, 0, 1))):
        raise ValueError("status must contain only -1 (left), 0 (observed) and 1 (right).")
    if mixed_strategy not in _STRATEGIES:
        raise ValueError(f"Unknown mixed_strategy '{mixed_strategy}'. Options: {_STRATEGIES}.")

    rng = np.random.default_rng(random_state)

    if mixed_strategy == 'turnbull':
        warnings.warn("mixed_strategy='turnbull' is experimental (not a published estimator); "
                      "prefer 'sequential' or method='parametric'.", UserWarning, stacklevel=2)
        lower_open = 0.0 if dist == 'lognormal' else -np.inf
        left = np.where(status == -1, lower_open, values)
        right = np.where(status == 1, np.inf, values)
        return impute_interval_ros(left, right, dist=dist, impute_type=impute_type or 'stochastic',
                                   random_state=rng, return_fit=return_fit)

    if right_dist is None:
        right_dist = 'weibull' if lwp_compat else dist
    return _impute_sequential(values, status, dist, plotting_position, return_fit,
                              impute_type or 'quantile', rng, right_dist, right_impute_type,
                              lwp_compat, clamp)


# Backwards-compatible name
def impute_ros_mixed_heuristic(values, status, return_fit=False, **kwargs):
    return impute_ros_mixed(values, status, return_fit=return_fit, **kwargs)


def _impute_sequential(values, status, dist, plotting_position, return_fit, impute_type, rng,
                       right_dist, right_impute_type, lwp_compat, clamp):
    n = len(values)
    mask_left = status == -1
    mask_right = status == 1
    r_squared = np.nan

    # ---------------- Step 1: left-censored values (LWPTrends Impute.lower) ----------------
    step1 = values.copy()
    if np.any(mask_left):
        ros_values = values.copy()
        if lwp_compat:
            # LWPTrends: "<L" with L >= max(non-"<" values) are reset to 0.5 * max
            max_uncen = np.max(values[~mask_left]) if np.any(~mask_left) else np.nan
            bad = mask_left & (values >= max_uncen)
            ros_values[bad] = 0.5 * max_uncen
            # LWPTrends requires > minObsforfit (10) non-censored (neither < nor >) values
            n_detect = int(np.sum(status == 0))
            use_ros = n_detect > 10 and np.any(~mask_left)
        else:
            use_ros = True

        if use_ros:
            if lwp_compat:
                clamp = False
            try:
                fitted = impute_ros_left(ros_values, mask_left, dist=dist,
                                         plotting_position=plotting_position, return_fit=True,
                                         impute_type=impute_type, random_state=rng, clamp=clamp)
                step1, r_squared = fitted
                step1 = np.asarray(step1, dtype=float)
                step1[~mask_left] = values[~mask_left]
            except ValueError:
                if not lwp_compat:
                    raise
                use_ros = False
        if not use_ros:
            warnings.warn("ROS not fitted (LWPTrends rule: <= 10 detects or fit failure); "
                          "left-censored values set to half their limit.", UserWarning, stacklevel=3)
            step1 = values.copy()
            step1[mask_left] = values[mask_left] / 2.0

    # ---------------- Step 2: right-censored values (LWPTrends Impute.upper) ----------------
    result = step1.copy()
    if np.any(mask_right):
        if np.all(mask_right):
            raise ValueError("All observations are right-censored; cannot fit a model.")
        if lwp_compat and n < 24:
            warnings.warn("n < 24: right-censored values multiplied by 1.1 (LWPTrends rule).",
                          UserWarning, stacklevel=3)
            result[mask_right] = 1.1 * values[mask_right]
        else:
            right_status = np.where(mask_right, 1, 0)
            if lwp_compat and right_dist == 'weibull' and right_impute_type == 'stochastic':
                fit = fit_parametric(np.where(mask_right, values, step1),
                                     np.where(mask_right, np.inf, step1), dist='weibull')
                k, lam = fit['params']['shape'], fit['params']['scale']
                cap = lam * (-np.log1p(-0.995)) ** (1.0 / k)  # 99.5th percentile
                lo = np.minimum(values[mask_right], cap)
                u = rng.uniform(size=int(np.sum(mask_right)))
                result[mask_right] = weibull_trunc_sample(k, lam, lo, np.inf, u)
            else:
                imputed = impute_mixed_parametric(step1, right_status, dist=right_dist,
                                                  impute_type=right_impute_type, random_state=rng)
                result[mask_right] = imputed[mask_right]

    result[status == 0] = values[status == 0]
    return (result, r_squared) if return_fit else result
