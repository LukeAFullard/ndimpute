import warnings

import numpy as np
import pandas as pd

from ._ros_left import impute_ros_left
from ._ros_right import impute_ros_right
from ._ros_mixed import impute_ros_mixed
from ._parametric import impute_mixed_parametric, impute_parametric_bounds, log_likelihood
from ._substitution import impute_sub_left, impute_sub_right, impute_sub_mixed
from ._interval import impute_interval_ros, impute_interval_parametric, _validate_bounds
from ._preprocess import parse_censored

_CENSORING_TYPES = (None, 'left', 'right', 'mixed', 'interval')

_ALLOWED_KWARGS = {
    'dist', 'plotting_position', 'impute_type', 'random_state', 'clamp',
    'mixed_strategy', 'right_dist', 'right_impute_type', 'lwp_compat',
    'strategy', 'multiplier', 'left_strategy', 'left_multiplier',
    'right_strategy', 'right_multiplier', 'left_marker', 'right_marker',
}

# String status codes, e.g. LWPTrends' CenType ("lt", "gt", "not")
_STATUS_STRINGS = {
    'lt': -1, '<': -1, 'left': -1,
    'gt': 1, '>': 1, 'right': 1,
    'not': 0, 'ok': 0, 'obs': 0, 'observed': 0, '': 0,
}


def _normalize_status(status, censoring_type, n):
    """
    Convert a user-supplied status to the internal code (-1 left, 0 observed,
    1 right) and resolve the censoring type.

    Rules:
      * Boolean status (True = censored): direction is given by censoring_type;
        'left' when censoring_type is None or 'mixed' (environmental convention).
      * Numeric status: 'left'/'right' accept {0, 1} (1 = censored); 'mixed'
        accepts {-1, 0, 1}. With censoring_type=None, a status containing -1 is
        mixed, and a {0, 1} status is LEFT-censored (1 = "<L").
      * String status: 'lt'/'<', 'gt'/'>', 'not'/'ok' (LWPTrends CenType).
    """
    arr = np.asarray(status)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"status must be 1-D with the same length as values ({n}); got shape {arr.shape}.")

    if arr.dtype == bool:
        flags = arr.astype(bool)
        if censoring_type == 'right':
            return np.where(flags, 1, 0), 'right'
        return np.where(flags, -1, 0), censoring_type or 'left'

    if arr.dtype.kind in 'OUS':
        try:
            codes = np.array([_STATUS_STRINGS[str(s).strip().lower()] for s in arr], dtype=int)
        except KeyError as e:
            raise ValueError(f"Unrecognised status string {e}. Use 'lt', 'gt' or 'not'.") from None
        inferred = 'mixed' if (np.any(codes == -1) and np.any(codes == 1)) else \
            ('right' if np.any(codes == 1) else 'left')
        ctype = censoring_type or inferred
        _check_status_matches_type(codes, ctype)
        return codes, ctype

    arr = arr.astype(float)
    if not np.all(np.isfinite(arr)) or not np.all(arr == np.round(arr)):
        raise ValueError("Numeric status must contain only integers (-1, 0, 1).")
    codes = arr.astype(int)
    uniq = set(np.unique(codes).tolist())

    if censoring_type in ('left', 'right'):
        if not uniq <= {0, 1}:
            raise ValueError(f"For censoring_type='{censoring_type}', status must be boolean or 0/1 "
                             f"(1 = censored); got values {sorted(uniq)}.")
        sign = -1 if censoring_type == 'left' else 1
        return codes * sign, censoring_type

    if not uniq <= {-1, 0, 1}:
        raise ValueError(f"Status must contain only -1 (left), 0 (observed), 1 (right); got {sorted(uniq)}.")
    if censoring_type == 'mixed':
        return codes, 'mixed'
    # censoring_type is None
    if -1 in uniq:
        return codes, 'mixed'
    # {0, 1} without an explicit type: left-censoring indicator (documented default)
    return -codes, 'left'


def _check_status_matches_type(codes, ctype):
    if ctype == 'left' and np.any(codes == 1):
        raise ValueError("censoring_type='left' was requested, but the data contain right-censored "
                         "values. Use censoring_type='mixed' (or omit it).")
    if ctype == 'right' and np.any(codes == -1):
        raise ValueError("censoring_type='right' was requested, but the data contain left-censored "
                         "values. Use censoring_type='mixed' (or omit it).")


def impute(values, status=None, method='ros', censoring_type=None, **kwargs):
    """
    Unified imputation function for censored data.

    Args:
        values (array-like): The data values.
            - Strings with markers (e.g. "<0.5", ">10") are parsed automatically.
            - Left/Right/Mixed: 1-D array of values (limits for censored entries).
            - Interval: (N, 2) array of (lower, upper) bounds.
            - Missing values (NaN, None, "", "NA") are carried through as NaN
              and excluded from fitting.
        status (array-like, optional): Censoring indicator. If omitted it is
            taken from the string markers.
            - Boolean (True = censored): left-censored unless
              censoring_type='right'.
            - Integer: -1 left, 0 observed, 1 right. A 0/1 status with no
              censoring_type is treated as LEFT-censoring (1 = "<L"); with
              censoring_type='right' 1 means ">C".
            - Strings: 'lt', 'gt', 'not' (LWPTrends CenType).
        method (str): 'ros' (default), 'parametric', or 'substitution'.
        censoring_type (str, optional): 'left', 'right', 'mixed', or 'interval'.
            Inferred from markers / status when omitted.
        **kwargs:
            dist: 'lognormal' (default), 'normal', 'weibull' (parametric only),
                or 'auto' (ROS: best probability-plot R^2; parametric: maximum
                log-likelihood).
            plotting_position: 'kaplan-meier' (default; Hirsch-Stedinger) or 'simple'.
            impute_type: ROS: 'quantile' (default, NADA), 'stochastic', 'mean'.
                Parametric: 'mean' (default) or 'stochastic'.
                Interval ROS: 'stochastic' (default) or 'mean'.
            random_state: seed for stochastic imputation.
            clamp: cap ROS imputations at their detection limit (default True;
                False reproduces NADA::ros exactly).
            mixed_strategy: mixed ROS strategy, 'sequential' (default, LWPTrends) or 'turnbull'.
            right_dist, right_impute_type, lwp_compat: see ``impute_ros_mixed``.
            strategy, multiplier, left_/right_strategy, left_/right_multiplier: substitution.
            left_marker, right_marker: string markers (default '<', '>').

    Returns:
        pd.DataFrame with columns:
            - 'imputed_value': observed value, or the imputation for censored values.
            - 'original_value' (or 'original_left'/'original_right' for intervals).
            - 'censoring_status': boolean (left/right), -1/0/1 (mixed), or 'interval'.
            - 'is_imputed': True where a value was imputed.
        ``df.attrs`` records the method, censoring type, distribution and fit details.
    """
    unknown = set(kwargs) - _ALLOWED_KWARGS
    if unknown:
        raise TypeError(f"Unknown keyword argument(s): {sorted(unknown)}.")
    if censoring_type not in _CENSORING_TYPES:
        raise ValueError("censoring_type must be 'left', 'right', 'mixed', or 'interval'.")
    if method not in ('ros', 'parametric', 'substitution'):
        raise ValueError(f"Unknown method '{method}'. Options: 'ros', 'parametric', 'substitution'.")

    dist = kwargs.get('dist', 'lognormal')

    if censoring_type == 'interval':
        return _impute_interval(values, method, dist, kwargs)

    # ---------------------------------------------------------------
    # 1. Parse values (strings with markers allowed)
    # ---------------------------------------------------------------
    l_marker = kwargs.get('left_marker', '<')
    r_marker = kwargs.get('right_marker', '>')
    vals, marker_status, has_left, has_right = parse_censored(values, l_marker, r_marker)
    n = len(vals)

    if status is None:
        codes = marker_status
        inferred = 'mixed' if (has_left and has_right) else ('right' if has_right else 'left')
        ctype = censoring_type or inferred
        _check_status_matches_type(codes, ctype)
    else:
        if has_left or has_right:
            warnings.warn("Censoring markers in `values` are ignored because `status` was supplied.",
                          UserWarning, stacklevel=2)
        codes, ctype = _normalize_status(status, censoring_type, n)
        _check_status_matches_type(codes, ctype)

    # ---------------------------------------------------------------
    # 2. Missing values
    # ---------------------------------------------------------------
    missing = np.isnan(vals)
    if np.any(np.isinf(vals)):
        raise ValueError("values contain infinite entries.")
    ok = ~missing
    v = vals[ok]
    s = codes[ok]

    # ---------------------------------------------------------------
    # 3. Dispatch
    # ---------------------------------------------------------------
    attrs = {'method': method, 'censoring_type': ctype}
    imputed_ok = _dispatch(v, s, ctype, method, dist, kwargs, attrs)

    imputed = np.full(n, np.nan)
    imputed[ok] = imputed_ok

    if ctype == 'mixed':
        status_out = codes
    elif ctype == 'right':
        status_out = codes == 1
    else:
        status_out = codes == -1

    df = pd.DataFrame({
        'imputed_value': imputed,
        'original_value': vals,
        'censoring_status': status_out,
        'is_imputed': (codes != 0) & ok,
    })
    if np.any(missing):
        attrs['n_missing'] = int(np.sum(missing))
    df.attrs.update(attrs)
    return df


def _dispatch(v, s, ctype, method, dist, kwargs, attrs):
    ros_kw = {k: kwargs[k] for k in ('impute_type', 'random_state', 'clamp') if k in kwargs}
    pp = kwargs.get('plotting_position', 'kaplan-meier')
    random_state = kwargs.get('random_state')

    if method == 'substitution':
        attrs['dist'] = None
        if ctype == 'left':
            return impute_sub_left(v, s == -1, strategy=kwargs.get('strategy', 'half'),
                                   multiplier=kwargs.get('multiplier'))
        if ctype == 'right':
            return impute_sub_right(v, s == 1, strategy=kwargs.get('strategy', 'value'),
                                    multiplier=kwargs.get('multiplier'))
        return impute_sub_mixed(
            v, s,
            left_kwargs={'strategy': kwargs.get('left_strategy', 'half'),
                         'multiplier': kwargs.get('left_multiplier')},
            right_kwargs={'strategy': kwargs.get('right_strategy', 'value'),
                          'multiplier': kwargs.get('right_multiplier')})

    if method == 'parametric':
        impute_type = kwargs.get('impute_type', 'mean')
        candidates = ['lognormal', 'normal', 'weibull'] if dist == 'auto' else [dist]
        lower_open = lambda d: 0.0 if d in ('lognormal', 'weibull') else -np.inf  # noqa: E731
        best = None
        errors = {}
        for d in candidates:
            if d in ('lognormal', 'weibull') and np.any(v <= 0):
                errors[d] = "non-positive values"
                continue
            try:
                imp, fit = impute_mixed_parametric(v, s, dist=d, impute_type=impute_type,
                                                   random_state=random_state, return_fit=True)
            except ValueError as e:
                if dist != 'auto':
                    raise
                errors[d] = str(e)
                continue
            lo = np.where(s == -1, lower_open(d), v)
            hi = np.where(s == 1, np.inf, v)
            ll = log_likelihood(fit, lo, hi)
            if best is None or ll > best[2]:
                best = (d, imp, ll, fit)
        if best is None:
            raise ValueError(f"No parametric distribution could be fitted: {errors}")
        d, imp, ll, fit = best
        attrs.update({'dist': d, 'impute_type': impute_type, 'fit': fit, 'log_likelihood': ll})
        if dist == 'auto':
            attrs.update({'best_dist': d, 'fit_score': ll})
        return imp

    # ---- ROS ----
    if dist not in ('lognormal', 'normal', 'auto'):
        raise ValueError(f"dist='{dist}' is not available for ROS. Use 'lognormal', 'normal' or 'auto'.")
    candidates = ['lognormal', 'normal'] if dist == 'auto' else [dist]
    mixed_kw = {k: kwargs[k] for k in ('mixed_strategy', 'right_dist', 'right_impute_type', 'lwp_compat')
                if k in kwargs}

    best = None
    errors = {}
    for d in candidates:
        if d == 'lognormal' and np.any(v <= 0):
            errors[d] = "non-positive values"
            continue
        try:
            if ctype == 'left':
                imp, r2 = impute_ros_left(v, s == -1, dist=d, plotting_position=pp, return_fit=True, **ros_kw)
            elif ctype == 'right':
                imp, r2 = impute_ros_right(v, s == 1, dist=d, plotting_position=pp, return_fit=True, **ros_kw)
            else:
                imp, r2 = impute_ros_mixed(v, s, dist=d, plotting_position=pp, return_fit=True,
                                           **ros_kw, **mixed_kw)
        except ValueError as e:
            if dist != 'auto':
                raise
            errors[d] = str(e)
            continue
        score = -np.inf if not np.isfinite(r2) else r2
        if best is None or score > best[2]:
            best = (d, imp, score)
    if best is None:
        raise ValueError(f"Auto-distribution selection failed; no candidate could be fitted: {errors}")
    d, imp, r2 = best
    attrs.update({'dist': d, 'impute_type': ros_kw.get('impute_type', 'quantile'),
                  'plotting_position': pp, 'r_squared': r2})
    if ctype == 'mixed':
        attrs['mixed_strategy'] = mixed_kw.get('mixed_strategy', 'sequential')
    if dist == 'auto':
        attrs.update({'best_dist': d, 'fit_score': r2})
    return imp


def _impute_interval(values, method, dist, kwargs):
    bounds = np.asarray(values, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("For censoring_type='interval', values must be (N, 2) array of bounds.")
    left, right = bounds[:, 0], bounds[:, 1]
    missing = np.isnan(left) | np.isnan(right)
    ok = ~missing
    random_state = kwargs.get('random_state')
    attrs = {'method': method, 'censoring_type': 'interval'}

    if method == 'ros':
        impute_type = kwargs.get('impute_type', 'stochastic')
        candidates = ['lognormal', 'normal'] if dist == 'auto' else [dist]
        best = None
        errors = {}
        for d in candidates:
            try:
                imp, r2 = impute_interval_ros(left[ok], right[ok], dist=d, impute_type=impute_type,
                                              random_state=random_state, return_fit=True)
            except ValueError as e:
                if dist != 'auto':
                    raise
                errors[d] = str(e)
                continue
            if best is None or r2 > best[2]:
                best = (d, imp, r2)
        if best is None:
            raise ValueError(f"Auto-distribution selection failed for interval data: {errors}")
        d, imp, r2 = best
        attrs.update({'dist': d, 'impute_type': impute_type, 'r_squared': r2})
        if dist == 'auto':
            attrs.update({'best_dist': d, 'fit_score': r2})
    elif method == 'parametric':
        impute_type = kwargs.get('impute_type', 'mean')
        candidates = ['lognormal', 'normal', 'weibull'] if dist == 'auto' else [dist]
        best = None
        errors = {}
        for d in candidates:
            try:
                imp, fit = impute_interval_parametric(left[ok], right[ok], dist=d, impute_type=impute_type,
                                                      random_state=random_state, return_fit=True)
                lo, hi = _validate_bounds(left[ok], right[ok], d)
                ll = log_likelihood(fit, lo, hi)
            except ValueError as e:
                if dist != 'auto':
                    raise
                errors[d] = str(e)
                continue
            if best is None or ll > best[2]:
                best = (d, imp, ll, fit)
        if best is None:
            raise ValueError(f"No parametric distribution could be fitted: {errors}")
        d, imp, ll, fit = best
        attrs.update({'dist': d, 'impute_type': impute_type, 'fit': fit, 'log_likelihood': ll})
        if dist == 'auto':
            attrs.update({'best_dist': d, 'fit_score': ll})
    else:
        raise NotImplementedError(f"Method '{method}' not implemented for interval censoring.")

    imputed = np.full(len(left), np.nan)
    imputed[ok] = imp
    df = pd.DataFrame({
        'imputed_value': imputed,
        'original_left': left,
        'original_right': right,
        'censoring_status': np.where(left == right, 'observed', 'interval'),
        'is_imputed': (left != right) & ok,
    })
    df.attrs.update(attrs)
    return df
