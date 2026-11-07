"""
Parametric (maximum-likelihood) imputation for arbitrarily censored data.

1. A distribution is fitted by maximum likelihood to all observations, using
   the full censored likelihood
       L(theta) = prod_exact f(x_i) * prod_left F(L_i) * prod_right S(R_i)
                  * prod_interval [F(r_i) - F(l_i)]
   (``scipy.stats.CensoredData``; equivalent to ``survival::survreg`` with an
   intercept-only model, see ``validation/11_r_parity``).
2. Each censored observation is replaced either by its conditional mean
   E[X | lo <= X <= hi; theta_hat] or by a random draw from the fitted
   distribution truncated to [lo, hi].

Conditional means and draws are computed with the numerically stable routines
in ``_truncated.py`` so imputations always respect their censoring bounds.

Note on uncertainty: imputations condition on the point estimate theta_hat.
Stochastic draws are therefore *improper* single imputations in the sense of
Rubin (1987): they reproduce the spread of the data but not the uncertainty in
theta. Standard errors computed from imputed data will be too small.
"""

import warnings

import numpy as np
from scipy.stats import norm, weibull_min, CensoredData

from ._truncated import (
    norm_trunc_mean, norm_trunc_sample,
    lognorm_trunc_mean, lognorm_trunc_sample,
    weibull_trunc_mean, weibull_trunc_sample,
)

_DISTS = ('lognormal', 'normal', 'weibull')


def _classify(lo, hi, positive):
    """Split bounds into exact / left / right / interval groups."""
    lower_open = (lo <= 0) if positive else np.isneginf(lo)
    upper_open = np.isposinf(hi)
    exact = (lo == hi)
    left = ~exact & lower_open & ~upper_open
    right = ~exact & ~lower_open & upper_open
    interval = ~exact & ~lower_open & ~upper_open
    both_open = ~exact & lower_open & upper_open
    return exact, left, right, interval, both_open


def fit_parametric(lo, hi, dist='lognormal'):
    """
    Maximum-likelihood fit of an intercept-only model to censored data.

    Args:
        lo, hi (array): Bounds for each observation. Exact: lo == hi.
            Left-censored: lo = -inf (or <= 0 for lognormal/weibull), hi = L.
            Right-censored: lo = R, hi = inf. Interval: finite lo < hi.
        dist (str): 'lognormal', 'normal' or 'weibull'.

    Returns:
        dict: {'dist', 'params', 'n_exact', 'n_censored', ...}.
            normal/lognormal params: {'mu', 'sigma'} (lognormal on log scale);
            weibull params: {'shape', 'scale'}.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if dist not in _DISTS:
        raise ValueError(f"Unknown distribution '{dist}' for parametric imputation. Options: {_DISTS}.")
    if np.any(lo > hi):
        raise ValueError("Lower bounds must not exceed upper bounds.")

    positive = dist in ('lognormal', 'weibull')
    if positive:
        # Exact values and upper bounds must be > 0; lower bounds >= 0 (0 = unbounded below).
        if np.any(hi <= 0) or np.any(lo < 0) or np.any(lo[lo == hi] <= 0):
            raise ValueError(f"Values must be positive for {dist} distribution.")

    exact, left, right, interval, both_open = _classify(lo, hi, positive)
    n_exact = int(np.sum(exact))
    n_bounded = n_exact + int(np.sum(interval))  # observations bounded on both sides
    if n_bounded < 2:
        raise ValueError("At least 2 exactly observed (or finite-interval) values are needed to fit "
                         "a parametric model.")
    if n_bounded < 5:
        warnings.warn(f"Only {n_bounded} exactly observed / finite-interval values; the parametric fit "
                      "is poorly determined.", UserWarning, stacklevel=3)

    keep = ~both_open  # (-inf, inf) carries no information

    if dist == 'lognormal':
        t = np.log
    else:
        t = lambda x: x  # noqa: E731

    cd = CensoredData(
        uncensored=t(lo[exact & keep]),
        left=t(hi[left & keep]),
        right=t(lo[right & keep]),
        interval=np.column_stack([t(lo[interval & keep]), t(hi[interval & keep])])
        if np.any(interval) else None,
    )

    if dist in ('lognormal', 'normal'):
        mu, sigma = norm.fit(cd)
        params = {'mu': float(mu), 'sigma': float(sigma)}
        ok = np.isfinite(mu) and np.isfinite(sigma) and sigma > 0
    else:
        shape, _, scale = weibull_min.fit(cd, floc=0)
        params = {'shape': float(shape), 'scale': float(scale)}
        ok = np.isfinite(shape) and np.isfinite(scale) and shape > 0 and scale > 0

    if not ok:
        raise ValueError(f"Maximum-likelihood fit for '{dist}' failed (parameters: {params}).")

    return {
        'dist': dist,
        'params': params,
        'n_exact': n_exact,
        'n_left': int(np.sum(left)),
        'n_right': int(np.sum(right)),
        'n_interval': int(np.sum(interval)),
    }


def impute_parametric_bounds(lo, hi, dist='lognormal', impute_type='mean', random_state=None,
                             return_fit=False):
    """
    Fit a distribution by MLE and impute every non-exact observation.

    Args:
        lo, hi (array): Bounds (see ``fit_parametric``).
        dist (str): 'lognormal', 'normal' or 'weibull'.
        impute_type (str): 'mean' (conditional expectation) or 'stochastic'
            (draw from the fitted distribution truncated to [lo, hi]).
        random_state (int or Generator, optional): Seed for 'stochastic'.
        return_fit (bool): If True, also return the fit dictionary.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if impute_type not in ('mean', 'stochastic'):
        raise ValueError(f"Unknown impute_type '{impute_type}' for parametric imputation. "
                         "Options: 'mean', 'stochastic'.")

    fit = fit_parametric(lo, hi, dist=dist)
    p = fit['params']

    imputed = lo.copy()
    cens = lo != hi
    if np.any(cens):
        l_c, h_c = lo[cens], hi[cens]
        if impute_type == 'stochastic':
            rng = np.random.default_rng(random_state)
            u = rng.uniform(size=int(np.sum(cens)))
            if dist == 'normal':
                imputed[cens] = norm_trunc_sample(p['mu'], p['sigma'], l_c, h_c, u)
            elif dist == 'lognormal':
                imputed[cens] = lognorm_trunc_sample(p['mu'], p['sigma'], l_c, h_c, u)
            else:
                imputed[cens] = weibull_trunc_sample(p['shape'], p['scale'], l_c, h_c, u)
        else:
            if dist == 'normal':
                imputed[cens] = norm_trunc_mean(p['mu'], p['sigma'], l_c, h_c)
            elif dist == 'lognormal':
                imputed[cens] = lognorm_trunc_mean(p['mu'], p['sigma'], l_c, h_c)
            else:
                imputed[cens] = weibull_trunc_mean(p['shape'], p['scale'], l_c, h_c)

    return (imputed, fit) if return_fit else imputed


def _status_to_bounds(data, status, dist):
    data = np.asarray(data, dtype=float)
    status = np.asarray(status, dtype=int)
    if not np.all(np.isin(status, (-1, 0, 1))):
        raise ValueError("status must contain only -1 (left), 0 (observed) and 1 (right).")
    lower_open = 0.0 if dist in ('lognormal', 'weibull') else -np.inf
    lo = np.where(status == -1, lower_open, data)
    hi = np.where(status == 1, np.inf, data)
    return lo, hi


def impute_mixed_parametric(values, status, dist='lognormal', impute_type='mean', random_state=None,
                            return_fit=False):
    """
    Parametric imputation for left-, right- and mixed-censored data.

    Args:
        values (array): Observed values or censoring limits.
        status (array): -1 left-censored ("<L"), 0 observed, 1 right-censored (">R").
        dist (str): 'lognormal', 'normal' or 'weibull'.
        impute_type (str): 'mean' (default) or 'stochastic'.
        random_state (int, optional): Seed for 'stochastic'.
        return_fit (bool): If True, returns (imputed, fit_dict).
    """
    if dist not in _DISTS:
        raise ValueError(f"Unknown distribution '{dist}' for parametric imputation. Options: {_DISTS}.")
    data = np.asarray(values, dtype=float)
    if dist in ('lognormal', 'weibull') and np.any(data <= 0):
        raise ValueError(f"Values must be positive for {dist} distribution.")
    lo, hi = _status_to_bounds(data, status, dist)
    return impute_parametric_bounds(lo, hi, dist=dist, impute_type=impute_type,
                                    random_state=random_state, return_fit=return_fit)


def impute_right_conditional(values, is_censored, dist='lognormal', impute_type='mean', random_state=None,
                             return_fit=False):
    """Parametric imputation for right-censored data (True = censored ">C")."""
    status = np.where(np.asarray(is_censored, dtype=bool), 1, 0)
    return impute_mixed_parametric(values, status, dist=dist, impute_type=impute_type,
                                   random_state=random_state, return_fit=return_fit)


def impute_left_parametric(values, is_censored, dist='lognormal', impute_type='mean', random_state=None,
                           return_fit=False):
    """Parametric imputation for left-censored data (True = censored "<L")."""
    status = np.where(np.asarray(is_censored, dtype=bool), -1, 0)
    return impute_mixed_parametric(values, status, dist=dist, impute_type=impute_type,
                                   random_state=random_state, return_fit=return_fit)


def log_likelihood(fit, lo, hi):
    """
    Censored log-likelihood of a fitted model on the ORIGINAL data scale
    (lognormal densities include the Jacobian), so values are comparable
    across distributions with the same number of parameters.
    """
    from scipy.stats import lognorm
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    p = fit['params']
    if fit['dist'] == 'normal':
        d = norm(loc=p['mu'], scale=p['sigma'])
        lo_eff = lo
    elif fit['dist'] == 'lognormal':
        d = lognorm(s=p['sigma'], scale=np.exp(p['mu']))
        lo_eff = np.where(lo < 0, 0.0, lo)
    else:
        d = weibull_min(p['shape'], scale=p['scale'])
        lo_eff = np.where(lo < 0, 0.0, lo)
    exact = lo_eff == hi
    ll = np.sum(d.logpdf(lo_eff[exact]))
    c = ~exact
    if np.any(c):
        with np.errstate(divide='ignore'):
            mass = d.cdf(hi[c]) - d.cdf(lo_eff[c])
            # Use survival functions in the upper tail for accuracy
            upper = lo_eff[c] > d.median()
            mass_up = d.sf(lo_eff[c]) - d.sf(hi[c])
            mass = np.where(upper, mass_up, mass)
            ll += np.sum(np.log(np.maximum(mass, 1e-300)))
    return float(ll)
