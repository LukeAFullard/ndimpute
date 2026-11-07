"""
Numerically stable conditional (truncated) moments and samplers.

Every imputation in ndimpute ultimately needs one of two quantities for a
censored observation known to lie in ``[lo, hi]``:

* the conditional mean ``E[X | lo <= X <= hi]``; or
* a random draw from the distribution of ``X`` truncated to ``[lo, hi]``.

Naive implementations (``1 - cdf``, inverse-CDF on clipped uniforms, ratio of
two underflowing probabilities) fail in the distribution tails: they return
values on the wrong side of the censoring limit, ``-inf``, or silently fall back
to the limit itself. The helpers below work in log-probability space, or use
exact transformations, so that results are valid for arbitrarily extreme
truncation points. All results are finally clipped to ``[lo, hi]`` to remove
floating-point round-off.

Bounds may be infinite (``-inf`` / ``inf``). For the lognormal and Weibull
distributions a lower bound ``<= 0`` means "no lower bound".
"""

import numpy as np
from scipy.special import log_ndtr, ndtri, gammainc, gammaincc, gammaln
from scipy.stats import truncnorm

__all__ = [
    "log_norm_interval_prob",
    "norm_trunc_mean",
    "norm_trunc_sample",
    "lognorm_trunc_mean",
    "lognorm_trunc_sample",
    "weibull_trunc_mean",
    "weibull_trunc_sample",
]


def _as_float_arrays(*arrs):
    out = np.broadcast_arrays(*[np.atleast_1d(np.asarray(a, dtype=float)) for a in arrs])
    return [np.array(a, dtype=float) for a in out]


def log_norm_interval_prob(a, b):
    """
    log P(a < Z < b) for a standard normal Z, stable in both tails.

    Args:
        a, b (array-like): Standardised bounds with a <= b (may be +/- inf).

    Returns:
        ndarray: log-probabilities (``-inf`` where a == b).
    """
    a, b = _as_float_arrays(a, b)
    out = np.full(a.shape, -np.inf)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Upper tail: use survival functions  P = S(a) - S(b),  log S(x) = log_ndtr(-x)
        upper = a > 0
        if np.any(upper):
            la, lb = log_ndtr(-a[upper]), log_ndtr(-b[upper])
            out[upper] = la + np.log1p(-np.exp(lb - la))

        # Lower tail (and straddling zero): P = Phi(b) - Phi(a)
        lower = ~upper
        if np.any(lower):
            la, lb = log_ndtr(a[lower]), log_ndtr(b[lower])
            out[lower] = lb + np.log1p(-np.exp(la - lb))

    out[a >= b] = -np.inf
    return out


def norm_trunc_mean(mu, sigma, lo, hi):
    """
    E[X | lo <= X <= hi] for X ~ Normal(mu, sigma).
    """
    mu, sigma, lo, hi = _as_float_arrays(mu, sigma, lo, hi)
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    degenerate = ~(a < b)

    out = np.empty(a.shape)
    ok = ~degenerate
    if np.any(ok):
        # scipy's truncnorm.mean is stable in the tails
        out[ok] = mu[ok] + sigma[ok] * truncnorm.mean(a[ok], b[ok])
    out[degenerate] = lo[degenerate]
    return _clip(out, lo, hi)


def norm_trunc_sample(mu, sigma, lo, hi, u):
    """
    Inverse-CDF draw from Normal(mu, sigma) truncated to [lo, hi].

    Args:
        u (array-like): Uniform(0, 1) variates (supplied by the caller so that
            the caller controls the random stream).
    """
    mu, sigma, lo, hi, u = _as_float_arrays(mu, sigma, lo, hi, u)
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    degenerate = ~(a < b)

    out = np.empty(a.shape)
    ok = ~degenerate
    if np.any(ok):
        out[ok] = mu[ok] + sigma[ok] * truncnorm.ppf(u[ok], a[ok], b[ok])
    out[degenerate] = lo[degenerate]
    return _clip(out, lo, hi)


def lognorm_trunc_mean(mu, sigma, lo, hi):
    """
    E[X | lo <= X <= hi] for log(X) ~ Normal(mu, sigma), on the ORIGINAL scale.

    Uses  E = exp(mu + sigma^2/2) * P(a - sigma < Z < b - sigma) / P(a < Z < b),
    evaluated in log space.
    """
    mu, sigma, lo, hi = _as_float_arrays(mu, sigma, lo, hi)
    with np.errstate(divide="ignore"):
        log_lo = np.where(lo > 0, np.log(np.where(lo > 0, lo, 1.0)), -np.inf)
        log_hi = np.where(np.isinf(hi), np.inf, np.log(np.where(hi > 0, hi, 1.0)))
    a = (log_lo - mu) / sigma
    b = (log_hi - mu) / sigma

    degenerate = ~(a < b)
    out = np.empty(a.shape)
    ok = ~degenerate
    if np.any(ok):
        log_num = log_norm_interval_prob(a[ok] - sigma[ok], b[ok] - sigma[ok])
        log_den = log_norm_interval_prob(a[ok], b[ok])
        out[ok] = np.exp(mu[ok] + 0.5 * sigma[ok] ** 2 + log_num - log_den)
    out[degenerate] = lo[degenerate]
    lo_eff = np.where(lo > 0, lo, 0.0)
    return _clip(out, lo_eff, hi)


def lognorm_trunc_sample(mu, sigma, lo, hi, u):
    """
    Draw from a lognormal (log-scale parameters mu, sigma) truncated to [lo, hi].
    """
    mu, sigma, lo, hi, u = _as_float_arrays(mu, sigma, lo, hi, u)
    with np.errstate(divide="ignore"):
        log_lo = np.where(lo > 0, np.log(np.where(lo > 0, lo, 1.0)), -np.inf)
        log_hi = np.where(np.isinf(hi), np.inf, np.log(np.where(hi > 0, hi, 1.0)))
    z = norm_trunc_sample(mu, sigma, log_lo, log_hi, u)
    out = np.exp(z)
    lo_eff = np.where(lo > 0, lo, 0.0)
    return _clip(out, lo_eff, hi)


def _weibull_u(x, shape, scale):
    """Cumulative hazard (x/scale)^shape, with x <= 0 -> 0 and inf -> inf."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u = np.where(x > 0, (np.where(x > 0, x, 1.0) / scale) ** shape, 0.0)
    u = np.where(np.isposinf(x), np.inf, u)
    return u


def weibull_trunc_mean(shape, scale, lo, hi):
    """
    E[X | lo <= X <= hi] for X ~ Weibull(shape k, scale lambda).

    Uses  E = lambda * Gamma(1 + 1/k) * [P(1+1/k, u_hi) - P(1+1/k, u_lo)]
                / [exp(-u_lo) - exp(-u_hi)],   u = (x/lambda)^k,
    computed with whichever of the lower/upper regularised incomplete gamma
    functions is accurate, and an asymptotic expansion when both underflow.
    """
    shape, scale, lo, hi = _as_float_arrays(shape, scale, lo, hi)
    a = 1.0 + 1.0 / shape
    u_lo = _weibull_u(lo, shape, scale)
    u_hi = _weibull_u(hi, shape, scale)

    out = np.empty(u_lo.shape)
    degenerate = ~(u_lo < u_hi)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        # Probability mass in log space: log(exp(-u_lo) - exp(-u_hi))
        diff = u_hi - u_lo
        log_mass = -u_lo + np.log(-np.expm1(-np.where(np.isinf(diff), np.inf, diff)))

        # Numerator  Gamma(a) * [P(a,u_hi) - P(a,u_lo)]  -- choose representation
        upper_region = u_lo > a  # deep in the upper tail -> use Q (gammaincc)
        num_reg = np.where(
            upper_region,
            gammaincc(a, u_lo) - gammaincc(a, u_hi),
            gammainc(a, u_hi) - gammainc(a, u_lo),
        )
        log_num = gammaln(a) + np.log(num_reg)
        mean = scale * np.exp(log_num - log_mass)

    bad = ~np.isfinite(mean) | (num_reg <= 0)
    bad &= ~degenerate
    if np.any(bad):
        # Asymptotics.
        #  * upper tail (u_lo large, hi = inf):  E ~ lo * (1 + 1/(k*u_lo))
        #  * lower tail (u_hi tiny, lo = 0):     density ~ x^(k-1), E ~ hi * k/(k+1)
        #  * otherwise: midpoint of the (very narrow in probability) interval
        fb = np.where(
            np.isinf(hi),
            lo * (1.0 + 1.0 / (shape * np.maximum(u_lo, 1e-300))),
            np.where(lo <= 0, hi * shape / (shape + 1.0), 0.5 * (lo + hi)),
        )
        mean = np.where(bad, fb, mean)

    out[:] = mean
    out[degenerate] = lo[degenerate]
    lo_eff = np.where(lo > 0, lo, 0.0)
    return _clip(out, lo_eff, hi)


def weibull_trunc_sample(shape, scale, lo, hi, u):
    """
    Exact draw from Weibull(shape, scale) truncated to [lo, hi].

    With Y = (X/scale)^shape ~ Exp(1) truncated to [u_lo, u_hi]:
        Y = u_lo - log(1 - V * (1 - exp(-(u_hi - u_lo)))),   V ~ U(0, 1)
    which is exact even far in the tails (memorylessness of the exponential).
    """
    shape, scale, lo, hi, v = _as_float_arrays(shape, scale, lo, hi, u)
    u_lo = _weibull_u(lo, shape, scale)
    u_hi = _weibull_u(hi, shape, scale)
    degenerate = ~(u_lo < u_hi)

    with np.errstate(over="ignore", invalid="ignore"):
        width = -np.expm1(-(u_hi - u_lo))  # 1 - exp(-(u_hi-u_lo)), = 1 when hi = inf
        y = u_lo - np.log1p(-v * width)
        out = scale * y ** (1.0 / shape)
    out[degenerate] = lo[degenerate]
    lo_eff = np.where(lo > 0, lo, 0.0)
    return _clip(out, lo_eff, hi)


def _clip(x, lo, hi):
    lo = np.broadcast_to(lo, x.shape)
    hi = np.broadcast_to(hi, x.shape)
    return np.minimum(np.maximum(x, lo), hi)
