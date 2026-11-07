"""
Turnbull (1976) non-parametric maximum likelihood estimator (NPMLE) for
interval-censored data.

Each observation i is a closed interval [L_i, R_i] (L_i == R_i for exact
values, L_i = -inf for left-censored, R_i = inf for right-censored).

The NPMLE places all probability mass on the *innermost intervals* (Peto 1973;
Turnbull 1976): intervals [l, r] where l is a left endpoint and r is the next
endpoint in sorted order, which is a right endpoint, with no other endpoint in
between. For closed intervals, ties are ordered left-endpoints-first so that
exact observations produce singleton innermost intervals [x, x]. Masses are
found with the self-consistency (EM) algorithm.

For right-censored-only data this reproduces the Kaplan-Meier estimator, and
for left-censored-only data the reverse Kaplan-Meier estimator.
"""

import numpy as np
from scipy import sparse


def innermost_intervals(left, right):
    """
    Turnbull innermost intervals for closed intervals [left_i, right_i].

    Returns:
        ndarray: (M, 2) array of [start, end].
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    pts = np.concatenate([left, right])
    kind = np.concatenate([np.zeros(len(left)), np.ones(len(right))])  # 0 = L, 1 = R
    # Sort by value, then L before R at ties (closed intervals).
    order = np.lexsort((kind, pts))
    pts, kind = pts[order], kind[order]
    is_pair = (kind[:-1] == 0) & (kind[1:] == 1)
    starts = pts[:-1][is_pair]
    ends = pts[1:][is_pair]
    out = np.unique(np.column_stack([starts, ends]), axis=0)
    return out


def turnbull_em(left, right, max_iter=10000, tol=1e-10):
    """
    NPMLE for interval-censored data via the Turnbull self-consistency algorithm.

    Args:
        left (array): Lower bounds (use -inf for left-censored).
        right (array): Upper bounds (use inf for right-censored; = left for exact).
        max_iter (int): Maximum EM iterations.
        tol (float): Convergence tolerance on the max absolute change in mass.

    Returns:
        tuple: (intervals, probs)
            intervals: (M, 2) array of innermost intervals [start, end].
            probs: (M,) probability mass assigned to each interval (sums to 1).
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    n = len(left)
    if n == 0:
        return np.empty((0, 2)), np.empty(0)
    if np.any(left > right):
        raise ValueError("Interval lower bounds must not exceed upper bounds.")

    intervals = innermost_intervals(left, right)
    m = len(intervals)

    # alpha[i, j] = 1 if innermost interval j is contained in observation i.
    s, e = intervals[:, 0], intervals[:, 1]
    rows, cols = [], []
    order_s = np.argsort(s)
    s_sorted = s[order_s]
    for i in range(n):
        # Candidates with start >= left_i, then require end <= right_i
        k0 = np.searchsorted(s_sorted, left[i], side='left')
        cand = order_s[k0:]
        cand = cand[e[cand] <= right[i]]
        rows.append(np.full(len(cand), i))
        cols.append(cand)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    alpha = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, m))

    if np.any(np.diff(alpha.indptr) == 0):
        raise ValueError("Internal error: an observation contains no innermost interval.")

    p = np.full(m, 1.0 / m)
    alpha_t = alpha.T.tocsr()
    for _ in range(max_iter):
        denom = alpha @ p                     # P(observation i)
        p_new = p * (alpha_t @ (1.0 / denom)) / n
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new

    p = p / p.sum()
    return intervals, p


def predict_turnbull(intervals, probs, times):
    """
    Survival probability S(t) = P(T > t) from Turnbull estimates.

    Within a non-degenerate innermost interval the NPMLE does not specify how
    the mass is distributed. Mass on (s, e] with s >= t is counted as > t,
    mass on a singleton [s, s] is counted when s > t; i.e. S(t) is the lower
    envelope of the NPMLE survival curve.
    """
    times = np.atleast_1d(np.asarray(times, dtype=float))
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    is_singleton = starts == ends
    surv = np.empty(len(times))
    for i, t in enumerate(times):
        mask = (is_singleton & (starts > t)) | (~is_singleton & (starts >= t))
        surv[i] = np.sum(probs[mask])
    return surv
