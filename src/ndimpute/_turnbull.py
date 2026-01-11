import numpy as np

def turnbull_em(left, right, max_iter=1000, tol=1e-5):
    """
    Computes the Non-Parametric Maximum Likelihood Estimator (NPMLE)
    for interval-censored data using the Turnbull EM algorithm.

    Args:
        left (array): Lower bounds of intervals.
        right (array): Upper bounds of intervals.
                       Use np.inf for right-censored (L, inf).
                       Use L for exact observations (L, L).

    Returns:
        tuple: (intervals, probs)
            intervals: (M, 2) array of equivalence classes [start, end].
            probs: (M,) array of probability mass assigned to each interval.
    """
    left = np.array(left)
    right = np.array(right)
    n = len(left)

    # 1. Determine Equivalence Intervals
    # Collect all unique endpoints
    endpoints = np.unique(np.concatenate([left, right]))
    endpoints = endpoints[~np.isinf(endpoints)] # Remove infinity
    endpoints.sort()

    if len(endpoints) == 0:
        return np.array([]), np.array([])

    # If we have only 1 unique endpoint (e.g., all data is exactly x),
    # we need to handle it.
    if len(endpoints) == 1:
        e = endpoints[0]
        intervals = np.array([[e, e]])
        probs = np.array([1.0])
        return intervals, probs

    # Generate Candidates
    # 1. Singleton [e_i, e_i] for each endpoint
    # 2. Open Interval (e_i, e_{i+1}) for each gap

    candidates = []
    for i in range(len(endpoints)):
        # Singleton
        candidates.append([endpoints[i], endpoints[i]])

        if i < len(endpoints) - 1:
            # Interval between
            # We represent (e_i, e_{i+1}) as [e_i, e_{i+1}] for storage.
            # We rely on 'is_singleton' check below to distinguish from closed singletons.
            candidates.append([endpoints[i], endpoints[i+1]])

    valid_intervals = []
    alpha_list = []

    for start, end in candidates:
        is_singleton = (start == end)

        if is_singleton:
            # Interval is [start, start]
            # Covered if L <= start AND start <= R
            covered = (left <= start) & (right >= start)
        else:
            # Interval is (start, end)
            # Covered if [L, R] contains the open interval (start, end).
            # i.e. L <= start and end <= R.
            # Standard Turnbull logic: candidate must be subset of observation.
            covered = (left <= start) & (right >= end)

        if np.any(covered):
            valid_intervals.append([start, end])
            alpha_list.append(covered)

    if not valid_intervals:
        return np.array([]), np.array([])

    intervals = np.array(valid_intervals)
    alpha = np.column_stack(alpha_list).astype(float) # (N, M)
    m = len(intervals)

    # 2. EM Algorithm (Self-Consistency)
    # Initialize probabilities uniform
    p = np.ones(m) / m

    for iteration in range(max_iter):
        p_prev = p.copy()

        # E-step
        denom = alpha @ p # (N,)
        denom[denom == 0] = 1e-100 # Safety

        # Contribution
        contrib = alpha * p[None, :]
        contrib = contrib / denom[:, None]

        # M-step
        p = np.sum(contrib, axis=0) / n

        if np.max(np.abs(p - p_prev)) < tol:
            break

    return intervals, p

def predict_turnbull(intervals, probs, times):
    """
    Calculates Survival Probability S(t) = P(T > t) from Turnbull estimates.

    Args:
        intervals (array): Intervals from Turnbull EM.
        probs (array): Probability mass for each interval.
        times (array): Time points to evaluate S(t).

    Returns:
        array: Survival probabilities.
    """
    times = np.atleast_1d(times)
    surv = np.zeros_like(times, dtype=float)

    starts = intervals[:, 0]
    ends = intervals[:, 1]

    # Intervals are either [s, s] or (s, e).
    # Logic:
    # 1. Singleton [s, s]: Include mass if s > t.
    # 2. Open Interval (s, e): All values > s. So if s >= t, entire interval > t.

    is_singleton = (starts == ends)

    for i, t in enumerate(times):
        # Case 1: Singletons. Include if start > t.
        mask_single = is_singleton & (starts > t)

        # Case 2: Open Intervals (start, end). Include if start >= t.
        # because if start == t, interval is (t, end), all values > t.
        mask_open = (~is_singleton) & (starts >= t)

        # Combine
        mask = mask_single | mask_open

        surv[i] = np.sum(probs[mask])

    return surv
