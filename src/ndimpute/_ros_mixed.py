import numpy as np
from ._ros_left import impute_ros_left
from ._ros_right import impute_ros_right
from ._interval import impute_interval_ros
import warnings

def impute_ros_mixed_heuristic(values, status, return_fit=False, **kwargs):
    """
    Imputes mixed-censored data using a rigorous Interval Imputation approach.
    (Formerly implemented as a sequential heuristic, now upgraded to Interval/Turnbull).

    Strategy:
    Convert mixed censoring to interval censoring:
    - Left Censored (L): Interval (-inf, L] (or [0, L] for lognormal)
    - Right Censored (R): Interval [R, inf)
    - Observed (O): Interval [O, O]

    Then apply Interval ROS (Turnbull Estimator + Regression).

    Args:
        values (array): Data values.
        status (array): Status codes (-1: Left, 0: Obs, 1: Right).
        return_fit (bool): If True, returns (imputed_values, r_squared).
        **kwargs: Arguments passed (dist, plotting_position).

    Returns:
        array: Imputed values.
    """
    values = np.array(values, dtype=float)
    status = np.array(status, dtype=int)
    dist = kwargs.get('dist', 'lognormal')

    n = len(values)
    left_bounds = np.zeros(n)
    right_bounds = np.zeros(n)

    # 1. Convert to Interval Format
    for i in range(n):
        s = status[i]
        v = values[i]

        if s == 0: # Observed
            left_bounds[i] = v
            right_bounds[i] = v
        elif s == -1: # Left Censored (< v)
            if dist == 'lognormal':
                left_bounds[i] = 0.0
            else:
                left_bounds[i] = -np.inf
            right_bounds[i] = v
        elif s == 1: # Right Censored (> v)
            left_bounds[i] = v
            right_bounds[i] = np.inf
        else:
            raise ValueError(f"Unknown status code {s}")

    # 2. Apply Interval ROS
    # Propagate impute_type (default 'stochastic') and random_state
    impute_type = kwargs.get('impute_type', 'stochastic')
    random_state = kwargs.get('random_state', None)

    try:
        result = impute_interval_ros(left_bounds, right_bounds, dist=dist, impute_type=impute_type, random_state=random_state, return_fit=return_fit)
        return result
    except Exception as e:
        # Check for specific failure modes we might want to handle silently or with specific advice
        msg = str(e)

        # Fallback to legacy heuristic if Interval fails (e.g. convergence issues)
        # We generally warn, unless it's a known edge case where fallback is standard.
        warnings.warn(f"Interval ROS failed for Mixed Censoring: {msg}. Falling back to sequential heuristic.")

        return _impute_ros_mixed_legacy(values, status, return_fit=return_fit, **kwargs)


def _impute_ros_mixed_legacy(values, status, return_fit=False, **kwargs):
    """
    Legacy sequential heuristic (Pass 1 Left, Pass 2 Right).
    Used as fallback.

    Args:
        values (array): Data values.
        status (array): Status codes.
        return_fit (bool): If True, returns (imputed_values, r_squared).
        **kwargs: Additional arguments.

    Returns:
        array or tuple: Imputed values, or (imputed_values, r_squared) if return_fit=True.
    """
    # --- Pass 1: Impute Left ---
    mask_left = (status == -1)
    pass1_values = impute_ros_left(values, mask_left, **kwargs)

    # --- Pass 2: Impute Right ---
    mask_right = (status == 1)
    final_values = impute_ros_right(pass1_values, mask_right, **kwargs)

    if return_fit:
        return final_values, np.nan
    return final_values
