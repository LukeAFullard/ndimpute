import numpy as np

from ._ros_left import impute_ros_left


def impute_ros_right(values, is_censored, dist='lognormal', plotting_position='kaplan-meier',
                     return_fit=False, **kwargs):
    """
    Impute right-censored data (">C") using "reverse" ROS.

    The data are mapped to a left-censored problem with a transformation that
    preserves the assumed distribution family, ROS is applied, and the result
    is mapped back:

    * lognormal: y = 1/x   (log y = -log x, so lognormality is preserved)
    * normal:    y = -x

    A value right-censored at C becomes left-censored at 1/C (or -C).

    Note: reverse ROS is a symmetric re-use of the left-censoring algorithm.
    For reliability / survival data (heavily right-censored), parametric MLE
    (``method='parametric'``) is usually preferable.

    Args:
        values (array): Observed values, or the censoring limit for censored values.
        is_censored (bool array): True where the value is ">C".
        dist (str): 'lognormal' or 'normal'.
        plotting_position (str): See ``impute_ros_left``.
        return_fit (bool): If True, returns (imputed_values, r_squared).
        **kwargs: impute_type, random_state, clamp -- passed to ``impute_ros_left``.
    """
    values = np.asarray(values, dtype=float)

    if dist == 'lognormal':
        if np.any(values <= 0):
            raise ValueError("Values must be positive for lognormal distribution.")
        fwd = lambda x: 1.0 / x  # noqa: E731
        inv = fwd
    elif dist == 'normal':
        fwd = lambda x: -x  # noqa: E731
        inv = fwd
    else:
        raise ValueError(f"Unknown distribution '{dist}' for ROS. Options: 'lognormal', 'normal'.")

    result = impute_ros_left(fwd(values), is_censored, dist=dist, plotting_position=plotting_position,
                             return_fit=return_fit, **kwargs)
    if return_fit:
        imputed, r2 = result
        out = inv(imputed)
    else:
        out = inv(result)

    # Restore observed values bit-for-bit (avoid 1/(1/x) round-off)
    is_censored = np.asarray(is_censored, dtype=bool)
    out[~is_censored] = values[~is_censored]
    return (out, r2) if return_fit else out
