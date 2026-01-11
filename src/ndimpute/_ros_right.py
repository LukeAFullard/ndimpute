from ._ros_left import impute_ros_left

def impute_ros_right(values, is_censored, dist='lognormal', plotting_position='kaplan-meier', return_fit=False, **kwargs):
    """
    Imputes right-censored data using Reverse ROS.

    Args:
        values (array): Observed values (Censoring Limit for censored).
        is_censored (bool array): True if value is censored (>).
        dist (str): Distribution assumption ('lognormal' or 'normal').
        plotting_position (str): Method for calculating plotting positions.
        return_fit (bool): If True, returns (imputed_values, r_squared).
        **kwargs:
            - impute_type (str): 'stochastic' (default) or 'mean'.
            - other kwargs passed to impute_ros_left.
    """
    # 1. Reverse domain
    # For lognormal (dist>0), we can't just flip sign and log.
    # Instead, we invert: y' = 1/y.
    # Large values become small (Left Censored).

    if dist == 'lognormal':
        # Right censored at 100 -> Value is > 100
        # Inverted: Value is < 1/100 (Left Censored)

        # Ensure no zeros if lognormal
        if (values <= 0).any():
             raise ValueError("Values must be positive for lognormal distribution.")

        inv_values = 1.0 / values

        # Call Left ROS
        result = impute_ros_left(inv_values, is_censored, dist='lognormal', plotting_position=plotting_position, return_fit=return_fit, **kwargs)

        if return_fit:
            imputed_inv, r2 = result
            return 1.0 / imputed_inv, r2
        else:
            return 1.0 / result

    else:
        # Normal distribution -> flip sign
        flipped_values = -values
        result = impute_ros_left(flipped_values, is_censored, dist='normal', plotting_position=plotting_position, return_fit=return_fit, **kwargs)

        if return_fit:
            imputed_flipped, r2 = result
            return -imputed_flipped, r2
        else:
            return -result
