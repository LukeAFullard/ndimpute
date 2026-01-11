import pandas as pd
import numpy as np
from ._ros_left import impute_ros_left
from ._ros_right import impute_ros_right
from ._ros_mixed import impute_ros_mixed_heuristic
from ._parametric import impute_right_conditional, impute_mixed_parametric
from ._substitution import impute_sub_left, impute_sub_right, impute_sub_mixed
from ._interval import impute_interval_ros
from ._preprocess import detect_and_parse

def impute(values, status=None, method='ros', censoring_type=None, **kwargs):
    """
    Unified imputation function.

    Args:
        values (array-like): The data values.
            - Mixed Strings (e.g. "<0.5", ">10"): Automatically parsed if status is None.
            - Left/Right/Mixed (numeric): 1D array of values.
            - Interval: 2D array of shape (N, 2) representing (Left, Right) bounds.
        status (array-like, optional): Indicator.
            - If None: Automatically inferred from `values` if strings (e.g. "<") are present.
            - Left: True if < LOD.
            - Right: True if censored (> C).
            - Mixed: Integer array (-1: Left, 0: Observed, 1: Right).
        method (str): 'ros', 'parametric', or 'substitution'.
        censoring_type (str, optional): 'left', 'right', 'mixed', or 'interval'.
            - Defaults to 'left' if status is provided but type is not.
            - Defaults to inferred type if status is None.
        **kwargs: Additional arguments (dist, plotting_position, strategy, impute_type, random_state, etc.)

    Returns:
        pd.DataFrame: A dataframe containing:
            - 'imputed_value': The final value (observed or imputed).
            - 'original_value': The input value (or string repr for intervals).
            - 'censoring_status': The original (or inferred) status input.
            - 'is_imputed': Boolean flag.
    """
    # 1. Handle Automatic Preprocessing & Value Extraction
    parsed_type = None

    # Always attempt to parse values if not interval, to handle string inputs (e.g. "<0.5")
    # even if an explicit status is provided.
    if censoring_type != 'interval':
        try:
             l_marker = kwargs.get('left_marker', '<')
             r_marker = kwargs.get('right_marker', '>')
             p_values, p_status, p_type = detect_and_parse(values, left_marker=l_marker, right_marker=r_marker)

             # Update values to numeric form
             values = p_values

             # Only update status if it wasn't provided by the user
             if status is None:
                 status = p_status
                 parsed_type = p_type

        except (ValueError, TypeError):
             # Parsing failed or wasn't needed.
             # If status is None, this may be handled later or treated as uncensored.
             # If status is provided, we fall through to standard processing.
             pass

    # Validate Conflicts with Inferred Type
    if parsed_type is not None:
         l_marker = kwargs.get('left_marker', '<') # Retrieve again for error msg
         r_marker = kwargs.get('right_marker', '>')

         if censoring_type is None:
             censoring_type = parsed_type
         elif censoring_type != parsed_type:
             if (censoring_type == 'left' and parsed_type == 'right') or \
                (censoring_type == 'right' and parsed_type == 'left'):
                 raise ValueError(
                     f"Conflict detected: censoring_type='{censoring_type}' was requested, "
                     f"but data contains '{parsed_type}' censoring indicators (e.g. '{l_marker}' vs '{r_marker}'). "
                     "Please remove 'censoring_type' to use auto-detection, or ensure data matches the requested type."
                 )
             elif censoring_type == 'left' and parsed_type == 'mixed':
                  raise ValueError(
                     f"Conflict detected: censoring_type='{censoring_type}' was requested, "
                     f"but data contains 'mixed' censoring indicators (both '{l_marker}' and '{r_marker}'). "
                  )
             elif censoring_type == 'right' and parsed_type == 'mixed':
                  raise ValueError(
                     f"Conflict detected: censoring_type='{censoring_type}' was requested, "
                     f"but data contains 'mixed' censoring indicators (both '{l_marker}' and '{r_marker}'). "
                  )
             # Allow 'mixed' request for inferred 'left' data as it is a subset.
             pass

    # 2. Default censoring_type logic
    if censoring_type is None:
        # Default to mixed censoring if unspecified.
        censoring_type = 'mixed'

    # Extract common args
    dist = kwargs.get('dist', 'lognormal')
    plotting_position = kwargs.get('plotting_position', 'kaplan-meier')
    impute_type_arg = kwargs.get('impute_type') # None if not present
    random_state = kwargs.get('random_state', None)

    if censoring_type == 'interval':
        # Values should be (N, 2)
        bounds = np.array(values)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("For censoring_type='interval', values must be (N, 2) array of bounds.")

        left, right = bounds[:, 0], bounds[:, 1]

        # Interval Auto-Detect Logic
        fit_score = None
        best_dist = None
        imputed_vals = None

        if method == 'ros':
            it = impute_type_arg if impute_type_arg is not None else 'stochastic'

            if dist == 'auto':
                 # Select best distribution
                 candidates = ['lognormal', 'normal']
                 best_r2 = -1.0

                 for d in candidates:
                     try:
                         # Sanity check for lognormal: bounds must be positive (except 0 if left censored)
                         # _interval.py handles 0 for lognormal internally.
                         curr_vals, curr_r2 = impute_interval_ros(left, right, dist=d, impute_type=it, random_state=random_state, return_fit=True)

                         if curr_r2 > best_r2:
                             best_r2 = curr_r2
                             best_dist = d
                             imputed_vals = curr_vals
                     except Exception:
                         continue

                 if imputed_vals is None:
                     raise ValueError("Auto-distribution selection failed for interval data.")

                 fit_score = best_r2

            else:
                 imputed_vals = impute_interval_ros(left, right, dist=dist, impute_type=it, random_state=random_state)

        else:
            raise NotImplementedError(f"Method '{method}' not implemented for interval censoring.")

        df = pd.DataFrame({
            'imputed_value': imputed_vals,
            'original_left': left,
            'original_right': right,
            'censoring_status': 'interval',
            'is_imputed': True
        })

        if fit_score is not None:
             df.attrs['fit_score'] = fit_score
             df.attrs['best_dist'] = best_dist

        return df

    # ... Existing Logic for Left/Right/Mixed ...
    values = np.array(values)
    if status is None:
        raise ValueError("Status argument is required for left/right/mixed censoring (or provide strings like '<0.5').")

    # Status handling depends on type
    if censoring_type == 'mixed':
        # If status is boolean (True/False), map to Mixed standard (-1/0/1).
        # Assuming boolean input was intended as Left Censoring (most common default).
        # True -> -1 (Left Censored), False -> 0 (Observed).
        status_arr = np.array(status)
        if status_arr.dtype == bool or np.issubdtype(status_arr.dtype, np.bool_):
            status = np.where(status_arr, -1, 0).astype(int)
        else:
            status = status_arr.astype(int)
        is_imputed = (status != 0)
    else:
        status = np.array(status, dtype=bool)
        is_imputed = status

    # Prepare kwargs to propagate
    kwargs_prop = kwargs.copy()
    kwargs_prop.pop('dist', None)
    kwargs_prop.pop('plotting_position', None)

    # --- Auto Distribution Selection ---
    fit_score = None
    best_dist = None

    if dist == 'auto' and method == 'ros':
        # Select best distribution from candidates
        candidates = ['lognormal', 'normal']
        best_r2 = -1.0
        selected_vals = None

        for d in candidates:
            try:
                # Sanity check for lognormal: data must be positive
                if d == 'lognormal':
                    if censoring_type in ['left', 'right'] and np.any(values <= 0):
                        continue
                    # For mixed, values might be < LOD. But if observed values are <= 0, lognormal is invalid.
                    if censoring_type == 'mixed':
                         # Status 0 is observed.
                         observed = values[status == 0]
                         if np.any(observed <= 0):
                             continue

                # Run imputation with fit metric
                curr_vals = None
                curr_r2 = -1.0

                if censoring_type == 'left':
                    curr_vals, curr_r2 = impute_ros_left(values, status, dist=d, plotting_position=plotting_position, return_fit=True, **kwargs_prop)
                elif censoring_type == 'right':
                    curr_vals, curr_r2 = impute_ros_right(values, status, dist=d, plotting_position=plotting_position, return_fit=True, **kwargs_prop)
                elif censoring_type == 'mixed':
                    curr_vals, curr_r2 = impute_ros_mixed_heuristic(values, status, dist=d, plotting_position=plotting_position, return_fit=True, **kwargs_prop)

                # Check fit
                if curr_r2 > best_r2:
                    best_r2 = curr_r2
                    best_dist = d
                    selected_vals = curr_vals

            except Exception:
                # Candidate failed (e.g. lognormal on negative data or regression failure)
                continue

        if selected_vals is None:
            raise ValueError("Auto-distribution selection failed. No valid distribution found for data (or regression failed).")

        dist = best_dist # Update for record
        imputed_vals = selected_vals
        fit_score = best_r2

        # Skip standard dispatch below by marking method as handled
        method = 'DONE_AUTO'

    if method == 'DONE_AUTO':
        pass

    elif censoring_type == 'left':
        if method == 'ros':
            imputed_vals = impute_ros_left(values, status, dist=dist, plotting_position=plotting_position, **kwargs_prop)
        elif method == 'substitution':
            strategy = kwargs.get('strategy', 'half')
            multiplier = kwargs.get('multiplier', None)
            imputed_vals = impute_sub_left(values, status, strategy=strategy, multiplier=multiplier)
        else:
            raise NotImplementedError(f"Method '{method}' not implemented for left censoring.")

    elif censoring_type == 'right':
        if method == 'ros':
            imputed_vals = impute_ros_right(values, status, dist=dist, plotting_position=plotting_position, **kwargs_prop)
        elif method == 'parametric':
            it = impute_type_arg if impute_type_arg is not None else 'mean'
            imputed_vals = impute_right_conditional(values, status, dist=dist, impute_type=it, random_state=random_state)
        elif method == 'substitution':
            strategy = kwargs.get('strategy', 'value')
            multiplier = kwargs.get('multiplier', None)
            imputed_vals = impute_sub_right(values, status, strategy=strategy, multiplier=multiplier)
        else:
            raise ValueError(f"Unknown method '{method}' for right censoring.")

    elif censoring_type == 'mixed':
        if method == 'parametric':
            it = impute_type_arg if impute_type_arg is not None else 'mean'
            imputed_vals = impute_mixed_parametric(values, status, dist=dist, impute_type=it, random_state=random_state)
        elif method == 'substitution':
            # Extract mixed kwargs
            left_kwargs = {
                'strategy': kwargs.get('left_strategy', 'half'),
                'multiplier': kwargs.get('left_multiplier', None)
            }
            right_kwargs = {
                'strategy': kwargs.get('right_strategy', 'value'),
                'multiplier': kwargs.get('right_multiplier', None)
            }
            imputed_vals = impute_sub_mixed(values, status, left_kwargs=left_kwargs, right_kwargs=right_kwargs)
        elif method == 'ros':
            import warnings
            warnings.warn(
                "Mixed ROS (heuristics/interval) can be biased for mixed censoring. "
                "Validation suggests using method='parametric' for higher accuracy.",
                UserWarning
            )
            imputed_vals = impute_ros_mixed_heuristic(values, status, dist=dist, plotting_position=plotting_position, **kwargs_prop)
        else:
            raise ValueError(f"Unknown method '{method}' for mixed censoring.")

    else:
        raise ValueError("censoring_type must be 'left', 'right', 'mixed', or 'interval'")

    df = pd.DataFrame({
        'imputed_value': imputed_vals,
        'original_value': values,
        'censoring_status': status,
        'is_imputed': is_imputed
    })

    if fit_score is not None:
        df.attrs['fit_score'] = fit_score
        df.attrs['best_dist'] = best_dist

    return df
