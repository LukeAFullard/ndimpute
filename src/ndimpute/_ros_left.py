import numpy as np
import pandas as pd
from scipy.stats import norm, linregress, ecdf, CensoredData

def impute_ros_left(values, is_censored, dist='lognormal', plotting_position='kaplan-meier', return_fit=False, **kwargs):
    """
    Imputes left-censored data using Robust ROS.

    Args:
        values (array): Observed values (LOD for censored).
        is_censored (bool array): True if value is censored (<).
        dist (str): Distribution assumption ('lognormal' or 'normal').
        plotting_position (str): Method for calculating plotting positions.
            - 'kaplan-meier' (default): Uses Hirsch-Stedinger logic via scipy.stats.ecdf.
              Best for multiple detection limits.
            - 'simple' or 'weibull': Uses simple ranking (rank/(n+1)).
              Matches simple NADA approximations for single limits.
        return_fit (bool): If True, returns (imputed_values, r_squared).
        **kwargs:
            - impute_type (str): 'stochastic' (distribute/random) or 'mean' (default for KM).
              Note: 'simple' plotting_position is inherently 'stochastic' (quantile-based).
              For 'kaplan-meier', default is now 'stochastic' to preserve variance,
              unless 'mean' is explicitly requested.
    """
    values = np.array(values)
    is_censored = np.array(is_censored, dtype=bool)
    n = len(values)

    # Impute type strategy
    # Default to 'stochastic' (quantile spacing) for scientific variance preservation
    impute_type = kwargs.get('impute_type', 'stochastic')
    random_state = kwargs.get('random_state', None)

    # Common Setup: Log Transform if needed for regression Y
    unc_mask = ~is_censored
    y_unc = values[unc_mask]

    if len(y_unc) < 2:
         raise ValueError("Too few uncensored observations to fit regression.")

    if dist == 'lognormal':
        if (values <= 0).any():
             raise ValueError("Values must be positive for lognormal distribution.")
        y_reg = np.log(y_unc)
    elif dist == 'normal':
        y_reg = y_unc
    else:
        raise ValueError(f"Unknown distribution '{dist}'")

    # --- Branch 1: Kaplan-Meier (Hirsch-Stedinger) ---
    if plotting_position in ['kaplan-meier', 'ecdf', 'hirsch-stedinger']:
        neg_values = -values
        unc_data = neg_values[~is_censored]
        cens_data = neg_values[is_censored]

        cd = CensoredData(uncensored=unc_data, right=cens_data)
        res = ecdf(cd)

        # PPs for Uncensored
        pp_unc = res.sf.evaluate(-y_unc)

        # Scaling to avoid 0 and 1
        pp_unc = pp_unc * (n / (n + 1))
        pp_unc[pp_unc == 0] = 0.5 / (n + 1)
        pp_unc[pp_unc == 1] = 1.0 - (0.5 / (n + 1))

        z_unc = norm.ppf(pp_unc)

        # Fit
        slope, intercept, r_value, _, _ = linregress(z_unc, y_reg)
        r_squared = r_value**2

        # Impute
        y_cens = values[is_censored]
        pp_limits = res.sf.evaluate(-y_cens)

        pp_limits = pp_limits * (n / (n + 1))
        # Ensure non-zero to define tail
        pp_limits[pp_limits == 0] = 0.5 / (n + 1)

        z_limits = norm.ppf(pp_limits)

        if impute_type == 'mean':
            # Original Conditional Mean Logic
            numerator = norm.pdf(z_limits)
            denominator = norm.cdf(z_limits)
            z_imputed = -numerator / denominator

        else: # 'stochastic'
            # Two sub-modes for 'stochastic':
            # 1. Deterministic Quantile Spacing (Default if random_state is None).
            #    Spreads points evenly to reconstruct shape.
            # 2. Random Sampling (If random_state is provided).
            #    Samples U[0, p_max] to preserve randomness.

            z_imputed = np.zeros_like(z_limits)

            if random_state is not None:
                # True Stochastic Sampling
                rng = np.random.default_rng(random_state)
                # Sample uniform random numbers for each censored point
                # Range [0, pp_limit]
                # pp_limits is array of P(X < L) for each point
                u_noise = rng.uniform(0, 1, size=len(y_cens))
                p_rand = u_noise * pp_limits

                # Handle edge case where p_rand could be 0
                p_rand = np.clip(p_rand, 1e-15, 1.0 - 1e-15) # Should be < pp_limits anyway

                z_imputed = norm.ppf(p_rand)

            else:
                # Deterministic Quantile Spacing (Original Robust ROS)
                # Distribute censored values in the tail [0, P(X < L)]
                # We group by limit to distribute them evenly in their respective tails.

                # Identify unique limits to handle ties
                # We iterate over unique limit values found in the censored set
                unique_limits = np.unique(y_cens)

                for lim in unique_limits:
                    # Indices in the censored array corresponding to this limit
                    # Note: y_cens is the subset array.
                    idxs = np.where(y_cens == lim)[0]
                    k = len(idxs)

                    if k == 0: continue

                    # Get the max probability for this limit
                    # All these should have same pp_limit
                    p_max = pp_limits[idxs[0]]

                    # Generate spaced probabilities in (0, p_max]
                    # Standard plotting position within the subset:
                    # We want k points.
                    # If we use (i / (k+1)) * p_max, we avoid 0 and p_max.
                    # i ranges 1 to k.
                    ranks_internal = np.arange(1, k + 1)
                    p_sub = (ranks_internal / (k + 1)) * p_max

                    # Map to Z
                    z_sub = norm.ppf(p_sub)

                    # Assign back (order doesn't matter for exchangeable censored values)
                    z_imputed[idxs] = z_sub

        predicted = intercept + slope * z_imputed

    # --- Branch 2: Simple Ranking (Weibull) ---
    elif plotting_position in ['simple', 'weibull']:
        # Sort data to assign ranks
        df = pd.DataFrame({'val': values, 'cens': is_censored})
        df = df.sort_values('val')

        df['rank'] = np.arange(1, n + 1)
        df['pp'] = df['rank'] / (n + 1)
        df['z'] = norm.ppf(df['pp'])

        # Fit on Uncensored
        y_reg_sorted = df.loc[~df['cens'], 'val']
        if dist == 'lognormal':
            y_reg_sorted = np.log(y_reg_sorted)

        x_obs = df.loc[~df['cens'], 'z']

        slope, intercept, r_value, _, _ = linregress(x_obs, y_reg_sorted)
        r_squared = r_value**2

        # Impute
        z_cens = df.loc[df['cens'], 'z']
        predicted_sorted = intercept + slope * z_cens
        predicted = predicted_sorted

    else:
        raise ValueError(f"Unknown plotting_position '{plotting_position}'.")

    # --- Common Finalization ---
    if plotting_position in ['simple', 'weibull']:
        if dist == 'lognormal':
            imputed_vals = np.exp(predicted)
        else:
            imputed_vals = predicted

        df.loc[df['cens'], 'imputed'] = imputed_vals
        limit_vals = df.loc[df['cens'], 'val']
        df.loc[df['cens'], 'imputed'] = np.minimum(df.loc[df['cens'], 'imputed'], limit_vals)

        result = df.sort_index()['val'].copy()
        result[is_censored] = df.sort_index().loc[is_censored, 'imputed']

        if return_fit:
            return result.values, r_squared
        return result.values

    else:
        # Kaplan-Meier path
        if dist == 'lognormal':
            imputed_vals = np.exp(predicted)
        else:
            imputed_vals = predicted

        y_cens = values[is_censored]
        imputed_vals = np.minimum(imputed_vals, y_cens)

        result = values.copy()
        result[is_censored] = imputed_vals

        if return_fit:
            return result, r_squared
        return result
