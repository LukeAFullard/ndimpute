import numpy as np
import pandas as pd

def impute_sub_left(values, is_censored, strategy='half', multiplier=None):
    """
    Imputes left-censored data using simple substitution.

    Args:
        values (array): Observed values (LOD for censored).
        is_censored (bool array): True if value is censored (<).
        strategy (str):
            - 'half': Replace <LOD with LOD/2 (default).
            - 'zero': Replace <LOD with 0.
            - 'value' or 'lod': Replace <LOD with LOD.
            - 'multiple': Replace <LOD with LOD * multiplier.
        multiplier (float, optional): Factor to multiply by when strategy='multiple'.
    """
    values = np.array(values, dtype=float)
    is_censored = np.array(is_censored, dtype=bool)

    imputed = values.copy()
    cens_vals = values[is_censored]

    if strategy == 'half':
        imputed[is_censored] = cens_vals / 2.0
    elif strategy == 'zero':
        imputed[is_censored] = 0.0
    elif strategy in ['value', 'lod']:
        imputed[is_censored] = cens_vals
    elif strategy == 'multiple':
        if multiplier is None:
            raise ValueError("Must provide 'multiplier' argument when strategy='multiple'.")
        imputed[is_censored] = cens_vals * multiplier
    else:
        raise ValueError(f"Unknown strategy '{strategy}' for left censoring substitution. Options: 'half', 'zero', 'value', 'multiple'.")

    return imputed

def impute_sub_right(values, is_censored, strategy='value', multiplier=None):
    """
    Imputes right-censored data using simple substitution.

    Args:
        values (array): Observed values (Censoring time C).
        is_censored (bool array): True if value is censored (>).
        strategy (str):
            - 'value' or 'c': Replace >C with C.
            - 'multiple': Replace >C with C * multiplier.
        multiplier (float, optional): Factor to multiply by when strategy='multiple'.
    """
    values = np.array(values, dtype=float)
    is_censored = np.array(is_censored, dtype=bool)

    imputed = values.copy()
    cens_vals = values[is_censored]

    if strategy in ['value', 'c']:
        imputed[is_censored] = cens_vals
    elif strategy == 'multiple':
        if multiplier is None:
            raise ValueError("Must provide 'multiplier' argument when strategy='multiple'.")
        imputed[is_censored] = cens_vals * multiplier
    else:
        raise ValueError(f"Unknown strategy '{strategy}' for right censoring substitution. Options: 'value', 'multiple'.")

    return imputed

def impute_sub_mixed(values, status, left_kwargs=None, right_kwargs=None):
    """
    Imputes mixed-censored data using substitution.

    Args:
        values (array): Data values.
        status (array): Status codes (-1: Left, 0: Obs, 1: Right).
        left_kwargs (dict): Arguments for left substitution (strategy, multiplier).
        right_kwargs (dict): Arguments for right substitution (strategy, multiplier).
    """
    values = np.array(values, dtype=float)
    status = np.array(status, dtype=int)

    if left_kwargs is None: left_kwargs = {}
    if right_kwargs is None: right_kwargs = {}

    imputed = values.copy()

    # Left Censored
    mask_left = (status == -1)
    if np.any(mask_left):
        # We pass only the left-censored subset to the helper.
        # impute_sub_left returns a full array with imputations.
        left_imputed = impute_sub_left(values, mask_left, **left_kwargs)
        imputed[mask_left] = left_imputed[mask_left]

    # Right Censored
    mask_right = (status == 1)
    if np.any(mask_right):
        right_imputed = impute_sub_right(values, mask_right, **right_kwargs)
        imputed[mask_right] = right_imputed[mask_right]

    return imputed
