import numpy as np
import re

def detect_and_parse(data, left_marker='<', right_marker='>'):
    """
    Parses a list of values (strings/floats) to detect censoring indicators.

    Supported formats:
        "{left_marker} Value" -> Left Censored
        "{right_marker} Value" -> Right Censored
        "Value"               -> Observed

    Args:
        data (list or array): Input data containing mixed types.
        left_marker (str): String indicating left censoring (default '<').
        right_marker (str): String indicating right censoring (default '>').

    Returns:
        tuple: (values, status, censoring_type)
            values: Float array of limits/values.
            status: Boolean array (Left/Right) or Int array (Mixed).
            censoring_type: 'left', 'right', or 'mixed'.
    """
    values = []
    # Temporary status storage: 0=Obs, -1=Left, 1=Right
    temp_status = []

    # Regex patterns
    # Build regex dynamically based on markers
    # Escape markers to handle special regex characters
    l_esc = re.escape(left_marker)
    r_esc = re.escape(right_marker)

    # Pattern: ^ [whitespace] (left|right)? [whitespace] (number) $
    # We capture the marker group to check which one matched
    pattern_str = f"^\\s*({l_esc}|{r_esc})?\\s*([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\\s*$"
    pattern = re.compile(pattern_str)

    has_left = False
    has_right = False

    for item in data:
        # If already a number, treat as observed
        if isinstance(item, (int, float, np.number)):
            values.append(float(item))
            temp_status.append(0)
            continue

        s_item = str(item).strip()
        match = pattern.match(s_item)

        if not match:
            # Try to handle case where marker is text like "ND" which might be separated by space
            # The regex handles optional space.
            raise ValueError(f"Could not parse value: '{item}'. Expected format like '{left_marker}0.5', '{right_marker}10', or '5.0'.")

        indicator = match.group(1) # marker or None
        val_str = match.group(2)
        val = float(val_str)

        values.append(val)

        if indicator == left_marker:
            temp_status.append(-1)
            has_left = True
        elif indicator == right_marker:
            temp_status.append(1)
            has_right = True
        else:
            temp_status.append(0)

    values = np.array(values, dtype=float)

    # Determine Type
    if has_left and has_right:
        censoring_type = 'mixed'
        status = np.array(temp_status, dtype=int)
    elif has_right:
        censoring_type = 'right'
        # For Right censoring, status is Boolean True if censored
        # temp_status has 1 for Right. 0 for Obs.
        status = (np.array(temp_status) == 1)
    else:
        # Default to 'left' if only left or no censoring found (Env standard)
        censoring_type = 'left'
        # temp_status has -1 for Left.
        status = (np.array(temp_status) == -1)

    return values, status, censoring_type
