import re

import numpy as np

_MISSING_STRINGS = {"", "na", "nan", "n/a", "none", "null"}


def parse_censored(data, left_marker='<', right_marker='>'):
    """
    Parse values that may carry censoring markers.

    Supported formats (whitespace around the marker is allowed):
        "{left_marker}0.5"  -> left-censored at 0.5   (status -1)
        "{right_marker}10"  -> right-censored at 10   (status  1)
        "5", "5.", ".5", "1e-3", 5.0 -> observed      (status  0)
        None, NaN, "", "NA", "NaN" -> missing (value NaN, status 0)

    Args:
        data (iterable): Strings and/or numbers.
        left_marker (str): Marker for left censoring (default '<').
        right_marker (str): Marker for right censoring (default '>').

    Returns:
        tuple: (values, status, has_left, has_right)
            values: float array (NaN for missing).
            status: int array, -1 left / 0 observed / 1 right.
    """
    if left_marker == right_marker:
        raise ValueError("left_marker and right_marker must differ.")
    l_esc = re.escape(left_marker)
    r_esc = re.escape(right_marker)
    number = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    pattern = re.compile(rf"^\s*({l_esc}|{r_esc})?\s*({number})\s*$")

    data = list(np.atleast_1d(np.asarray(data, dtype=object)).ravel())
    values = np.empty(len(data), dtype=float)
    status = np.zeros(len(data), dtype=int)

    for i, item in enumerate(data):
        if item is None:
            values[i] = np.nan
            continue
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"Boolean value {item!r} found in values; pass censoring flags via `status`.")
        if isinstance(item, (int, float, np.integer, np.floating)):
            values[i] = float(item)
            continue

        s_item = str(item).strip()
        if s_item.lower() in _MISSING_STRINGS:
            values[i] = np.nan
            continue
        match = pattern.match(s_item)
        if not match:
            raise ValueError(f"Could not parse value: '{item}'. Expected format like "
                             f"'{left_marker}0.5', '{right_marker}10', or '5.0'.")
        indicator, val_str = match.group(1), match.group(2)
        values[i] = float(val_str)
        if indicator == left_marker:
            status[i] = -1
        elif indicator == right_marker:
            status[i] = 1

    return values, status, bool(np.any(status == -1)), bool(np.any(status == 1))


def detect_and_parse(data, left_marker='<', right_marker='>'):
    """
    Parse a list of values (strings/floats) and infer the censoring type.

    Returns:
        tuple: (values, status, censoring_type)
            values: Float array of limits/values.
            status: Int array (-1/0/1) for 'mixed'; Boolean array (True =
                censored in the direction of the type) for 'left' / 'right'.
            censoring_type: 'left', 'right', or 'mixed'. Data with no markers
                are reported as 'left' with no censored values.
    """
    values, status, has_left, has_right = parse_censored(data, left_marker, right_marker)
    if has_left and has_right:
        return values, status, 'mixed'
    if has_right:
        return values, status == 1, 'right'
    return values, status == -1, 'left'
