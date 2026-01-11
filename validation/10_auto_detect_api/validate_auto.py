import numpy as np
import pandas as pd
import pytest
import sys
import os

# Ensure we import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ndimpute import impute

def test_auto_detect_simple_left():
    """Test standard left censoring string input."""
    # Data: <1, 2, 3, <1, 4
    # Should imply left censoring.
    data = ["<1", "2", "3", "<1", "4"]
    df = impute(data, dist='normal', random_state=42)

    # If inferred type is Left, status is boolean (True for censored)
    # If inferred type is Mixed, status is int (-1 for left)
    # The detect_and_parse returns 'left' if only < found.
    # api.py sets status to bool if type is left.

    assert df['censoring_status'].iloc[0] == True
    assert df['censoring_status'].iloc[1] == False
    assert df['imputed_value'].iloc[0] <= 1.0 # Should be imputed
    assert df['imputed_value'].iloc[1] == 2.0

    # Check that it actually ran ROS
    # Imputed values for <1 should be <= 1 (can be equal to limit if capped)
    assert (df.loc[df['censoring_status'] == True, 'imputed_value'] <= 1.0).all()

def test_auto_detect_mixed():
    """Test mixed censoring string input (< and >)."""
    # Data: <1, 2, 3, >5
    data = ["<1", "2", "3", ">5"]
    df = impute(data, dist='normal', random_state=42)

    assert df['censoring_status'].iloc[0] == -1 # Left
    assert df['censoring_status'].iloc[3] == 1  # Right

    # Check values
    assert df['imputed_value'].iloc[0] <= 1.0
    assert df['imputed_value'].iloc[3] >= 5.0

def test_auto_dist_selection():
    """Test dist='auto' picks correct distribution."""
    np.random.seed(42)

    # 1. Generate Lognormal Data
    true_vals = np.random.lognormal(mean=2, sigma=0.5, size=50)
    # Censor some
    limit = np.percentile(true_vals, 30)
    str_data = []
    for v in true_vals:
        if v < limit:
            str_data.append(f"<{limit:.2f}")
        else:
            str_data.append(f"{v:.2f}")

    df = impute(str_data, dist='auto', random_state=42)

    # Should prefer lognormal (fit score)
    assert df.attrs['best_dist'] == 'lognormal'
    assert 'fit_score' in df.attrs

    # 2. Generate Normal Data (with negatives to force normal)
    true_vals_norm = np.random.normal(0, 1, size=50)
    limit_norm = -0.5
    str_data_norm = []
    for v in true_vals_norm:
        if v < limit_norm:
            str_data_norm.append(f"<{limit_norm:.2f}")
        else:
            str_data_norm.append(f"{v:.2f}")

    df_norm = impute(str_data_norm, dist='auto', random_state=42)
    assert df_norm.attrs['best_dist'] == 'normal'

def test_custom_markers_auto():
    """Test auto-detect with custom markers (e.g. ND for left, E for right)."""
    # Requires passing markers to detect_and_parse via kwargs?
    # api.py passes kwargs to detect_and_parse?
    # Yes: l_marker = kwargs.get('left_marker', '<')

    data = ["ND10", "20", "30", "E50"]
    # ND10 -> Left Censored < 10
    # E50 -> Right Censored > 50

    df = impute(data, left_marker="ND", right_marker="E", dist='normal')

    assert df['censoring_status'].iloc[0] == -1
    assert df['original_value'].iloc[0] == 10.0
    assert df['censoring_status'].iloc[3] == 1
    assert df['original_value'].iloc[3] == 50.0

if __name__ == "__main__":
    # Manually run if executed as script
    try:
        test_auto_detect_simple_left()
        print("test_auto_detect_simple_left PASSED")
        test_auto_detect_mixed()
        print("test_auto_detect_mixed PASSED")
        test_auto_dist_selection()
        print("test_auto_dist_selection PASSED")
        test_custom_markers_auto()
        print("test_custom_markers_auto PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
