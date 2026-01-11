import numpy as np
import pytest
from ndimpute._preprocess import detect_and_parse

def test_parse_left_censored():
    raw = ["<0.5", "2.0", "< 0.5", "5"]
    vals, status, ctype = detect_and_parse(raw)

    assert ctype == 'left'
    np.testing.assert_array_equal(vals, [0.5, 2.0, 0.5, 5.0])
    np.testing.assert_array_equal(status, [True, False, True, False])

def test_parse_right_censored():
    raw = ["100", "> 120", "150", ">200"]
    vals, status, ctype = detect_and_parse(raw)

    assert ctype == 'right'
    np.testing.assert_array_equal(vals, [100.0, 120.0, 150.0, 200.0])
    np.testing.assert_array_equal(status, [False, True, False, True])

def test_parse_mixed_censored():
    raw = ["<10", "50", ">100", "50"]
    vals, status, ctype = detect_and_parse(raw)

    assert ctype == 'mixed'
    np.testing.assert_array_equal(vals, [10.0, 50.0, 100.0, 50.0])
    # Mixed status: -1 (Left), 0 (Obs), 1 (Right)
    np.testing.assert_array_equal(status, [-1, 0, 1, 0])

def test_no_censoring():
    raw = ["10", "20", "30"]
    vals, status, ctype = detect_and_parse(raw)

    # Logic: If no markers found, assume observed.
    # Status should be all False/0.
    np.testing.assert_array_equal(vals, [10.0, 20.0, 30.0])
    assert not np.any(status)

def test_mixed_types_input():
    # Input has float and string
    raw = [0.5, "<1.0", 2.0]
    vals, status, ctype = detect_and_parse(raw)

    assert ctype == 'left'
    np.testing.assert_array_equal(vals, [0.5, 1.0, 2.0])
    np.testing.assert_array_equal(status, [False, True, False])

def test_invalid_string():
    raw = ["abc", "123"]
    with pytest.raises(ValueError):
        detect_and_parse(raw)
