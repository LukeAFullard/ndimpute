# Validation 10: Auto-Detect API

This validation suite stresses the "Auto-Detect" capabilities of `ndimpute.impute()`, which allows users to pass string-formatted data (e.g., `["<0.5", "1.2"]`) and have the library automatically handle parsing, status inference, and distribution selection.

## Test Cases

1.  **Simple Left Censoring**: Input strings like `<1`. Verifies boolean status and value imputation.
2.  **Mixed Censoring**: Input strings like `<1` and `>5`. Verifies mixed status codes (-1, 0, 1) and correct imputation ranges.
3.  **Auto Distribution Selection**:
    *   Generates Lognormal data -> Verifies `dist='auto'` selects `lognormal`.
    *   Generates Normal data (with negatives) -> Verifies `dist='auto'` selects `normal`.
4.  **Custom Markers**: Verifies support for custom strings like `ND` (Non-Detect) or `E` (Estimated/Exceeds) via `left_marker` and `right_marker`.

## Usage

```bash
python validate_auto.py
```
