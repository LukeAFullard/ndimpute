# Validation Report: Auto-Detect API

**Date:** 2025-02-12
**Status:** PASSED

## Summary

The Auto-Detect API was validated to ensure it correctly infers censoring types, parses string inputs, and selects the optimal distribution.

## Results

| Test Case | Description | Result |
| :--- | :--- | :--- |
| **Simple Left** | Standard `<LOD` inputs. Correctly inferred as Left Censoring. | PASS |
| **Mixed** | Both `<L` and `>R` inputs. Correctly inferred as Mixed Censoring. | PASS |
| **Auto Dist** | Correctly identified 'lognormal' for lognormal data and 'normal' for data with negatives. | PASS |
| **Custom Markers** | Correctly parsed `ND` and `E` prefixes. | PASS |

## Conclusion

The `impute()` function's auto-detection logic is robust and ready for primary use in documentation.
