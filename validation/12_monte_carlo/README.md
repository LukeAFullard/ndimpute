# Validation 12: Monte Carlo bias and RMSE

`run_mc.py` draws data from a lognormal(μ = 0, σ = 1) population, censors them, imputes with each method, and compares the completed data's mean of logs (true 0), SD of logs (true 1) and arithmetic mean (true $e^{0.5}$ = 1.649) with the truth.

Scenarios:

* **Left censoring:** n = 20, 50, 200; a single detection limit, or three limits assigned at random; about 30 % or 60 % censored.
* **Mixed censoring:** n = 30, 100; "<0.5" and ">4".

1,000 replicates per scenario (`python run_mc.py 1000`); results are in `mc_results.csv` and `mc_output.txt`.

## Summary

See [`SUMMARY.md`](SUMMARY.md).
