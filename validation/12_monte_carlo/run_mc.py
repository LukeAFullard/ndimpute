"""
Monte Carlo evaluation of ndimpute's estimators (validation case 12).

For each scenario, data are drawn from a known lognormal(mu=0, sigma=1)
population, censored, imputed, and the summary statistics of the completed
data are compared with the true population values:

    mean of log values (true 0), SD of log values (true 1),
    arithmetic mean (true exp(0.5) = 1.6487).

Reported: bias and RMSE over replicates. Run:  python run_mc.py [n_reps]
"""
import sys
import warnings

import numpy as np
import pandas as pd

from ndimpute import impute

warnings.simplefilter("ignore")
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
TRUE = {"mean_log": 0.0, "sd_log": 1.0, "mean": float(np.exp(0.5))}


def left_scenario(rng, n, multi, cens_q):
    x = rng.lognormal(0, 1, n)
    if multi:
        lims = np.exp(np.array([-1.0, 0.0, 0.5]) + (cens_q - 0.4) * 2)
        L = rng.choice(lims, n)
    else:
        L = np.full(n, np.exp(np.quantile(rng.normal(size=100000), cens_q)))
    c = x < L
    return np.where(c, L, x), c


def mixed_scenario(rng, n):
    x = rng.lognormal(0, 1, n)
    st = np.where(x < 0.5, -1, np.where(x > 4.0, 1, 0))
    return np.where(st == -1, 0.5, np.where(st == 1, 4.0, x)), st


LEFT_METHODS = {
    "ROS (HS, clamp)": dict(method="ros"),
    "ROS (HS, NADA, no clamp)": dict(method="ros", clamp=False),
    "ROS simple": dict(method="ros", plotting_position="simple"),
    "Parametric mean": dict(method="parametric"),
    "Parametric stochastic": dict(method="parametric", impute_type="stochastic"),
    "Substitution DL/2": dict(method="substitution"),
}
MIXED_METHODS = {
    "Mixed ROS sequential (default)": dict(method="ros"),
    "Mixed ROS sequential, Weibull right (LWPTrends)": dict(method="ros", right_dist="weibull"),
    "Mixed ROS turnbull (experimental)": dict(method="ros", mixed_strategy="turnbull"),
    "Parametric mean": dict(method="parametric"),
    "Parametric stochastic": dict(method="parametric", impute_type="stochastic"),
}


def stats(v):
    lv = np.log(v)
    return {"mean_log": lv.mean(), "sd_log": lv.std(ddof=1), "mean": v.mean()}


def run():
    rng = np.random.default_rng(12345)
    rows = []
    scenarios = [("left", n, multi, q) for n in (20, 50, 200) for multi in (False, True) for q in (0.3, 0.6)]
    scenarios += [("mixed", n, None, None) for n in (30, 100)]
    for kind, n, multi, q in scenarios:
        methods = LEFT_METHODS if kind == "left" else MIXED_METHODS
        acc = {m: [] for m in methods}
        for rep in range(N_REPS):
            if kind == "left":
                v, c = left_scenario(rng, n, multi, q)
                if (~c).sum() < 3 or c.sum() == 0:
                    continue
                status, ctype = c, "left"
            else:
                v, c = mixed_scenario(rng, n)
                if (c == 0).sum() < 3:
                    continue
                status, ctype = c, "mixed"
            for m, kw in methods.items():
                try:
                    out = impute(v, status, censoring_type=ctype, random_state=rep, **kw)["imputed_value"].values
                    if np.any(out <= 0):
                        continue
                    acc[m].append(stats(out))
                except ValueError:
                    pass
        label = f"{kind} n={n}" + ("" if kind == "mixed" else f" {'multi' if multi else 'single'}DL cens~{q:.0%}")
        for m, lst in acc.items():
            d = pd.DataFrame(lst)
            row = {"scenario": label, "method": m, "reps": len(d)}
            for k, t in TRUE.items():
                row[f"bias_{k}"] = d[k].mean() - t
                row[f"rmse_{k}"] = np.sqrt(((d[k] - t) ** 2).mean())
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv("mc_results.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
