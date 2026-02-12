"""Recover and display results from a partial/complete holdout run.

Usage:
    python experiments/recover_results.py experiments/holdout_results_obj9.json
"""

import json
import sys
import numpy as np


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return p_hat, max(0, center - margin), min(1, center + margin)


def mcnemar_test(per_a, per_b):
    n = min(len(per_a), len(per_b))
    a_wins = sum(1 for i in range(n) if per_a[i]["accuracy"] > 0 and per_b[i]["accuracy"] == 0)
    b_wins = sum(1 for i in range(n) if per_a[i]["accuracy"] == 0 and per_b[i]["accuracy"] > 0)
    n_discord = a_wins + b_wins
    if n_discord == 0:
        return a_wins, b_wins, 1.0
    try:
        from scipy.stats import binomtest
        result = binomtest(b_wins, n_discord, 0.5)
        return a_wins, b_wins, float(result.pvalue)
    except (ImportError, AttributeError):
        from math import comb
        k = min(a_wins, b_wins)
        p = sum(comb(n_discord, i) for i in range(k + 1)) / 2**n_discord * 2
        return a_wins, b_wins, min(float(p), 1.0)


def analyze_method(name, per_instance):
    accs = [r["accuracy"] for r in per_instance]
    logZs = [r["log_ml"] for r in per_instance]
    finite_logZs = [z for z in logZs if np.isfinite(z)]
    times = [r["wall_time"] for r in per_instance]
    ess_vals = [r.get("ess", 0.0) for r in per_instance]

    n = len(accs)
    k = sum(1 for a in accs if a > 0)
    acc, ci_lo, ci_hi = wilson_ci(k, n)

    return {
        "method": name,
        "n": n,
        "accuracy": acc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_solved": k,
        "mean_log_ml": float(np.mean(finite_logZs)) if finite_logZs else float("-inf"),
        "var_log_ml": float(np.var(finite_logZs)) if len(finite_logZs) > 1 else 0.0,
        "n_finite": len(finite_logZs),
        "mean_ess": float(np.mean(ess_vals)),
        "mean_time": float(np.mean(times)),
        "total_time": float(np.sum(times)),
        "per_instance": per_instance,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "experiments/holdout_results_obj9.json"
    with open(path) as f:
        data = json.load(f)

    print(f"File: {path}")
    print(f"Status: {data.get('status', 'unknown')}")
    print(f"Model: {data.get('model', '?')}")
    print(f"max_objects: {data.get('max_objects', '?')}")
    print(f"n_particles: {data.get('n_particles', '?')}")
    print(f"n_train: {data.get('n_train', '?')}, n_test: {data.get('n_test', '?')}")

    # Use completed summaries if available, otherwise recover from incremental data
    if "results" in data and data["results"]:
        summaries = data["results"]
        # Ensure per_instance is present
        for s in summaries:
            if "per_instance" not in s and "methods" in data and s["method"] in data["methods"]:
                s["per_instance"] = data["methods"][s["method"]]
    elif "methods" in data:
        print("\n  (Recovering from incremental data — run did not complete)")
        summaries = []
        for name, per_instance in data["methods"].items():
            summaries.append(analyze_method(name, per_instance))
    else:
        print("No results found in file.")
        return

    # Print table
    print(f"\n{'='*95}")
    print(f"{'Method':<20} {'N':>4} {'Acc':>6} {'95% CI':>14} {'Solved':>8} {'E[logZ]':>8} {'Var[logZ]':>10} {'ESS':>5} {'Time':>6}")
    print("-" * 95)
    for s in summaries:
        n = s.get("n", s.get("n_total", "?"))
        mean_ess = s.get("mean_ess", 0.0)
        var_logml = s.get("var_log_ml", s.get("var_log_ml", 0.0))
        mean_logml = s.get("mean_log_ml", s.get("mean_log_ml", float("-inf")))
        mean_time = s.get("mean_time", s.get("mean_wall_time", 0.0))
        print(f"{s['method']:<20} {n:>4} {s['accuracy']:>6.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] "
              f"{s['n_solved']:>3}/{n:<3}  {mean_logml:>8.2f} {var_logml:>10.2f} "
              f"{mean_ess:>5.1f} {mean_time:>6.1f}s")

    # Paired comparisons
    if len(summaries) > 1:
        print(f"\n{'='*95}")
        print("PAIRED COMPARISONS (McNemar's test, on shared instances)")
        print(f"{'='*95}")
        for i, sa in enumerate(summaries):
            for j, sb in enumerate(summaries):
                if j <= i:
                    continue
                per_a = sa.get("per_instance", [])
                per_b = sb.get("per_instance", [])
                n_shared = min(len(per_a), len(per_b))
                if n_shared == 0:
                    continue
                a_wins, b_wins, p = mcnemar_test(per_a, per_b)
                sig = " *" if p < 0.05 else ""
                sig = " **" if p < 0.01 else sig
                print(f"  {sa['method']:<20} vs {sb['method']:<20} (n={n_shared}): "
                      f"A_wins={a_wins}, B_wins={b_wins}  p={p:.4f}{sig}")


if __name__ == "__main__":
    main()
