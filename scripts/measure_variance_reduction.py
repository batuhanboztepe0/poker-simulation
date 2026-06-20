"""
measure_variance_reduction.py
-----------------------------
How much does each variance-reduction technique narrow the confidence interval
of a measured edge? Compares four arms of the SAME seeded matchup:

    raw            single-orientation, realised outcomes
    mirror         duplicate/mirror matching (swap seats, average)
    luck_adjusted  all-in EV control variate (EV-score all-in pots)
    mirror+luck    both stacked

and reports each arm's mean edge + 95% bootstrap CI + CI width. This is the
honest demonstration of the rigor layer (references.md §2): the CI should narrow
as variance is removed. Expect mirror to help most in this multi-hand bust-match
format and the all-in EV adjustment to help modestly (match-outcome variance is
dominated by bust path-dependence, not single-hand runout luck — the large AIVAT
gains are for per-hand win-rate estimation, not this format).

    OMP_NUM_THREADS=1 python -m scripts.measure_variance_reduction --out results/variance_reduction.json
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json

from src.player import BotPlayer
from src.monte_carlo import MonteCarloEngine
from src.evaluation import evaluate_matchup, bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=120)
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--mc-sims", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def a_f(pid, s):
        return BotPlayer(pid, "Myopic", s, tight_threshold=0.2, aggression=0.5,
                         mc_engine=MonteCarloEngine(args.mc_sims))

    def b_f(pid, s):  # aggressive -> frequent all-ins (exercises the EV adj.)
        return BotPlayer(pid, "Aggro", s, tight_threshold=0.1, aggression=0.9,
                         mc_engine=MonteCarloEngine(args.mc_sims))

    seeds = list(range(args.seeds))
    arms = [
        ("raw", {}),
        ("mirror", {"mirror": True}),
        ("luck_adjusted", {"luck_adjusted": True}),
        ("mirror+luck", {"mirror": True, "luck_adjusted": True}),
    ]
    print(f"Myopic vs Aggro, {args.seeds} seeds x {args.hands} hands\n")
    rows = []
    for name, kw in arms:
        mr = evaluate_matchup(a_f, b_f, "Myopic", "Aggro", seeds,
                              n_hands=args.hands, **kw)
        ci = bootstrap_ci(mr.diffs)
        width = ci["hi"] - ci["lo"]
        rows.append({"arm": name, "mean": mr.mean_diff, "lo": ci["lo"],
                     "hi": ci["hi"], "ci_width": width,
                     "n_seeds": len(seeds), "n_hands": args.hands})
        print(f"  {name:14s} mean {mr.mean_diff:+7.0f}  95% CI "
              f"[{ci['lo']:+.0f}, {ci['hi']:+.0f}]  width {width:.0f}")

    base = rows[0]["ci_width"] or 1.0
    for r in rows:
        r["ci_width_vs_raw"] = r["ci_width"] / base
    print(f"\nCI width vs raw: " + "  ".join(
        f"{r['arm']}={r['ci_width_vs_raw']:.2f}" for r in rows))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"arms": rows}, f)


if __name__ == "__main__":
    main()
