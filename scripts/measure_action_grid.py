"""
measure_action_grid.py
----------------------
Phase B2: does the finer 7-action grid (raise quarter/half/two-thirds/pot + all-in)
beat the default 5-action grid (raise half/pot + all-in) as the RL policy's action
space?

A finer bet-sizing grid is strictly more expressive (it can size bets the 5-grid
cannot), but it also spreads the same training budget over more actions, so the
Q-values for each sizing are estimated from fewer samples. This is the honest A/B:
identical training EXCEPT the action grid, per init seed, evaluated heads-up vs the
myopic baseline on held-out seeds (the headline format).

Run ONE (grid, init-seed) cell and emit a JSON line so cells can be launched
thread-pinned in parallel (the small MLP only thrashes under parallel BLAS):

    OMP_NUM_THREADS=1 python -m scripts.measure_action_grid --grid five --seed 0
    OMP_NUM_THREADS=1 python -m scripts.measure_action_grid --grid seven --seed 0

`--aggregate f1.json f2.json ...` folds the emitted JSON lines into a report.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json

import torch

from src.rl_agent import SelfPlayTrainer, evaluate_vs_baseline, paired_t_test


def run_cell(grid, init_seed, steps):
    """Train one agent with the chosen action grid; eval it vs myopic."""
    torch.manual_seed(init_seed)
    ext = (grid == "seven")
    tr = SelfPlayTrainer(seed=1, opponent_mode="fixed", mc_sims=100,
                         epsilon_start=1.0, epsilon_end=0.05,
                         extended_actions=ext)
    tr.train(steps, batch_size=64, refresh_every=10, hands_per_refresh=12)
    # Held-out eval: three disjoint 50-seed blocks (seeds 0..149) -> 150 diffs.
    diffs = []
    for start in (0, 50, 100):
        diffs += evaluate_vs_baseline(tr.qnet, n_seeds=50, n_hands=150,
                                      mc_sims=100, seed_start=start,
                                      extended_actions=ext)["per_seed_diffs"]
    tt = paired_t_test(diffs)
    return {
        "grid": grid,
        "n_actions": tr.qnet.net[-1].out_features,
        "init_seed": init_seed,
        "wins": sum(1 for d in diffs if d > 0),
        "n": len(diffs),
        "mean": tt["mean"],
        "p_value": tt["p_value"],
    }


def aggregate(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    by = {}
    for r in rows:
        by.setdefault(r["grid"], []).append(r)
    print("\nB2 action-grid A/B (single-hand RL, heads-up vs myopic)\n")
    print(f"  {'grid':>6} {'seed':>4} {'wins/n':>9} {'mean':>8} {'p':>8}")
    for grid in sorted(by):
        g = sorted(by[grid], key=lambda x: x["init_seed"])
        for r in g:
            p = r["p_value"]
            pstr = f"{p:.4f}" if p is not None else "  n/a"
            print(f"  {grid:>6} {r['init_seed']:>4} {r['wins']:>4}/{r['n']:<4} "
                  f"{r['mean']:>+8.0f} {pstr:>8}")
        mw = sum(r["wins"] for r in g) / len(g)
        mm = sum(r["mean"] for r in g) / len(g)
        worst = min(r["mean"] for r in g)
        print(f"  {grid:>6}  AVG({len(g)})    {mw:>4.0f}/{g[0]['n']:<4} "
              f"{mm:>+8.0f}   (worst seed mean {worst:+.0f})\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["five", "seven"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default=None,
                    help="append the result JSON line to this file")
    ap.add_argument("--aggregate", nargs="+", default=None,
                    help="fold these JSON-line files into a report and exit")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.aggregate)
        return

    res = run_cell(args.grid, args.seed, args.steps)
    line = json.dumps(res)
    print(line)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
