"""
measure_scale.py
----------------
v2 Phase 2 scale-up (Step 2d): does a neural method learn where tabular CFR cannot
practically converge?

On the parameterised R-rank Leduc (`big_leduc`) at R = `--ranks` (default 20: 40
cards, ~12k info-sets, ~59k deals per CFR iteration), this:

  1. Measures the tabular-CFR cost curve (time per full CFR iteration vs R) and
     extrapolates the time to CONVERGE (the ~10^4 iterations the 3-rank CFR needed
     to reach ~0.009 exploitability) — establishing that tabular CFR convergence is
     infeasible at this scale.
  2. Runs a MATCHED-WALL-CLOCK head-to-head: tabular CFR for `--cfr-iters`
     iterations vs neural NFSP for `--neural-episodes` episodes over `--seeds`
     seeds (pre-committed counts chosen to be ~equal wall-clock; the actual
     wall-clock of each is recorded). Both average policies are scored by the
     EXACT NashConv metric (still feasible as a one-time best response at this R,
     even though CFR convergence is not) and cross-checked by the validated LBR
     lower bound.

The protocol and counts are frozen in PREREGISTRATION.md §12 before the run (the
two-commit git-provable-gap discipline). Reported in full regardless of which method
wins the head-to-head (§8); the convergence-infeasibility result stands independently.

    OMP_NUM_THREADS=1 python -m scripts.measure_scale --out results/scale_experiment.json
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time

from src.big_leduc import (BigLeducCFR, exploitability_of, all_info_sets,
                           num_deals)
from src.big_leduc_lbr import lbr_exploitability
from src.big_leduc_nfsp import BigLeducNeuralNFSP
from src.leduc_cfr import _avail_from_key, NUM_ACTIONS


def _uniform_table(ranks):
    t = {}
    for s in all_info_sets(ranks):
        av = _avail_from_key(s)
        t[s] = [1.0 / len(av) if a in av else 0.0 for a in range(NUM_ACTIONS)]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=20)
    ap.add_argument("--neural-episodes", type=int, default=200000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cfr-iters", type=int, default=30)
    ap.add_argument("--converge-iters", type=int, default=10000,
                    help="iteration count the 3-rank CFR needed for ~0.009 (for the "
                         "convergence-time extrapolation)")
    ap.add_argument("--lbr-samples", type=int, default=None,
                    help="cap deals in the LBR final-value estimate (None=exact)")
    ap.add_argument("--out", default="results/scale_experiment.json")
    args = ap.parse_args()
    R = args.ranks

    # 1. tabular-CFR cost curve + convergence extrapolation -------------------
    # Dedupe the probe ranks so R appears once even when R is one of the fixed
    # probe points (3/6/10/16); otherwise the `next(... ranks == R)` below would
    # pick the first (cold-cache) timing instead of a single definitive one.
    cost_curve = []
    probe_ranks = [3, 6, 10, 16]
    for r in probe_ranks + ([R] if R not in probe_ranks else []):
        c = BigLeducCFR(r)
        t0 = time.time()
        c.train(1)
        cost_curve.append({"ranks": r, "deals": num_deals(r),
                           "info_sets": len(all_info_sets(r)),
                           "secs_per_iter": round(time.time() - t0, 3)})
    spi = next(c["secs_per_iter"] for c in cost_curve if c["ranks"] == R)
    converge_hours = spi * args.converge_iters / 3600.0
    print(f"[1] R={R}: {spi:.2f}s/CFR-iter -> ~{args.converge_iters} iters to "
          f"converge ~= {converge_hours:.1f} h (infeasible)", flush=True)

    # reference: the uniform-random ceiling, exact + LBR -----------------------
    uni = _uniform_table(R)
    uni_exact = exploitability_of(uni, R)
    uni_lbr = lbr_exploitability(uni, R, samples=args.lbr_samples)
    print(f"[ref] uniform: exact={uni_exact:.3f} LBR={uni_lbr:.3f}", flush=True)

    # 2a. tabular CFR at the matched budget -----------------------------------
    t0 = time.time()
    cfr = BigLeducCFR(R)
    cfr.train(args.cfr_iters)
    cfr_secs = time.time() - t0
    cfr_table = cfr.strategy_table()
    cfr_exact = exploitability_of(cfr_table, R)
    cfr_lbr = lbr_exploitability(cfr_table, R, samples=args.lbr_samples)
    print(f"[2a] tabular CFR ({args.cfr_iters} iters, {cfr_secs:.0f}s): "
          f"exact={cfr_exact:.3f} LBR={cfr_lbr:.3f}", flush=True)

    # 2b. neural NFSP at the matched budget, multi-seed -----------------------
    neural = []
    for s in range(args.seeds):
        t0 = time.time()
        m = BigLeducNeuralNFSP(ranks=R, seed=s)
        m.train(args.neural_episodes)
        secs = time.time() - t0
        tab = m.average_strategy_table()
        exact = exploitability_of(tab, R)
        lbr = lbr_exploitability(tab, R, samples=args.lbr_samples)
        neural.append({"seed": s, "train_secs": round(secs, 0),
                       "exact": exact, "lbr": lbr})
        print(f"[2b] neural seed {s} ({secs:.0f}s): exact={exact:.3f} LBR={lbr:.3f}",
              flush=True)

    n_exacts = [x["exact"] for x in neural]
    n_mean = sum(n_exacts) / len(n_exacts)
    out = {
        "ranks": R, "deals_per_cfr_iter": num_deals(R),
        "info_sets": len(all_info_sets(R)),
        "cost_curve": cost_curve,
        "cfr_secs_per_iter": spi,
        "converge_iters_ref": args.converge_iters,
        "cfr_converge_hours_est": converge_hours,
        "uniform": {"exact": uni_exact, "lbr": uni_lbr},
        "matched_budget": {"neural_episodes": args.neural_episodes,
                           "cfr_iters": args.cfr_iters},
        "tabular_cfr": {"iters": args.cfr_iters, "train_secs": round(cfr_secs, 0),
                        "exact": cfr_exact, "lbr": cfr_lbr},
        "neural_nfsp": {"seeds": args.seeds, "per_seed": neural,
                        "exact_mean": n_mean,
                        "exact_min": min(n_exacts), "exact_max": max(n_exacts)},
        "head_to_head": {
            "neural_exact_mean": n_mean, "tabular_exact": cfr_exact,
            # "matched budget" = the pre-committed COUNTS (200k episodes vs 30 CFR
            # iters, §12.1); the wall-clock was NOT matched (neural got ~2.4x) — a
            # disclosed deviation (see PREREGISTRATION.md §12.3).
            "neural_beats_tabular_at_matched_budget": n_mean < cfr_exact,
            "both_far_from_nash": min(n_mean, cfr_exact) > 0.1,
        },
        # The LBR lower-bound guarantee (LBR <= exact under full enumeration) must
        # hold for EVERY scored policy — uniform included, not just neural/CFR.
        "lbr_le_exact_check": (uni_lbr <= uni_exact + 1e-6
                               and cfr_lbr <= cfr_exact + 1e-6
                               and all(x["lbr"] <= x["exact"] + 1e-6 for x in neural)),
        "registered_in": "PREREGISTRATION.md §12",
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nHEAD-TO-HEAD @R={R}, matched ~budget: neural exact mean {n_mean:.3f} "
          f"vs tabular CFR {cfr_exact:.3f} -> "
          f"{'neural lower' if n_mean < cfr_exact else 'tabular lower'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
