"""
measure_neural_nfsp.py
----------------------
v2 Phase 2: measure NEURAL NFSP convergence on Leduc by the EXACT NashConv
metric (`leduc_eval.exploitability_of`) — the identical, non-tunable metric the
tabular learners are scored on — and compare it head-to-head with the committed
tabular NFSP baseline (`results/exploitability.json` `nfsp_curve`).

Applying the Phase 0 lesson (never trust a single training seed), this trains
neural NFSP over several seeds and reports the exploitability at each checkpoint
as a distribution (mean and min/max across seeds), at episode counts that MATCH
the tabular baseline's checkpoints so the comparison is apples-to-apples.

    OMP_NUM_THREADS=1 python -m scripts.measure_neural_nfsp --out results/neural_nfsp.json

Frozen config and the neural-vs-tabular comparison are pre-registered in
PREREGISTRATION.md §11 before this is run (two-commit git-provable gap).
Requires torch (optional dependency).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time

from src.leduc_neural_nfsp import LeducNeuralNFSP, FEAT_DIM
from src.leduc_eval import (exploitability_of, uniform_strategy_table)


def _tabular_nfsp_ref():
    """The committed tabular NFSP curve (for the head-to-head figure/comparison),
    plus the CFR average-policy floor and uniform ceiling, from the existing
    exploitability result if present."""
    path = "results/exploitability.json"
    ref = {"tabular_nfsp_curve": None, "cfr_avg_final": None,
           "uniform": exploitability_of(uniform_strategy_table())}
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        ref["tabular_nfsp_curve"] = d.get("nfsp_curve")
        if d.get("curve"):
            ref["cfr_avg_final"] = d["curve"][-1]["avg_exploitability"]
    return ref


def run_seed(seed, checkpoints, cfg):
    """Train one neural-NFSP seed; return [{episodes, exploitability}] at each
    checkpoint (exact NashConv of the average policy Pi)."""
    m = LeducNeuralNFSP(seed=seed, **cfg)
    out = []
    prev = 0
    for ck in checkpoints:
        m.train(ck - prev)
        prev = ck
        out.append({"episodes": ck,
                    "exploitability": exploitability_of(m.average_strategy_table())})
    return out


def main():
    ap = argparse.ArgumentParser()
    # Hyperparameters are the a-priori LeducNeuralNFSP defaults (standard NFSP
    # practice), deliberately NOT tuned on the exploitability metric — selecting
    # them after seeing exploitability would be the garden-of-forking-paths the
    # Phase 0 work guards against. They are frozen in PREREGISTRATION.md §11.
    ap.add_argument("--seeds", type=int, default=5, help="number of training seeds")
    ap.add_argument("--checkpoints", default="50000,100000,200000",
                    help="comma-separated episode checkpoints (match tabular NFSP)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--lr-rl", type=float, default=0.01)
    ap.add_argument("--lr-sl", type=float, default=0.01)
    ap.add_argument("--eps-start", type=float, default=0.06)
    ap.add_argument("--eps-end", type=float, default=0.0)
    ap.add_argument("--target-update", type=int, default=1000)
    ap.add_argument("--out", default="results/neural_nfsp.json")
    args = ap.parse_args()

    checkpoints = [int(x) for x in args.checkpoints.split(",")]
    cfg = dict(hidden=args.hidden, eta=args.eta, lr_rl=args.lr_rl,
               lr_sl=args.lr_sl, eps_start=args.eps_start, eps_end=args.eps_end,
               target_update=args.target_update)

    t0 = time.time()
    per_seed = []
    for s in range(args.seeds):
        curve = run_seed(s, checkpoints, cfg)
        per_seed.append({"seed": s, "curve": curve})
        last = curve[-1]
        print(f"seed {s}: final ep={last['episodes']} "
              f"expl={last['exploitability']:.3f} ({time.time()-t0:.0f}s)",
              flush=True)

    # Aggregate per checkpoint across seeds: mean and min/max.
    agg = []
    for idx, ck in enumerate(checkpoints):
        vals = [ps["curve"][idx]["exploitability"] for ps in per_seed]
        n = len(vals)
        mean = sum(vals) / n
        agg.append({"episodes": ck, "mean_exploitability": mean,
                    "min": min(vals), "max": max(vals),
                    "sd": (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5
                          if n > 1 else 0.0})

    ref = _tabular_nfsp_ref()
    # Head-to-head at matched checkpoints: neural mean vs tabular at same episodes.
    head_to_head = []
    tab = {p["episodes"]: p["exploitability"]
           for p in (ref["tabular_nfsp_curve"] or [])}
    for a in agg:
        ep = a["episodes"]
        if ep in tab:
            head_to_head.append({"episodes": ep,
                                 "neural_mean": a["mean_exploitability"],
                                 "tabular": tab[ep],
                                 "neural_beats_tabular": a["mean_exploitability"] < tab[ep]})

    out = {
        "method": "neural NFSP (Heinrich & Silver 2016) on Leduc, exact NashConv",
        "feature_dim": FEAT_DIM,
        "config": cfg,
        "n_seeds": args.seeds,
        "checkpoints": checkpoints,
        "uniform_exploitability": ref["uniform"],
        "cfr_avg_final": ref["cfr_avg_final"],
        "tabular_nfsp_curve": ref["tabular_nfsp_curve"],
        "per_seed": per_seed,
        "curve": agg,
        "head_to_head": head_to_head,
        "registered_in": "PREREGISTRATION.md §11",
        "note": ("Exact Leduc exploitability (NashConv), identical metric to the "
                 "tabular learners. Neural NFSP generalises across info-sets via a "
                 "21-d feature; on this small game tabular is near-exact, so neural "
                 "is expected to win on sample efficiency (matched-episode), not "
                 "asymptotically. Reported in full per PREREGISTRATION.md §8."),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out} | head_to_head: {head_to_head}")


if __name__ == "__main__":
    main()
