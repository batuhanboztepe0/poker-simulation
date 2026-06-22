"""
measure_headline.py
--------------------
Train the headline fixed-vs-myopic RL agent (fixed myopic opponent) with DENSE
held-out eval, and dump the learning-curve history to JSON for the
figure layer (scripts/make_figures.py).

Each `SelfPlayTrainer.history` snapshot is
{step, wins, n_seeds, mean_chip_diff, per_seed_diffs}; the per-snapshot
per_seed_diffs spread is what gives the headline figure its confidence ribbon.
The belief generalist is trained with eval_every=None (its base-feature
vs-myopic curve would mismatch the belief net), so this fixed-vs-myopic curve is
the reproducible learning-curve source.

Run thread-pinned (small MLP; parallel BLAS only thrashes):
    OMP_NUM_THREADS=1 python -m scripts.measure_headline --out results/headline_history.json
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json

import torch

from src.rl_agent import SelfPlayTrainer, evaluate_vs_baseline, paired_t_test
from src.evaluation import bootstrap_ci
from src.stats import binomial_sign_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--torch-seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-seeds", type=int, default=50)
    ap.add_argument("--eval-hands", type=int, default=150)
    ap.add_argument("--final-seeds", type=int, default=200)
    ap.add_argument("--final-hands", type=int, default=200)
    ap.add_argument("--mc-sims", type=int, default=100)
    ap.add_argument("--out", default=None, help="write the history JSON here")
    args = ap.parse_args()

    # init_seed (torch) varies weight-init; trainer RNG held at seed=1 (the
    # rl_multihand_sweep convention) so decks/opponents are common -- consistent
    # with the Block B measurement scripts.
    torch.manual_seed(args.torch_seed)
    tr = SelfPlayTrainer(seed=1, opponent_mode="fixed", mc_sims=args.mc_sims,
                         hidden=args.hidden, epsilon_start=1.0, epsilon_end=0.05)
    tr.train(args.steps, batch_size=64, refresh_every=10, hands_per_refresh=12,
             eval_every=args.eval_every, eval_seeds=args.eval_seeds,
             eval_hands=args.eval_hands, eval_mc_sims=args.mc_sims)

    final = evaluate_vs_baseline(tr.qnet, n_seeds=args.final_seeds,
                                 n_hands=args.final_hands, mc_sims=args.mc_sims)
    tt = paired_t_test(final["per_seed_diffs"])
    binom = binomial_sign_test(final["per_seed_diffs"])
    out = {
        "torch_seed": args.torch_seed,
        "steps": args.steps,
        "eval_seeds": args.eval_seeds,
        "eval_hands": args.eval_hands,
        "history": tr.history,
        "final": {
            "wins": final["wins"],
            "n_seeds": final["n_seeds"],
            "n_hands": args.final_hands,
            "mean_chip_diff": final["mean_chip_diff"],
            # paired_t treats the binary bust outcomes as continuous and
            # understates p; binom is the exact sign test (the correct one).
            "p_value": tt["p_value"],
            "binom_p": binom["p_value"],
            "binom": binom,
            "ci95": bootstrap_ci(final["per_seed_diffs"]),
            "per_seed_diffs": final["per_seed_diffs"],
        },
    }
    print(f"HEADLINE wins {final['wins']}/{final['n_seeds']} "
          f"mean {final['mean_chip_diff']:+.0f} "
          f"paired_p={tt['p_value']:.4f} binom_p={binom['p_value']:.4f} "
          f"({len(tr.history)} curve points)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
