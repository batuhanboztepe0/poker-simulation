"""
measure_confirmatory.py
-----------------------
Execute the PRE-REGISTERED confirmatory RL-vs-Myopic evaluation exactly as locked
in PREREGISTRATION.md §4.3:

    result = evaluate_matchup(
        factory_rl, factory_myopic,
        name_a="RL", name_b="Myopic",
        seeds=list(range(500)),
        n_hands=100,
        mirror=True,
        luck_adjusted=False,
    )
    ci = bootstrap_ci(result.diffs)

The headline figure (results/headline_history.json) was an EXPLORATORY pilot that
used a different path (evaluate_vs_baseline, 200 seeds x 200 hands, single
orientation). This script runs the registered confirmatory protocol and reports
its result in full regardless of sign (PREREGISTRATION.md §8).

The RL agent is retrained deterministically with the SAME recipe as the pilot
(scripts/measure_headline.py): torch.manual_seed(0), SelfPlayTrainer(seed=1,
opponent_mode="fixed", mc_sims=100, hidden=64, epsilon_start=1.0,
epsilon_end=0.05), train(1500, batch_size=64, refresh_every=10,
hands_per_refresh=12). A self-check reproduces the pilot's evaluate_vs_baseline
number (≈125/200) to confirm the retrained policy is the same agent before the
confirmatory call is run, and the checkpoint is saved so the run is reproducible
without retraining (PREREGISTRATION.md §4.2).

    OMP_NUM_THREADS=1 python -m scripts.measure_confirmatory --out results/confirmatory.json
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json

import torch

import src.rl_agent as rl
from src.player import BotPlayer
from src.monte_carlo import MonteCarloEngine
from src.evaluation import evaluate_matchup, bootstrap_ci
from src.stats import binomial_sign_test


def _mc(sims):
    return MonteCarloEngine(n_simulations=sims)


def train_pilot_agent(steps, mc_sims, hidden, torch_seed):
    """Retrain the headline fixed-vs-myopic agent identically to
    scripts/measure_headline.py (in-training eval omitted — it uses an
    independent RNG and does not affect the learned weights)."""
    torch.manual_seed(torch_seed)
    tr = rl.SelfPlayTrainer(seed=1, opponent_mode="fixed", mc_sims=mc_sims,
                            hidden=hidden, epsilon_start=1.0, epsilon_end=0.05)
    tr.train(steps, batch_size=64, refresh_every=10, hands_per_refresh=12)
    return tr.qnet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--mc-sims", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--torch-seed", type=int, default=0)
    ap.add_argument("--selfcheck-seeds", type=int, default=200)
    ap.add_argument("--selfcheck-hands", type=int, default=200)
    ap.add_argument("--raw", action="store_true",
                    help="also run the raw single-orientation arm (PREREG §2.2 calibration)")
    ap.add_argument("--checkpoint", default="models/confirmatory_rl.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # --- 1. Retrain the pilot agent deterministically ---------------------
    print(f"[1/4] training RL agent ({args.steps} steps, torch_seed={args.torch_seed})...",
          flush=True)
    qnet = train_pilot_agent(args.steps, args.mc_sims, args.hidden, args.torch_seed)

    # --- 2. Self-check: reproduce the pilot number (same-agent gate) -------
    print(f"[2/4] self-check vs pilot protocol "
          f"({args.selfcheck_seeds} seeds x {args.selfcheck_hands} hands)...", flush=True)
    sc = rl.evaluate_vs_baseline(qnet, n_seeds=args.selfcheck_seeds,
                                 n_hands=args.selfcheck_hands, mc_sims=args.mc_sims)
    sc_ci = bootstrap_ci(sc["per_seed_diffs"])
    print(f"      pilot self-check: {sc['wins']}/{sc['n_seeds']} "
          f"mean {sc['mean_chip_diff']:+.0f} CI [{sc_ci['lo']:+.0f}, {sc_ci['hi']:+.0f}]",
          flush=True)

    # --- 3. Save the checkpoint (reproducible without retraining) ----------
    rl.save_checkpoint(qnet, args.checkpoint, hidden=args.hidden,
                       feature_mode="base",
                       meta={"recipe": "fixed-vs-myopic headline pilot",
                             "steps": args.steps, "torch_seed": args.torch_seed})
    print(f"[3/4] checkpoint saved -> {args.checkpoint}", flush=True)

    # --- 4. Pre-registered confirmatory call (PREREG §4.3) -----------------
    def factory_rl(pid, stack):
        return rl.RLBotPlayer(pid, "RL", stack, qnet=qnet, epsilon=0.0,
                              training=False, feature_mode="base",
                              mc_engine=_mc(args.mc_sims))

    def factory_myopic(pid, stack):
        return BotPlayer(pid, "Myopic", stack, tight_threshold=0.2,
                         aggression=0.5, mc_engine=_mc(args.mc_sims))

    seeds = list(range(args.seeds))

    def run_arm(mirror):
        mr = evaluate_matchup(factory_rl, factory_myopic, "RL", "Myopic", seeds,
                              n_hands=args.hands, mirror=mirror, luck_adjusted=False)
        ci = bootstrap_ci(mr.diffs)
        binom = binomial_sign_test(mr.diffs)
        return {
            "mirror": mirror, "n_seeds": len(seeds), "n_hands": args.hands,
            "wins_rl": mr.wins_a, "wins_myopic": mr.wins_b, "ties": mr.ties,
            "mean_diff": mr.mean_diff, "ci95": ci,
            "paired_t": mr.t_test, "binom": binom,
            "resolved": ci["lo"] > 0 or ci["hi"] < 0,
            "per_seed_diffs": list(mr.diffs),
        }

    print(f"[4/4] confirmatory evaluate_matchup: {args.seeds} seeds x {args.hands} hands, mirror=True...",
          flush=True)
    primary = run_arm(mirror=True)
    print(f"      CONFIRMATORY (mirror): {primary['wins_rl']}/{primary['n_seeds']} "
          f"mean {primary['mean_diff']:+.0f} "
          f"CI [{primary['ci95']['lo']:+.0f}, {primary['ci95']['hi']:+.0f}] "
          f"binom_p={primary['binom']['p_value']:.4f} resolved={primary['resolved']}",
          flush=True)

    raw = None
    if args.raw:
        print(f"      raw calibration arm (mirror=False)...", flush=True)
        raw = run_arm(mirror=False)
        print(f"      RAW: {raw['wins_rl']}/{raw['n_seeds']} mean {raw['mean_diff']:+.0f} "
              f"CI [{raw['ci95']['lo']:+.0f}, {raw['ci95']['hi']:+.0f}]", flush=True)

    out = {
        "protocol": {
            "function": "evaluate_matchup",
            "seeds": f"list(range({args.seeds}))",
            "n_hands": args.hands,
            "mirror": True,
            "luck_adjusted": False,
            "baseline": {"name": "Myopic", "type": "BotPlayer",
                         "tight_threshold": 0.2, "aggression": 0.5,
                         "mc_sims": args.mc_sims},
            "rl_agent": {"checkpoint": args.checkpoint, "steps": args.steps,
                         "torch_seed": args.torch_seed, "hidden": args.hidden,
                         "feature_mode": "base"},
            "registered_in": "PREREGISTRATION.md §4.3",
        },
        "selfcheck_pilot": {
            "function": "evaluate_vs_baseline",
            "n_seeds": sc["n_seeds"], "n_hands": args.selfcheck_hands,
            "wins": sc["wins"], "mean_chip_diff": sc["mean_chip_diff"],
            "ci95": sc_ci,
            "note": "reproduces the exploratory pilot (results/headline_history.json) "
                    "to confirm the retrained policy is the same agent before the "
                    "confirmatory call",
        },
        "confirmatory_primary": primary,
        "confirmatory_raw": raw,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
