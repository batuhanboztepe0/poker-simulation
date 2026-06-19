"""
measure_selfplay.py
-------------------
Phase B4: does snapshot self-play (opponent_mode='snapshot' -- the learner trains
vs frozen past snapshots of itself, fictitious self-play) produce a stronger or
more ROBUST agent than the proven fixed-vs-myopic recipe?

The other Block-B levers (finer action grid, wider clip, decoupled bonus, warmed
FE) tweak the reward/feature/action MECHANICS and came back neutral. Self-play
changes the TRAINING DISTRIBUTION instead -- the lever family that actually paid
off (opponent-mix/domain-randomization was a §10 win). Fixed-vs-myopic is directly
optimised for the myopic headline, so the interesting question is whether
self-play GENERALISES better to the adaptive pool {tilt, random} even if it gives
up something on the myopic bench it never trains against.

This is the honest A/B: identical training EXCEPT opponent_mode, per init seed,
each evaluated on held-out seeds vs the pool {myopic, tilt, random}.

Run ONE (config, init-seed) cell -> JSON line (launch cells thread-pinned in
parallel; the small MLP only thrashes under parallel BLAS):

    OMP_NUM_THREADS=1 python -m scripts.measure_selfplay --config snapshot --seed 0
    OMP_NUM_THREADS=1 python -m scripts.measure_selfplay --config fixed    --seed 0

`--aggregate f1.json ...` folds the JSON lines into a per-opponent table.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json

import torch

import src.rl_agent as rl
from src.player import BotPlayer
from src.adaptive_agent import AdaptiveBotPlayer
from src.monte_carlo import MonteCarloEngine
from src.evaluation import evaluate_matchup


def _mc(sims):
    return MonteCarloEngine(n_simulations=sims)


def train_config(config, init_seed, steps, mc_sims):
    """config 'fixed' = train vs the myopic baseline; 'snapshot' = self-play."""
    torch.manual_seed(init_seed)
    tr = rl.SelfPlayTrainer(
        n_players=2, seed=1, opponent_mode=config, mc_sims=mc_sims,
        epsilon_start=1.0, epsilon_end=0.05, gamma=0.97, snapshot_every=200)
    tr.train(steps, batch_size=64, refresh_every=10, hands_per_refresh=12)
    return tr.qnet


def eval_vs_pool(qnet, seeds, hands, mc_sims):
    def rl_f(pid, stack):
        return rl.RLBotPlayer(pid, "RL", stack, qnet=qnet, epsilon=0.0,
                              training=False, feature_mode="base",
                              mc_engine=_mc(mc_sims))

    pool = {
        "myopic": lambda pid, stack: BotPlayer(pid, "Myopic", stack,
                                               tight_threshold=0.2, aggression=0.5,
                                               mc_engine=_mc(mc_sims)),
        "tilt": lambda pid, stack: AdaptiveBotPlayer(pid, "Tilt", stack,
                                                     mode="tilt", mc_engine=_mc(mc_sims)),
        "random": lambda pid, stack: AdaptiveBotPlayer(pid, "Random", stack,
                                                       mode="random", mc_engine=_mc(mc_sims)),
    }
    out = {}
    for name, opp in pool.items():
        mr = evaluate_matchup(rl_f, opp, "RL", name, seeds, n_hands=hands)
        out[name] = sum(mr.net_a) / len(mr.net_a)
        out[name + "_wins"] = mr.wins_a
    return out


def run_cell(config, init_seed, steps, hands, mc_sims, eval_seeds, eval_start):
    qnet = train_config(config, init_seed, steps, mc_sims)
    seeds = list(range(eval_start, eval_start + eval_seeds))
    means = eval_vs_pool(qnet, seeds, hands, mc_sims)
    return {"config": config, "init_seed": init_seed, "n": eval_seeds, **means}


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
        by.setdefault(r["config"], []).append(r)
    print("\nB4 self-play A/B (RL mean net chips vs the held-out pool)\n")
    print(f"  {'config':>9} {'seed':>4} {'myopic':>8} {'tilt':>8} {'random':>8}")
    for config in sorted(by):
        g = sorted(by[config], key=lambda x: x["init_seed"])
        for r in g:
            print(f"  {config:>9} {r['init_seed']:>4} {r['myopic']:>+8.0f} "
                  f"{r['tilt']:>+8.0f} {r['random']:>+8.0f}")
        n = len(g)
        mm = {k: sum(r[k] for r in g) / n for k in ("myopic", "tilt", "random")}
        worst = {k: min(r[k] for r in g) for k in ("myopic", "tilt", "random")}
        print(f"  {config:>9}  AVG  {mm['myopic']:>+8.0f} {mm['tilt']:>+8.0f} "
              f"{mm['random']:>+8.0f}")
        print(f"  {config:>9} WORST {worst['myopic']:>+8.0f} {worst['tilt']:>+8.0f} "
              f"{worst['random']:>+8.0f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["fixed", "snapshot"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--hands", type=int, default=150)
    ap.add_argument("--mc-sims", type=int, default=100)
    ap.add_argument("--eval-seeds", type=int, default=40)
    ap.add_argument("--eval-start", type=int, default=1000)  # held out from training
    ap.add_argument("--out", default=None)
    ap.add_argument("--aggregate", nargs="+", default=None)
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.aggregate)
        return

    res = run_cell(args.config, args.seed, args.steps, args.hands, args.mc_sims,
                   args.eval_seeds, args.eval_start)
    line = json.dumps(res)
    print(line)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
