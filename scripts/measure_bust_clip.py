"""
measure_bust_clip.py
--------------------
Phase B3: does un-truncating the multi-hand log-utility ruin signal help (or at
least not regress) the heads-up bankroll headline?

In log-mode multi-hand the per-hand reward is the change in log-utility of the
stack; a full single-hand bust is -log(stack0/util_floor) ~ -6.9, but the old
hardcoded reward_clip=3.0 clobbered it to -3.0, truncating ~57% of the ruin
penalty so the gradient could not tell "lost 95%" from "busted".
This script is the honest A/B: identical training EXCEPT the clip, per init seed,
evaluated on held-out seeds. Reports wins/mean/p vs the myopic baseline AND each
agent's bust rate (the risk signature a growth-optimal log-utility policy should,
in theory, keep lower once the ruin signal actually reaches the gradient).

FINDING (NEGATIVE; 6 init seeds x 150 held-out matches, 1500 steps):
    clip   avg wins/150   avg mean   avg bust
    3.0          81          +165       0.46
    4.6          58          -340       0.48
    6.9          76           +30       0.49
Widening does NOT help and does NOT reduce bust rate. The tight 3.0 clip has the
best mean AND lowest bust rate; wider clips reintroduce training instability
(e.g. seed 0: bust 0.32 at 3.0 -> 0.74 at 6.9 -> 1.00 at 4.6). And bust rate
~= 1 - win rate here (winner-take-all bust match), so there is no independent
bust-reduction lever to recover by un-clipping. The ~57% truncation is the
deliberate stability price, kept on purpose -> the default clip stays 3.0.

Run ONE (clip, init-seed) cell and emit a JSON line so the cells can be launched
thread-pinned in parallel (the small MLP only thrashes under parallel BLAS):

    OMP_NUM_THREADS=1 python -m scripts.measure_bust_clip --clip old  --seed 0
    OMP_NUM_THREADS=1 python -m scripts.measure_bust_clip --clip 6.9  --seed 0

`--clip old`=3.0, `--clip wide`=the auto default (-log(stack0/util_floor)~6.9), or
a float. `--aggregate f1.json f2.json ...` folds the JSON lines into a report.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import random

import torch

from src.rl_agent import (SelfPlayTrainer, evaluate_vs_baseline, paired_t_test,
                          RLBotPlayer)
from src.monte_carlo import MonteCarloEngine
from src.game import GameEngine
from src.player import BotPlayer


def bust_rate(qnet, n_seeds=50, n_hands=200, mc_sims=100, seed_start=0,
              stack0=1000):
    """Fraction of held-out matches in which the RL agent busts to 0."""
    busts = 0
    for seed in range(seed_start, seed_start + n_seeds):
        rng = random.Random(seed)
        mc = MonteCarloEngine(mc_sims, rng=rng)
        rl = RLBotPlayer(1, "RL", stack0, qnet=qnet, epsilon=0.0,
                         training=False, mc_engine=mc, rng=rng)
        myo = BotPlayer(2, "M", stack0, tight_threshold=0.2, aggression=0.5,
                        mc_engine=mc, rng=rng)
        eng = GameEngine([rl, myo], 10, 20, verbose=False, rng=rng)
        for _ in range(n_hands):
            if min(rl.stack, myo.stack) <= 0:
                break
            eng.play_hand()
        if rl.stack <= 0:
            busts += 1
    return busts / n_seeds


def _resolve_clip(clip):
    """clip is 'old' (3.0), 'wide' (the full ruin range), or a float string.
    'wide' is the explicit -log(stack0/util_floor) ~ 6.9 (NOT None: the trainer's
    None-default is now 3.0 after B3 reverted, so None would silently == 'old')."""
    if clip == "old":
        return 3.0
    if clip == "wide":
        return math.log(1000 / 1.0)  # stack0=1000, util_floor=1.0 -> ~6.9
    return float(clip)


def run_cell(clip, init_seed, steps, hpe):
    """Train one log-mode multi-hand agent and evaluate it on held-out seeds.

    init_seed varies the network weight-init (torch seed); the trainer RNG is held
    at seed=1 (the rl_multihand_sweep convention), so the decks/opponents are
    common across cells. The A/B is therefore PAIRED per init_seed, and cross-seed
    spread reflects weight-init sensitivity (not fully-independent training runs).
    """
    torch.manual_seed(init_seed)
    reward_clip = _resolve_clip(clip)
    tr = SelfPlayTrainer(seed=1, opponent_mode="fixed", mc_sims=100,
                         epsilon_start=1.0, epsilon_end=0.05, gamma=0.99,
                         multi_hand=True, hands_per_episode=hpe,
                         reward_clip=reward_clip)
    tr.train(steps, batch_size=64, refresh_every=5, hands_per_refresh=20)
    # Held-out eval: three disjoint 50-seed blocks (seeds 0..149) -> 150 diffs.
    diffs = []
    for start in (0, 50, 100):
        diffs += evaluate_vs_baseline(tr.qnet, n_seeds=50, n_hands=150,
                                      mc_sims=100, seed_start=start)["per_seed_diffs"]
    tt = paired_t_test(diffs)
    br = bust_rate(tr.qnet, n_seeds=50, n_hands=200, mc_sims=100, seed_start=0)
    return {
        "clip": clip,
        "reward_clip": tr.reward_clip,
        "init_seed": init_seed,
        "wins": sum(1 for d in diffs if d > 0),
        "n": len(diffs),
        "mean": tt["mean"],
        "p_value": tt["p_value"],
        "bust_rate": br,
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
        by.setdefault(round(r["reward_clip"], 2), []).append(r)
    print(f"\nB3 bust-clip A/B (log-mode multi-hand, heads-up vs myopic)")
    print(f"  full ruin range -log(stack0/util_floor) = {math.log(1000/1.0):.2f}\n")
    hdr = f"  {'clip':>5} {'seed':>4} {'wins/n':>9} {'mean':>8} {'p':>8} {'bust':>6}"
    print(hdr)
    for clip in sorted(by):
        g = sorted(by[clip], key=lambda x: x["init_seed"])
        for r in g:
            p = r["p_value"]
            pstr = f"{p:.4f}" if p is not None else "  n/a"
            print(f"  {clip:>5.2f} {r['init_seed']:>4} "
                  f"{r['wins']:>4}/{r['n']:<4} {r['mean']:>+8.0f} {pstr:>8} "
                  f"{r['bust_rate']:>6.2f}")
        mw = sum(r["wins"] for r in g) / len(g)
        mm = sum(r["mean"] for r in g) / len(g)
        mb = sum(r["bust_rate"] for r in g) / len(g)
        worst = min(r["mean"] for r in g)
        print(f"  {clip:>5.2f}  AVG({len(g)})       {mw:>4.0f}/{g[0]['n']:<4} "
              f"{mm:>+8.0f} {'worst':>8} {mb:>6.2f}  (worst seed mean {worst:+.0f})\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="'old' (3.0), 'wide' (new default), or a float")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--hpe", type=int, default=15)
    ap.add_argument("--out", default=None,
                    help="append the result JSON line to this file")
    ap.add_argument("--aggregate", nargs="+", default=None,
                    help="fold these JSON-line files into a report and exit")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.aggregate)
        return

    res = run_cell(args.clip, args.seed, args.steps, args.hpe)
    line = json.dumps(res)
    print(line)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
