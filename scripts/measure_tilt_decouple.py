"""
measure_tilt_decouple.py
------------------------
Phase B5: can the gain-only tilt-bonus and the PnL->tilt belief feed COEXIST once
the bonus is decoupled from the learner's own realised PnL?

Background (RL_HANDOFF thread (d) / footgun): stacking the PnL feed with the NAIVE
tilt-bonus COLLAPSES the policy (zero-sum: a learner win == the opponent's loss
spikes p_tilted on the same hand the bonus then amplifies -> it amplifies the
learner's own wins, a reward distortion). PnL-feature and naive-bonus are
SUBSTITUTES for the tilt edge, not complements. B5 scales the bonus by p_tilted
ENTERING the hand (`tilt_bonus_decouple`), removing that same-hand coupling.

This is the honest A/B over four configs (all belief-conditioned, sharp HMM,
opponent-mix, multi-hand chips -- the §10 generalist recipe), each trained then
evaluated on held-out seeds vs the pool {myopic, tilt, random}:

  pnl_nobonus   PnL feed on,  no bonus           (the substitute baseline)
  nopnl_bonus   PnL feed off, naive bonus        (the other substitute)
  pnl_naive     PnL feed on,  naive bonus        (the footgun: expect collapse)
  pnl_decouple  PnL feed on,  DECOUPLED bonus    (B5: expect no collapse)

FINDING (NEGATIVE-ish; 3 init seeds x 30 held-out x 150 hands, 6000 steps):
    config        myopic   tilt   random
    pnl_nobonus     +200   +248    +300   <- best across the board
    nopnl_bonus     -160   -122    -195
    pnl_naive        +50   +141     -54   <- footgun (drags vs pnl_nobonus)
    pnl_decouple    +246   +164      +3
The decouple WORKS at its narrow goal: pnl_decouple beats the footgun pnl_naive
on all three opponents and never collapses. But it does NOT beat the PnL feature
ALONE (worse on tilt/random; the `random` column -- where there is no real tilt
to exploit -- shows the bonus only adds variance: +3 vs +300). So the feature and
the bonus stay SUBSTITUTES, not complements: the decouple makes them SAFE to
combine, not BENEFICIAL. PnL-feature-alone remains the recipe; the decouple is an
opt-in safety mechanism, not a headline win. (Per-seed variance is large at n=3.)

Run ONE (config, init-seed) cell -> JSON line (launch cells thread-pinned in
parallel; the small MLP only thrashes under parallel BLAS):

    OMP_NUM_THREADS=1 python -m scripts.measure_tilt_decouple --config pnl_decouple --seed 0

`--aggregate f1.json ...` folds the JSON lines into a per-opponent table.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json

import torch

import src.rl_agent as rl
from src.opponent_model import HMMBeliefState
from src.player import BotPlayer
from src.adaptive_agent import AdaptiveBotPlayer
from src.monte_carlo import MonteCarloEngine
from src.evaluation import evaluate_matchup


SHARP = dict(mu_normal=0.25, mu_tilted=0.92, recover=0.05)

# config -> (use_pnl, tilt_reward_bonus, tilt_bonus_decouple)
CONFIGS = {
    "pnl_nobonus":  (True,  0.0, False),
    "nopnl_bonus":  (False, 0.6, False),
    "pnl_naive":    (True,  0.6, False),
    "pnl_decouple": (True,  0.6, True),
}


def _mc(sims):
    return MonteCarloEngine(n_simulations=sims)


def train_config(config, init_seed, steps, mc_sims):
    use_pnl, bonus, decouple = CONFIGS[config]
    torch.manual_seed(init_seed)

    def myopic_f(pid, s, mc, rng):
        return BotPlayer(pid, "Myopic", s, tight_threshold=0.2, aggression=0.5,
                         mc_engine=mc, rng=rng)

    def tilt_f(pid, s, mc, rng):
        return AdaptiveBotPlayer(pid, "Tilt", s, mode="tilt", mc_engine=mc, rng=rng)

    def rand_f(pid, s, mc, rng):
        return AdaptiveBotPlayer(pid, "Random", s, mode="random", mc_engine=mc,
                                 rng=rng)

    bk = dict(SHARP, use_pnl=use_pnl)
    tr = rl.SelfPlayTrainer(
        n_players=2, seed=1, opponent_mode="fixed", mc_sims=mc_sims,
        epsilon_start=1.0, epsilon_end=0.05, gamma=0.99,
        multi_hand=True, hands_per_episode=15, reward_mode="chips",
        extended_features=True, feature_mode="belief",
        learner_belief_factory=lambda: HMMBeliefState(**bk),
        opponent_factories=[myopic_f, tilt_f, rand_f],
        tilt_reward_bonus=bonus, tilt_bonus_decouple=decouple)
    tr.train(steps, batch_size=64, refresh_every=5, hands_per_refresh=20)
    return tr.qnet, bk


def eval_vs_pool(qnet, bk, seeds, hands, mc_sims):
    def rl_f(pid, stack):
        return rl.RLBotPlayer(pid, "RL", stack, qnet=qnet, epsilon=0.0,
                              training=False, feature_mode="belief",
                              belief_state=HMMBeliefState(**bk),
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
    return out


def run_cell(config, init_seed, steps, hands, mc_sims, eval_seeds, eval_start):
    qnet, bk = train_config(config, init_seed, steps, mc_sims)
    seeds = list(range(eval_start, eval_start + eval_seeds))
    means = eval_vs_pool(qnet, bk, seeds, hands, mc_sims)
    return {"config": config, "init_seed": init_seed, **means}


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
    print("\nB5 tilt-bonus decouple A/B (RL mean net chips vs the held-out pool)\n")
    print(f"  {'config':>13} {'seed':>4} {'myopic':>8} {'tilt':>8} {'random':>8}")
    order = [c for c in CONFIGS if c in by] + [c for c in by if c not in CONFIGS]
    for config in order:
        g = sorted(by[config], key=lambda x: x["init_seed"])
        for r in g:
            print(f"  {config:>13} {r['init_seed']:>4} {r['myopic']:>+8.0f} "
                  f"{r['tilt']:>+8.0f} {r['random']:>+8.0f}")
        n = len(g)
        mm = {k: sum(r[k] for r in g) / n for k in ("myopic", "tilt", "random")}
        print(f"  {config:>13}  AVG  {mm['myopic']:>+8.0f} {mm['tilt']:>+8.0f} "
              f"{mm['random']:>+8.0f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--hands", type=int, default=150)
    ap.add_argument("--mc-sims", type=int, default=100)
    ap.add_argument("--eval-seeds", type=int, default=30)
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
