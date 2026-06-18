"""
eval_belief_rl.py
-----------------
Held-out PnL evaluation of a belief-conditioned RL checkpoint against the
opponent pool it was trained to beat: a myopic baseline, a tilting opponent, and
a random-aggression opponent. Mirrors the dashboard's RL factory (rebuilds the
HMM detector from the checkpoint's stored `belief_kwargs`, so use_pnl is honoured)
and reuses `evaluate_matchup` so the seeds are paired and chip conservation is
asserted per match.

Used to A/B the hand-boundary PnL->tilt belief feed: train two otherwise-identical
checkpoints (use_pnl on vs off) and compare mean RL net chips vs each opponent.

Run: python -m scripts.eval_belief_rl models/rl_pnl.pt [--seeds 50] [--hands 200]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rl_agent import load_checkpoint, RLBotPlayer
from src.opponent_model import HMMBeliefState
from src.player import BotPlayer
from src.adaptive_agent import AdaptiveBotPlayer
from src.monte_carlo import MonteCarloEngine
from src.evaluation import evaluate_matchup


def _mc(sims):
    return MonteCarloEngine(n_simulations=sims)


def rl_factory(qnet, fm, belief_kwargs, sims):
    def _f(pid, stack):
        belief = HMMBeliefState(**belief_kwargs) if fm == "belief" else None
        return RLBotPlayer(pid, "RL", stack, qnet=qnet, epsilon=0.0,
                           training=False, feature_mode=fm,
                           belief_state=belief, mc_engine=_mc(sims))
    return _f


def opponent_factories(sims):
    return {
        "Myopic": lambda pid, stack: BotPlayer(
            pid, "Myopic", stack, tight_threshold=0.2, aggression=0.5,
            mc_engine=_mc(sims)),
        "Tilt": lambda pid, stack: AdaptiveBotPlayer(
            pid, "Tilt", stack, mode="tilt", mc_engine=_mc(sims)),
        "Random": lambda pid, stack: AdaptiveBotPlayer(
            pid, "Random", stack, mode="random", mc_engine=_mc(sims)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed-start", type=int, default=1000)  # held out from training
    ap.add_argument("--hands", type=int, default=200)
    ap.add_argument("--mc-sims", type=int, default=100)
    args = ap.parse_args()

    qnet, ckpt = load_checkpoint(args.checkpoint)
    fm = ckpt.get("feature_mode", "base")
    bk = (ckpt.get("meta") or {}).get("belief_kwargs") or {}
    use_pnl = bk.get("use_pnl", "n/a")
    print(f"checkpoint: {args.checkpoint}")
    print(f"  feature_mode={fm}  belief_kwargs={bk}")
    print(f"  use_pnl={use_pnl}  seeds={args.seeds} (start {args.seed_start}) "
          f"hands={args.hands}")
    print()

    rl = rl_factory(qnet, fm, bk, args.mc_sims)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    print(f"  {'opponent':10s}  {'RL mean net':>12s}  {'wins':>8s}  "
          f"{'t':>6s}  {'p':>7s}")
    for name, opp in opponent_factories(args.mc_sims).items():
        mr = evaluate_matchup(rl, opp, "RL", name, seeds,
                              n_hands=args.hands)
        rl_mean = sum(mr.net_a) / len(mr.net_a)
        tt = mr.t_test
        print(f"  {name:10s}  {rl_mean:>+12.0f}  {mr.wins_a:>3d}/{len(seeds):<4d}  "
              f"{tt['t']:>+6.2f}  {tt['p_value']:>7.4f}")


if __name__ == "__main__":
    main()
