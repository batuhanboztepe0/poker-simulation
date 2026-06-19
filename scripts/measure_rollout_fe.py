"""
measure_rollout_fe.py
---------------------
Phase B1: does WARMING the rollout's opponent belief rescue fold-equity?

The depth-1 rollout can value a bluff via a FoldEquityModel, but with a COLD /
GTO-neutral belief it over-bluffs foes that don't fold at b/(pot+b) and loses
badly (RL_HANDOFF §10). B1 shares the bot's engine-updated per-opponent belief
(A4's belief_factory) into the policy AND its fold-equity model so p_fold
becomes opponent-specific once warm. This script measures three rollout configs
-- no fold-equity, cold fold-equity, warm fold-equity (B1) -- heads-up vs a
roster of opponents over paired seeds.

Finding (60 seeds x 200 hands, net chips of the rollout):
    config        myopic   tilt   random    nit(foldy)
    no_fe           +23    +357     +125        +1826
    cold_fe       -1133    -333      -67        +2000
    warm_fe (B1)    -67    +133     -200         -333
Warming FIXES the cold-FE over-bluff disaster vs call-happy opponents
(myopic -1133->-67, tilt -333->+133), but it is NOT a uniform win: vs a foldy
nit the warmed belief mis-prices bluffs against the nit's strong continuing
range and underperforms even cold-FE, and the plain no-FE value rollout
dominates the roster. So fold-equity stays OFF by default; B1 makes warmed-FE
SAFE (break-even) rather than catastrophic when it is used.

Run thread-pinned: OMP_NUM_THREADS=1 python -m scripts.measure_rollout_fe
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import random

from src.monte_carlo import MonteCarloEngine
from src.stochastic_control import RolloutBotPlayer, RolloutPolicy
from src.fold_equity import FoldEquityModel
from src.opponent_model import BeliefState
from src.adaptive_agent import AdaptiveBotPlayer
from src.player import BotPlayer
from src.evaluation import evaluate_matchup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--hands", type=int, default=200)
    ap.add_argument("--mc-sims", type=int, default=100)
    args = ap.parse_args()

    seeds = list(range(args.seeds))

    def mc():
        return MonteCarloEngine(n_simulations=args.mc_sims, rng=random.Random(0))

    def no_fe(pid, stack):
        return RolloutBotPlayer(pid, "Rollout", stack, mc_engine=mc(),
                                rollout_policy=RolloutPolicy(mc()))

    def cold_fe(pid, stack):
        return RolloutBotPlayer(pid, "Rollout", stack, mc_engine=mc(),
                                rollout_policy=RolloutPolicy(
                                    mc(), fold_equity_model=FoldEquityModel()))

    def warm_fe(pid, stack):  # B1: belief shared into policy + fold-equity model
        return RolloutBotPlayer(pid, "Rollout", stack, mc_engine=mc(),
                                belief_factory=lambda oid: BeliefState(),
                                rollout_policy=RolloutPolicy(
                                    mc(), fold_equity_model=FoldEquityModel()))

    opponents = {
        "myopic": lambda pid, s: BotPlayer(pid, "Myopic", s, tight_threshold=0.2,
                                           aggression=0.5, mc_engine=mc()),
        "tilt":   lambda pid, s: AdaptiveBotPlayer(pid, "Tilt", s, mode="tilt",
                                                   mc_engine=mc()),
        "random": lambda pid, s: AdaptiveBotPlayer(pid, "Random", s,
                                                   mode="random", mc_engine=mc()),
        "nit":    lambda pid, s: BotPlayer(pid, "Nit", s, tight_threshold=0.6,
                                           aggression=0.1, mc_engine=mc()),
    }
    configs = [("no_fe", no_fe), ("cold_fe", cold_fe), ("warm_fe(B1)", warm_fe)]

    print(f"{len(seeds)} paired seeds x {args.hands} hands "
          f"(net chips of the rollout vs each opponent)\n")
    print(f"{'config':14s}" + "".join(f"{o:>16s}" for o in opponents))
    for cname, cfac in configs:
        cells = []
        for oname, ofac in opponents.items():
            mr = evaluate_matchup(cfac, ofac, cname, oname, seeds,
                                  n_hands=args.hands)
            p = mr.t_test["p_value"]
            ptxt = f"{p:.3f}" if p is not None else "n/a"
            cells.append(f"{mr.mean_diff:+7.0f}(p={ptxt})")
        print(f"{cname:14s}" + "".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
