"""
EXPLORATORY premise probe for the §14 reliable-positive design (NOT confirmatory,
NOT committed as a result). Question: does a belief-conditioned-equity exploiter
(mechanism A: HMM belief, equity vs the tilt-widened opponent range) beat a
NON-modeling baseline, and a modeling-but-tilt-frozen control, against the tilting
opponent, at decent power? This calibrates the §14 effect size, noise floor, and
which baseline contrast is the cleanest, BEFORE anything is frozen. It tunes nothing.

  OMP_NUM_THREADS=1 python -m scripts.probe_v14_premise --seeds 60 --hands 150
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse

from src.player import BotPlayer
from src.adaptive_agent import AdaptiveBotPlayer
from src.monte_carlo import MonteCarloEngine
from src.opponent_model import BeliefState, HMMBeliefState
from src.evaluation import evaluate_matchup, bootstrap_ci
from src.stats import binomial_sign_test

SHARP = dict(mu_normal=0.25, mu_tilted=0.92, recover=0.05)


def _mc(sims):
    return MonteCarloEngine(n_simulations=sims)


def hero(kind, mc_sims):
    def f(pid, stack):
        if kind == "no_belief":          # non-modeling baseline (vanilla EV)
            return BotPlayer(pid, "no_belief", stack, tight_threshold=0.2,
                             aggression=0.5, mc_engine=_mc(mc_sims))
        if kind == "static_belief":      # models a looseness range, no tilt response
            return BotPlayer(pid, "static", stack, tight_threshold=0.2, aggression=0.5,
                             mc_engine=_mc(mc_sims), belief_state=BeliefState())
        if kind == "hmm_live":           # mechanism A: best-response to tilt-widened range
            return BotPlayer(pid, "hmm", stack, tight_threshold=0.2, aggression=0.5,
                             mc_engine=_mc(mc_sims), belief_state=HMMBeliefState(**SHARP))
        raise ValueError(kind)
    return f


def opp(mc_sims):
    def f(pid, stack):
        return AdaptiveBotPlayer(pid, "Tilt", stack, mode="tilt", mc_engine=_mc(mc_sims))
    return f


def arm(kind, seeds, hands, mc_sims):
    return evaluate_matchup(hero(kind, mc_sims), opp(mc_sims), kind, "Tilt", seeds,
                            n_hands=hands, mirror=True, luck_adjusted=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--hands", type=int, default=150)
    ap.add_argument("--mc-sims", type=int, default=100)
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    arms = {k: arm(k, seeds, args.hands, args.mc_sims)
            for k in ("no_belief", "static_belief", "hmm_live")}
    print(f"\nvs AdaptiveBotPlayer(tilt), {args.seeds} mirror seeds x {args.hands} hands, "
          f"all-in-EV adjusted:")
    for k, mr in arms.items():
        ci = bootstrap_ci(mr.net_a)
        print(f"  {k:14s} net {ci['mean']:+8.1f}  CI [{ci['lo']:+.0f}, {ci['hi']:+.0f}]")

    def paired(a, b):
        d = [x - y for x, y in zip(arms[a].net_a, arms[b].net_a)]
        ci = bootstrap_ci(d); st = binomial_sign_test(d)
        sd = (sum((x - ci['mean']) ** 2 for x in d) / max(1, len(d) - 1)) ** 0.5
        res = "POSITIVE" if ci['lo'] > 0 else ("NEGATIVE" if ci['hi'] < 0 else "null")
        print(f"  {a} - {b}: {ci['mean']:+.1f}  CI [{ci['lo']:+.0f}, {ci['hi']:+.0f}]  "
              f"sign {st['wins']}-{st['losses']} p={st['p_value']:.3f}  SD={sd:.0f}  -> {res}")

    print("\npaired deltas (the §14 candidate contrasts):")
    paired("hmm_live", "no_belief")       # exploiter vs non-modeler (big, reliable?)
    paired("hmm_live", "static_belief")   # pure tilt response (small/noisy in §13-land)
    paired("static_belief", "no_belief")  # value of range-modeling alone


if __name__ == "__main__":
    main()
