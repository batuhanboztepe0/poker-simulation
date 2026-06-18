"""
probe_tilt_detection.py
-----------------------
Quantify the hand-boundary PnL->tilt belief feed: does feeding the opponent's
realised per-hand PnL into the HMM regime transition detect tilt EARLIER and more
sharply than the aggression-emission-only detector?

Method (apples-to-apples on the IDENTICAL game): a myopic hero (decisions do NOT
use any belief, so the match trajectory is fixed) carries two passive shadow
detectors updated from the same engine observation stream -- one with the PnL
feed live (`use_pnl=True`), one without (`use_pnl=False`, the dormant pre-fix
detector). The opponent is an AdaptiveBotPlayer(tilt) whose hidden regime
(`is_tilted`) is the ground truth. Over many seeded matches we measure, per
detector:

  - separation  = mean p_tilted | tilted  -  mean p_tilted | calm   (current hand)
  - corr        = corr(p_tilted, is_tilted)                          (current hand)
  - onset lead  = mean p_tilted at the hand a calm->tilted switch happens
                  (the PnL detector reads the very loss that caused the tilt, so
                  it should already be elevated at onset; the emission detector
                  has seen no aggressive actions yet)

Run: python -m scripts.probe_tilt_detection [--seeds 40] [--hands 80]
"""

import argparse
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.player import BotPlayer
from src.adaptive_agent import AdaptiveBotPlayer
from src.game import GameEngine
from src.monte_carlo import MonteCarloEngine
from src.opponent_model import HMMBeliefState

SHARP = dict(mu_normal=0.25, mu_tilted=0.92, recover=0.05)  # the --belief-sharp config


class ProbeHero(BotPlayer):
    """
    Myopic decision-maker (belief_state=None -> vanilla MC equity, so its play is
    independent of the detectors) that carries two passive shadow beliefs and
    updates both from the engine's observation stream.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, belief_state=None, **kwargs)
        self.b_pnl = HMMBeliefState(use_pnl=True, **SHARP)
        self.b_nopnl = HMMBeliefState(use_pnl=False, **SHARP)

    def observe_action(self, obs):
        n_agg = 1 if obs.get("is_aggressive") else 0
        for b in (self.b_pnl, self.b_nopnl):
            b.update(action=obs.get("action"), n_aggressive=n_agg,
                     n_actions=1, delta_stack=0)

    def observe_hand_result(self, obs):
        for b in (self.b_pnl, self.b_nopnl):
            b.observe_pnl(obs.get("delta_stack", 0))


def _corr(xs, ys):
    n = len(xs)
    if n == 0:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / ((vx * vy) ** 0.5) if vx > 0 and vy > 0 else 0.0


def run_seed(seed, n_hands, mc_sims):
    rng = random.Random(seed)
    mc = MonteCarloEngine(n_simulations=mc_sims, rng=rng)
    hero = ProbeHero(1, "Hero", 1000, tight_threshold=0.2, aggression=0.5,
                     mc_engine=mc, rng=rng)
    villain = AdaptiveBotPlayer(2, "Tilt", 1000, mode="tilt",
                                mc_engine=mc, rng=rng)
    engine = GameEngine([hero, villain], 10, 20, verbose=False, rng=rng)

    truth, p_pnl, p_no = [], [], []
    prev_truth = False
    onset_pnl, onset_no = [], []
    for _ in range(n_hands):
        if sum(1 for p in (hero, villain) if p.stack > 0) < 2:
            break
        engine.play_hand()
        t = 1 if villain.is_tilted else 0          # regime DURING this hand
        vp, vn = hero.b_pnl.p_tilted(), hero.b_nopnl.p_tilted()
        truth.append(t)
        p_pnl.append(vp)
        p_no.append(vn)
        if t == 1 and not prev_truth:              # calm -> tilted onset
            onset_pnl.append(vp)
            onset_no.append(vn)
        prev_truth = bool(t)
    return truth, p_pnl, p_no, onset_pnl, onset_no


def _sep(truth, p):
    tilted = [v for t, v in zip(truth, p) if t == 1]
    calm = [v for t, v in zip(truth, p) if t == 0]
    mt = sum(tilted) / len(tilted) if tilted else 0.0
    mc = sum(calm) / len(calm) if calm else 0.0
    return mt, mc, mt - mc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--hands", type=int, default=80)
    ap.add_argument("--mc-sims", type=int, default=100)
    args = ap.parse_args()

    truth, p_pnl, p_no = [], [], []
    onset_pnl, onset_no = [], []
    # Lead-by-one alignment, kept per match so we never pair across seed
    # boundaries: detector value at END of hand n vs the opponent's regime in
    # hand n+1 (the operational question -- what does a decision-maker know at the
    # START of the next hand, before seeing any of its actions?).
    lead_pnl, lead_no, lead_truth = [], [], []
    for s in range(args.seeds):
        t, vp, vn, op, on = run_seed(s, args.hands, args.mc_sims)
        truth += t
        p_pnl += vp
        p_no += vn
        onset_pnl += op
        onset_no += on
        lead_pnl += vp[:-1]
        lead_no += vn[:-1]
        lead_truth += [float(x) for x in t[1:]]

    n_tilted = sum(truth)
    print(f"seeds={args.seeds} hands<={args.hands}  "
          f"hand-rows={len(truth)}  tilted-rows={n_tilted} "
          f"({100*n_tilted/max(1,len(truth)):.0f}%)  onsets={len(onset_pnl)}")
    print()
    for label, p in (("PnL feed  (use_pnl=True )", p_pnl),
                     ("emission  (use_pnl=False)", p_no)):
        mt, mc, sep = _sep(truth, p)
        print(f"  {label}:  p_tilted|tilted={mt:.3f}  p_tilted|calm={mc:.3f}  "
              f"separation={sep:+.3f}  corr={_corr(p, [float(t) for t in truth]):+.3f}")
    if onset_pnl:
        print()
        print(f"  onset p_tilted (calm->tilted hand):  "
              f"PnL={sum(onset_pnl)/len(onset_pnl):.3f}  "
              f"emission={sum(onset_no)/len(onset_no):.3f}  "
              f"(higher = detects tilt the hand it starts)")
    print()
    print("  LEAD-BY-ONE (end-of-hand belief vs NEXT hand's regime, "
          "pre-emission):")
    for label, p in (("PnL feed ", lead_pnl), ("emission ", lead_no)):
        sep = (_sep([int(t) for t in lead_truth], p)[2])
        print(f"    {label}:  separation={sep:+.3f}  "
              f"corr={_corr(p, lead_truth):+.3f}")


if __name__ == "__main__":
    main()
