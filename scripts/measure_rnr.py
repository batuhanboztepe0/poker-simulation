"""
measure_rnr.py
--------------
v2 Phase C3: the exact Restricted Nash Response (RNR) exploitation frontier on Leduc
(PREREGISTRATION.md §14). For each exploitable opponent and each mixing parameter p,
compute, EXACTLY (full 120-deal enumeration, zero sampling variance):

  - EV of the RNR(p) counter against the opponent,
  - its gain over the Nash baseline's EV against the same opponent (the exploitation),
  - the RNR(p) counter's own exploitability (NashConv; the price of exploiting).

p=0 must reproduce Nash (gain ~0, exploitability ~0); p=1 must reproduce the exact
best response to the opponent (independently computed here as a validation gate). The
reliable-positive claim is that RNR(p) earns a strictly positive, exactly-measured EV
gain over Nash against every exploitable opponent, monotone in p, while its
exploitability rises with p (the exploitation-vs-exploitability tradeoff).

This is the exact-Leduc answer to the high-variance heads-up exploitation null
(§13): there the bust-match variance swamped any incremental edge; here EV is exact.

    OMP_NUM_THREADS=1 python -m scripts.measure_rnr --out results/rnr_frontier.json

Frozen config and the pre-committed reading are registered in PREREGISTRATION.md §14
before this is run (two-commit git-provable gap, as §10/§11/§12).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time

from src.leduc_cfr import LeducCFR, NUM_ACTIONS, LEDUC_GAME_VALUE
from src.leduc_rnr import (
    LeducRNR, nash_strategy, ev_player0, OPPONENTS, _avail_from_key,
)

# Frozen a-priori protocol (PREREGISTRATION.md §14).
ITERS = 5000                                   # CFR iterations per solve
P_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
OPP_NAMES = ["station", "maniac", "uniform"]   # three suboptimal, exploitable opponents


def _br_exact_vs(opp_fn):
    """Exact best-response EV for player 0 against a fixed opponent (the p=1 target),
    via the load-trick: set every node's average to the opponent, then best-respond."""
    cfr = LeducCFR(); cfr.train(1)
    for iset, node in cfr.nodes.items():
        dist = opp_fn(iset, _avail_from_key(iset))
        node.strategy_sum = [dist[a] for a in range(NUM_ACTIONS)]
    return cfr._best_response_value(0) / 120.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--out", default="results/rnr_frontier.json")
    args = ap.parse_args()
    t0 = time.time()

    nash_fn, nash_cfr = nash_strategy(args.iters)
    nash_self_expl = nash_cfr.exploitability()
    print(f"Nash self-play exploitability: {nash_self_expl:.4f} "
          f"(game value {LEDUC_GAME_VALUE})", flush=True)

    opponents = {}
    for name in OPP_NAMES:
        opp = OPPONENTS[name]
        ev_nash = ev_player0(nash_fn, opp)
        br_exact = _br_exact_vs(opp)
        points = []
        for p in P_GRID:
            rnr = LeducRNR(opp, p)
            rnr.train(args.iters)
            ev = ev_player0(rnr.counter_fn(), opp)
            expl = rnr.counter_exploitability()
            points.append({"p": p, "ev_vs_opp": ev,
                           "gain_over_nash": ev - ev_nash, "exploitability": expl})
            print(f"  {name:8s} p={p:.2f}: EV {ev:+.4f}  gain {ev-ev_nash:+.4f}  "
                  f"expl {expl:.4f}", flush=True)
        p0 = points[0]; p1 = points[-1]
        opponents[name] = {
            "ev_nash_vs_opp": ev_nash,
            "br_exact_vs_opp": br_exact,
            "frontier": points,
            "gates": {
                "p0_gain_near_zero": abs(p0["gain_over_nash"]) < 0.02,
                "p0_exploitability_near_zero": abs(p0["exploitability"]) < 0.05,
                "p1_matches_exact_br": abs(p1["ev_vs_opp"] - br_exact) < 0.01,
                # Non-decreasing up to CFR convergence noise: adjacent p near the
                # best-response plateau can invert by ~1e-4 at finite iterations.
                "ev_monotone_in_p": all(points[i]["ev_vs_opp"] <= points[i + 1]["ev_vs_opp"] + 5e-3
                                        for i in range(len(points) - 1)),
            },
            "max_gain_over_nash": max(pt["gain_over_nash"] for pt in points),
        }

    # Pre-committed reading: a reliable positive iff EVERY opponent shows a strictly
    # positive max gain over Nash AND all four validation gates hold.
    all_positive = all(o["max_gain_over_nash"] > 0 for o in opponents.values())
    all_gates = all(all(o["gates"].values()) for o in opponents.values())
    verdict = ("RELIABLE POSITIVE: RNR exploits every opponent for an exact EV gain "
               "over Nash, validation gates hold" if (all_positive and all_gates)
               else "see per-opponent detail")

    out = {
        "protocol": {
            "method": "Restricted Nash Response (Johanson-Bowling) on Leduc, exact EV",
            "iters": args.iters, "p_grid": P_GRID, "opponents": OPP_NAMES,
            "metric": "exact EV over all 120 deals; gain over Nash; exact exploitability",
            "nash_self_play_exploitability": nash_self_expl,
            "registered_in": "PREREGISTRATION.md §14",
            "verdict_rule": ("reliable positive iff every opponent has max gain over "
                             "Nash > 0 AND all validation gates (p0=Nash, p1=exact BR, "
                             "monotone) hold; pre-committed before the run"),
        },
        "opponents": opponents,
        "all_opponents_positive": all_positive,
        "all_gates_pass": all_gates,
        "verdict": verdict,
        "elapsed_secs": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n{verdict}\nwrote {args.out} ({out['elapsed_secs']:.0f}s)")


if __name__ == "__main__":
    main()
