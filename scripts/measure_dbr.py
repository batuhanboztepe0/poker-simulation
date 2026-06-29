"""
measure_dbr.py
--------------
v2 Phase C3: detect-then-exploit on Leduc (PREREGISTRATION.md §15).

§14 handed the RNR solver the opponent's EXACT strategy, so the positive EV gain over
Nash was guaranteed by construction. This experiment removes that gift. The hero plays
Nash, OBSERVES the opponent for N hands, ESTIMATES its strategy (Dirichlet-smoothed
empirical frequencies), and only then computes RNR(estimate, p). We measure the
REALIZED exact EV against the TRUE opponent. Best-responding to a wrong estimate can do
WORSE than Nash, so a positive gain here is a genuine empirical result, not a theorem.

For each opponent we sweep observation count N, mixing parameter p, and several
observation seeds. The EV against the true opponent is exact (120-deal enumeration);
the only randomness is the N-hand observation sample, averaged over seeds.

Pre-committed reading (frozen in §15 before this runs, two-commit git-provable gap):
  G1 validity   : at the largest N, for the DETERMINISTIC opponents (station, maniac,
                  pinned by ~one visit), the mean realized gain matches the EXACT ceiling
                  (RNR handed the true opponent) within tolerance for every p. When the
                  opponent is easy to estimate, the full pipeline recovers the §14 exact
                  result at every p (and cross-checks §14). A stochastic opponent's raw
                  best response (p=1) stays data-starved at this N, which is itself G3.
  G2 positive   : at a moderate N with the conservative p=0.5, the 95% CI lower bound
                  of the mean gain is > 0 for EVERY opponent. (The reliable positive:
                  exploitation survives realistic detection.)
  G3 tradeoff   : SECONDARY. Against the stochastic non-uniform opponent at the smallest
                  N, the conservative p=0.5 has HIGHER mean realized gain than a raw best
                  response to the estimate (p=1): the data-biased response beats the raw
                  best response under estimation noise. The across-seed std is reported
                  alongside (higher at p=1 is the same overfitting, seen as variance).

    OMP_NUM_THREADS=1 python -m scripts.measure_dbr --out results/dbr_frontier.json
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import time

from src.leduc_rnr import LeducRNR, nash_strategy, ev_player0
from src.leduc_dbr import (
    detect_then_exploit, station, maniac, uniform, loose_passive,
)

# Frozen a-priori protocol (PREREGISTRATION.md §15). Iteration count, grids and seed
# count were fixed from a pre-freeze probe (convergence flat by ~800 iters; the
# overfitting signature is present on the stochastic opponent), exploratory only.
ITERS = 800                                           # CFR iters per RNR solve
CEILING_ITERS = 2000                                  # iters for the exact-ceiling solve
P_GRID = [0.5, 0.75, 1.0]                             # p=0 is Nash (gain 0 = the y baseline)
N_GRID = [12, 40, 120, 400]                           # observed hands
N_SEEDS = 6                                           # observation seeds averaged
ALPHA = 1.0                                           # Dirichlet smoothing
OPPS = {"station": station, "maniac": maniac,
        "uniform": uniform, "loose_passive": loose_passive}
DETERMINISTIC_OPPS = ["station", "maniac"]             # G1 is judged on these (easy to estimate)
STOCHASTIC_OPP = "loose_passive"                       # the opponent G3 is defined on
MODERATE_N = 40                                        # the N at which G2 is judged
G1_TOL = 0.10                                          # chips, large-N vs exact ceiling


def _ci_lo(mean, std, n):
    """95% normal CI lower bound on the mean (std is the population std across seeds)."""
    return mean - 1.96 * std / math.sqrt(n)


def _stats(xs):
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--out", default="results/dbr_frontier.json")
    args = ap.parse_args()
    t0 = time.time()

    nash_fn, _ = nash_strategy(4000)
    print("nash ready", flush=True)

    opponents = {}
    for name, opp in OPPS.items():
        ev_nash = ev_player0(nash_fn, opp)
        # exact ceiling: RNR handed the TRUE opponent (the §14 computation; for the
        # shared opponents this also cross-checks results/rnr_frontier.json).
        ceiling = {}
        for p in P_GRID:
            r = LeducRNR(opp, p); r.train(CEILING_ITERS)
            ceiling[p] = ev_player0(r.counter_fn(), opp) - ev_nash
        grid = {}
        for N in N_GRID:
            for p in P_GRID:
                gains, expls, seens = [], [], []
                for s in range(args.seeds):
                    r = detect_then_exploit(opp, N, s, nash_fn, p, args.iters, ALPHA)
                    gains.append(r["ev_vs_true"] - ev_nash)
                    expls.append(r["exploitability"])
                    seens.append(r["infosets_seen"])
                gm, gs = _stats(gains)
                grid[f"N{N}_p{p}"] = {
                    "N": N, "p": p, "mean_gain": gm, "std_gain": gs,
                    "ci_lo": _ci_lo(gm, gs, args.seeds),
                    "min_gain": min(gains), "max_gain": max(gains),
                    "mean_expl": sum(expls) / len(expls),
                    "mean_infosets_seen": sum(seens) / len(seens),
                }
                print(f"  {name:13s} N={N:4d} p={p:.2f}  gain {gm:+.3f}"
                      f"  (CIlo {grid[f'N{N}_p{p}']['ci_lo']:+.3f}, std {gs:.3f})"
                      f"  ceiling {ceiling[p]:+.3f}", flush=True)
        opponents[name] = {"ev_nash_vs_opp": ev_nash, "ceiling": ceiling, "grid": grid}

    bigN = max(N_GRID)
    # G1: large-N recovers the exact ceiling at every p, for the DETERMINISTIC opponents.
    g1 = all(abs(opponents[name]["grid"][f"N{bigN}_p{p}"]["mean_gain"]
                 - opponents[name]["ceiling"][p]) < G1_TOL
             for name in DETERMINISTIC_OPPS for p in P_GRID)
    # G2: at MODERATE_N, p=0.5, the 95% CI lower bound is positive for every opponent.
    g2 = all(o["grid"][f"N{MODERATE_N}_p0.5"]["ci_lo"] > 0 for o in opponents.values())
    # G3 (secondary): on the stochastic opponent at the smallest N, the conservative
    # p=0.5 has higher mean realized gain than the raw best response p=1.0.
    smallN = min(N_GRID)
    so = opponents[STOCHASTIC_OPP]["grid"]
    g3 = so[f"N{smallN}_p0.5"]["mean_gain"] > so[f"N{smallN}_p1.0"]["mean_gain"]

    verdict = ("RELIABLE POSITIVE: detect-then-exploit beats Nash (G2) and recovers the "
               "exact frontier with data (G1)" if (g1 and g2)
               else "see per-opponent detail (G1=%s G2=%s)" % (g1, g2))

    out = {
        "protocol": {
            "method": "detect-then-exploit (observe N hands, estimate, RNR on estimate)",
            "iters": args.iters, "ceiling_iters": CEILING_ITERS, "p_grid": P_GRID,
            "n_grid": N_GRID, "n_seeds": args.seeds, "alpha": ALPHA,
            "opponents": list(OPPS), "deterministic_opps": DETERMINISTIC_OPPS,
            "moderate_N": MODERATE_N, "g1_tol": G1_TOL,
            "stochastic_opp": STOCHASTIC_OPP,
            "metric": "exact EV over 120 deals vs the TRUE opponent; gain over Nash",
            "registered_in": "PREREGISTRATION.md §15",
            "verdict_rule": ("reliable positive iff G1 (largest-N recovers the exact "
                             "ceiling at every p for the deterministic opponents) AND G2 "
                             "(moderate-N p=0.5 CI lower bound > 0 for every opponent); "
                             "G3 (conservative p beats raw BR on the stochastic opponent "
                             "at the smallest N) is a secondary observation"),
        },
        "opponents": opponents,
        "G1_large_N_recovers_ceiling": g1,
        "G2_detection_positive": g2,
        "G3_overfitting_signature": g3,
        "verdict": verdict,
        "elapsed_secs": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nG1={g1} G2={g2} G3={g3}\n{verdict}\nwrote {args.out} "
          f"({out['elapsed_secs']:.0f}s)")


if __name__ == "__main__":
    main()
