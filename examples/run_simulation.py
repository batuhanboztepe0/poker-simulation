"""
examples/run_simulation.py
--------------------------
Headless, seeded demo session for reporting.

Runs a multi-bot self-play session, prints a per-player statistics table and
final standings, verifies chip conservation, and saves the full event log to
Parquet so the run is reproducible and reportable.

Usage:
    python examples/run_simulation.py                      # defaults
    python examples/run_simulation.py --hands 1000 --bots 6 --seed 7
    python examples/run_simulation.py --fast               # no Monte Carlo (faster)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation_runner import run_session
from src.stats import session_summary

# (name, tight_threshold, aggression) — distinct playing styles.
ARCHETYPES = [
    ("TAG",      0.45, 0.70),
    ("LAG",      0.20, 0.80),
    ("Rock",     0.50, 0.25),
    ("Station",  0.15, 0.30),
    ("Balanced", 0.35, 0.50),
    ("Maniac",   0.10, 0.90),
    ("Nit",      0.55, 0.35),
    ("Shark",    0.40, 0.60),
    ("Fish",     0.25, 0.40),
]

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


def main():
    ap = argparse.ArgumentParser(description="Headless poker simulation demo.")
    ap.add_argument("--hands", type=int, default=500)
    ap.add_argument("--bots", type=int, default=4, choices=range(2, 10))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--small-blind", type=int, default=10)
    ap.add_argument("--big-blind", type=int, default=20)
    ap.add_argument("--mc-sims", type=int, default=200)
    ap.add_argument("--fast", action="store_true",
                    help="disable Monte Carlo equity (random proxy, faster)")
    args = ap.parse_args()

    configs = [
        {"name": name, "tight_threshold": tight, "aggression": aggr}
        for name, tight, aggr in ARCHETYPES[:args.bots]
    ]

    print("=" * 64)
    print("  POKER SIMULATION — headless run")
    print("=" * 64)
    print(f"  seed={args.seed}  hands={args.hands}  bots={args.bots}  "
          f"blinds={args.small_blind}/{args.big_blind}  "
          f"mode={'fast' if args.fast else f'MC×{args.mc_sims}'}")
    print("-" * 64)

    result, collector = run_session(
        configs, n_hands=args.hands, seed=args.seed,
        small_blind=args.small_blind, big_blind=args.big_blind,
        mc_simulations=args.mc_sims, fast_mode=args.fast,
    )
    summary = session_summary(result)

    print(f"  Hands actually played: {result.n_hands}\n")
    print(f"  {'Player':10s} {'style(t/a)':11s} {'VPIP':>6s} {'AF':>6s} "
          f"{'SD%':>6s} {'EV/hand':>9s} {'net':>8s}")
    print("  " + "-" * 60)
    for snap in sorted(result.players,
                       key=lambda s: result.net_chips(s.player_id),
                       reverse=True):
        row = summary[snap.player_id]
        print(f"  {snap.name:10s} "
              f"{snap.tight_threshold:.2f}/{snap.aggression:.2f}  "
              f"{row['vpip']:>6.2f} {row['aggression_frequency']:>6.2f} "
              f"{row['showdown_win_rate']:>6.2f} "
              f"{row['chip_ev_per_hand']:>+9.2f} {row['net_chips']:>+8d}")

    print("\n  Final standings:")
    for rank, snap in enumerate(
            sorted(result.players, key=lambda s: s.final_stack, reverse=True), 1):
        print(f"    {rank}. {snap.name:10s} {snap.final_stack:>7d} chips")

    total_start = sum(result.starting_stacks.values())
    total_end = sum(result.final_stacks.values())
    ok = total_start == total_end
    print("\n  " + "-" * 60)
    print(f"  Chip conservation: {'OK' if ok else 'FAILED'} "
          f"({total_start} == {total_end})")

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"session_{args.seed}.parquet")
    collector.save_parquet(path)
    print(f"  Event log ({len(collector.events)} events) saved to: {path}")
    print("=" * 64)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
