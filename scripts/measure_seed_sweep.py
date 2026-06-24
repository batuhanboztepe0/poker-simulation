"""
measure_seed_sweep.py
---------------------
Phase 0 (v2): multi-seed training robustness for the pre-registered confirmatory
headline.

The committed confirmatory (PREREGISTRATION.md §4.3/§4.6, results/confirmatory.json)
fixes a single training seed, `torch_seed=0`, and reports +256 chips/match. §0 of
the pre-registration flags this as an open limitation: the sweep that would show
seed 0 was not chosen after seeing a favorable result is not committed, so
seed-selection independence cannot be externally verified.

This script closes that limitation. It retrains the RL agent over
`torch_seed in range(N)` with the recipe BYTE-IDENTICAL to the confirmatory
(imported directly from scripts.measure_confirmatory.train_pilot_agent), runs the
SAME frozen confirmatory evaluation per seed (evaluate_matchup, seeds 0..499,
n_hands=100, mirror=True, luck_adjusted=False, Myopic baseline), and reports the
edge as a DISTRIBUTION across training seeds, not a single point.

Only `torch_seed` varies. The training recipe, the eval call, the eval seed block
(0..499), and the opponent are all held identical to the confirmatory, so the
seed-0 arm must reproduce the committed +256 exactly. That reproduction is reported
as a correctness gate. Because the eval block is constant across all arms, its
fixed seed-range overlap (PREREG §4.6) cannot explain cross-seed variation.

The protocol is frozen in PREREGISTRATION.md §10 BEFORE this is run. Results are
reported in full regardless of direction (PREREG §8): a fragile, seed-dependent
edge is an honest finding, not a failure.

    OMP_NUM_THREADS=1 python -m scripts.measure_seed_sweep --out results/seed_sweep.json

Run thread-pinned (small MLP; parallel BLAS only thrashes).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time

from src.player import BotPlayer
from src.monte_carlo import MonteCarloEngine
from src.evaluation import evaluate_matchup, bootstrap_ci
from src.stats import binomial_sign_test
import src.rl_agent as rl

# Import the confirmatory training recipe verbatim so the sweep cannot drift from
# it. train_pilot_agent(steps, mc_sims, hidden, torch_seed) does
# torch.manual_seed(torch_seed); SelfPlayTrainer(seed=1, opponent_mode="fixed",
# ...); train(steps, batch_size=64, refresh_every=10, hands_per_refresh=12).
from scripts.measure_confirmatory import train_pilot_agent

# The committed confirmatory headline the seed-0 arm must reproduce.
_CONFIRMATORY_PATH = "results/confirmatory.json"


def _mc(sims):
    return MonteCarloEngine(n_simulations=sims)


def run_arm(qnet, seeds, n_hands, mc_sims, mirror, keep_diffs):
    """One confirmatory evaluate_matchup arm for a trained qnet, identical to
    scripts.measure_confirmatory.run_arm."""
    def factory_rl(pid, stack):
        return rl.RLBotPlayer(pid, "RL", stack, qnet=qnet, epsilon=0.0,
                              training=False, feature_mode="base",
                              mc_engine=_mc(mc_sims))

    def factory_myopic(pid, stack):
        return BotPlayer(pid, "Myopic", stack, tight_threshold=0.2,
                         aggression=0.5, mc_engine=_mc(mc_sims))

    mr = evaluate_matchup(factory_rl, factory_myopic, "RL", "Myopic", seeds,
                          n_hands=n_hands, mirror=mirror, luck_adjusted=False)
    ci = bootstrap_ci(mr.diffs)
    binom = binomial_sign_test(mr.diffs)
    arm = {
        "mirror": mirror, "n_seeds": len(seeds), "n_hands": n_hands,
        "wins_rl": mr.wins_a, "wins_myopic": mr.wins_b, "ties": mr.ties,
        "mean_diff": mr.mean_diff, "ci95": ci,
        "paired_t": mr.t_test, "binom": binom,
        "resolved": ci["lo"] > 0 or ci["hi"] < 0,
        "edge_sign": (1 if ci["lo"] > 0 else (-1 if ci["hi"] < 0 else 0)),
    }
    if keep_diffs:
        # Keep the primary-arm diffs so every reported per-seed edge CI is
        # independently reproducible (the audit standard, cf. confirmatory.json).
        arm["per_seed_diffs"] = list(mr.diffs)
    return arm


def summarize(per_seed, committed_seed0):
    """Across-seed distribution summary. The pre-committed primary outcome
    (PREREG §10): the per-seed mirror-arm edge reported as a distribution."""
    edges = [s["mirror"]["mean_diff"] for s in per_seed]
    seeds = [s["torch_seed"] for s in per_seed]
    n = len(edges)
    srt = sorted(edges)
    median = (srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2)
    mean = sum(edges) / n
    var = sum((e - mean) ** 2 for e in edges) / (n - 1) if n > 1 else 0.0
    sd = var ** 0.5
    # Across-seed CI: bootstrap over the seed-level edges, i.e. the uncertainty in
    # the EXPECTED edge of a randomly-initialised training run. Distinct from any
    # single seed's eval CI (which is over eval seeds, not training seeds).
    across_seed_ci = bootstrap_ci(edges)

    resolved_pos = sum(1 for s in per_seed if s["mirror"]["edge_sign"] > 0)
    resolved_neg = sum(1 for s in per_seed if s["mirror"]["edge_sign"] < 0)
    unresolved = n - resolved_pos - resolved_neg

    # Seed 0's location in the distribution (was the published seed favorably
    # placed?). Percentile = fraction of seeds with an edge <= seed 0's edge.
    seed0_edge = next(s["mirror"]["mean_diff"] for s in per_seed
                      if s["torch_seed"] == 0)
    seed0_pct = sum(1 for e in edges if e <= seed0_edge) / n

    # Pre-committed verdict (PREREG §10): robust iff median > 0 AND the across-seed
    # mean-edge CI excludes zero on the positive side.
    robust = median > 0 and across_seed_ci["lo"] > 0
    verdict = "robust" if robust else "seed-dependent"

    out = {
        "n_seeds_trained": n,
        "torch_seeds": seeds,
        "per_seed_edge": dict(zip(map(str, seeds), edges)),
        "mean_edge": mean, "median_edge": median, "sd_edge": sd,
        "min_edge": min(edges), "max_edge": max(edges),
        "across_seed_ci95": across_seed_ci,
        "n_resolved_positive": resolved_pos,
        "n_resolved_negative": resolved_neg,
        "n_unresolved": unresolved,
        "seed0_edge": seed0_edge,
        "seed0_percentile": seed0_pct,
        "verdict": verdict,
        "verdict_rule": ("robust iff median per-seed edge > 0 AND across-seed 95% "
                         "bootstrap CI of the mean per-seed edge excludes zero "
                         "(positive); else seed-dependent. Pre-committed in "
                         "PREREGISTRATION.md §10 before the run."),
    }
    if committed_seed0 is not None:
        out["seed0_reproduces_committed"] = (seed0_edge == committed_seed0)
        out["committed_seed0_edge"] = committed_seed0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-seeds", type=int, default=20,
                    help="train torch_seed in range(N); frozen at 20 in PREREG §10")
    ap.add_argument("--eval-seeds", type=int, default=500)
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--mc-sims", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--no-raw", action="store_true",
                    help="skip the raw single-orientation calibration arm")
    ap.add_argument("--out", default="results/seed_sweep.json")
    args = ap.parse_args()

    committed_seed0 = None
    if os.path.exists(_CONFIRMATORY_PATH):
        with open(_CONFIRMATORY_PATH) as f:
            committed_seed0 = json.load(f)["confirmatory_primary"]["mean_diff"]

    eval_seeds = list(range(args.eval_seeds))
    per_seed = []
    t_start = time.time()

    for ts in range(args.n_train_seeds):
        t0 = time.time()
        qnet = train_pilot_agent(args.steps, args.mc_sims, args.hidden, ts)
        primary = run_arm(qnet, eval_seeds, args.hands, args.mc_sims,
                          mirror=True, keep_diffs=True)
        raw = None
        if not args.no_raw:
            raw = run_arm(qnet, eval_seeds, args.hands, args.mc_sims,
                          mirror=False, keep_diffs=False)
        row = {"torch_seed": ts, "mirror": primary, "raw": raw,
               "train_eval_secs": round(time.time() - t0, 1)}
        per_seed.append(row)

        gate = ""
        if ts == 0 and committed_seed0 is not None:
            gate = (" [REPRODUCES committed +%.0f]" % committed_seed0
                    if primary["mean_diff"] == committed_seed0
                    else " [!! seed0=%.0f != committed %.0f]"
                    % (primary["mean_diff"], committed_seed0))
        print(f"ts={ts:2d}: edge {primary['mean_diff']:+.0f} "
              f"CI [{primary['ci95']['lo']:+.0f}, {primary['ci95']['hi']:+.0f}] "
              f"resolved={primary['resolved']} "
              f"({row['train_eval_secs']:.0f}s){gate}", flush=True)

        # Incremental write so a crash mid-sweep keeps completed arms.
        partial = {
            "protocol": {
                "function": "evaluate_matchup",
                "only_varied": "torch_seed",
                "eval_seeds": f"list(range({args.eval_seeds}))",
                "n_hands": args.hands, "mirror_primary": True,
                "luck_adjusted": False,
                "training_recipe": "scripts.measure_confirmatory.train_pilot_agent",
                "steps": args.steps, "mc_sims": args.mc_sims, "hidden": args.hidden,
                "baseline": {"name": "Myopic", "tight_threshold": 0.2,
                             "aggression": 0.5, "mc_sims": args.mc_sims},
                "registered_in": "PREREGISTRATION.md §10",
            },
            "per_seed": per_seed,
            "summary": summarize(per_seed, committed_seed0),
            "elapsed_secs": round(time.time() - t_start, 1),
        }
        with open(args.out, "w") as f:
            json.dump(partial, f, indent=1)

    s = partial["summary"]
    print(f"\nSWEEP DONE: {s['n_seeds_trained']} seeds, "
          f"mean edge {s['mean_edge']:+.0f}, median {s['median_edge']:+.0f}, "
          f"SD {s['sd_edge']:.0f}, range [{s['min_edge']:+.0f}, {s['max_edge']:+.0f}]")
    print(f"  across-seed CI [{s['across_seed_ci95']['lo']:+.0f}, "
          f"{s['across_seed_ci95']['hi']:+.0f}] | resolved +{s['n_resolved_positive']} "
          f"-{s['n_resolved_negative']} ~{s['n_unresolved']}")
    print(f"  seed 0 edge {s['seed0_edge']:+.0f} at percentile "
          f"{s['seed0_percentile']:.2f} | VERDICT: {s['verdict']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
