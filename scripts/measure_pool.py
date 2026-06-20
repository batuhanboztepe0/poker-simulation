"""
measure_pool.py
---------------
Generate the data for the §10 "generalist" figure (the marquee POSITIVE result,
RL_HANDOFF §10): train the belief-conditioned, opponent-mix RL generalist — the
recipe that beats the whole {myopic, tilt, random} pool AND ranks #1 in the
static (tight × aggression) personality sweep — then dump its cross-agent
leaderboard + the fitness landscape + the RL agent's sweep rank to JSON for
scripts/make_figures.py.

Recipe mirrors `train_rl --mode fixed --belief --belief-sharp --opponent-mix
--reward-mode chips`: the belief is a FEATURE (not equity); the engine
auto-updates it via observe_action during eval. ~12k steps are needed (the
generalist is uniformly underfit at ~1k, §10).

Run thread-pinned:
    OMP_NUM_THREADS=1 python -m scripts.measure_pool --steps 12000 --out results/pool.json
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import random

import torch

import src.rl_agent as rl
from src.rl_agent import SelfPlayTrainer, RLBotPlayer
from src.opponent_model import HMMBeliefState
from src.adaptive_agent import AdaptiveBotPlayer
from src.player import BotPlayer
from src.kelly_agent import KellyBotPlayer
from src.monte_carlo import MonteCarloEngine
from src.evaluation import evaluate_roster, parameter_sweep, bootstrap_ci

SHARP = dict(mu_normal=0.25, mu_tilted=0.92, recover=0.05)


def _mc():
    return MonteCarloEngine(n_simulations=100, rng=random.Random(0))


def train_generalist(steps, torch_seed):
    """Train the §10 belief+mix generalist (mirrors train_rl's construction)."""
    torch.manual_seed(torch_seed)

    def myo(pid, s, mc, rng):
        return BotPlayer(pid, "Myopic", s, tight_threshold=0.2, aggression=0.5,
                         mc_engine=mc, rng=rng)

    def tlt(pid, s, mc, rng):
        return AdaptiveBotPlayer(pid, "Tilt", s, mode="tilt", mc_engine=mc, rng=rng)

    def rnd(pid, s, mc, rng):
        return AdaptiveBotPlayer(pid, "Random", s, mode="random", mc_engine=mc,
                                 rng=rng)

    bk = dict(SHARP, use_pnl=True)
    tr = SelfPlayTrainer(
        n_players=2, hidden=64, seed=1, opponent_mode="fixed", mc_sims=100,
        epsilon_start=1.0, epsilon_end=0.05, gamma=0.99, snapshot_every=300,
        multi_hand=True, hands_per_episode=15, reward_mode="chips",
        extended_features=True, feature_mode="belief",
        learner_belief_factory=lambda: HMMBeliefState(**bk),
        opponent_factories=[myo, tlt, rnd])
    tr.train(steps, batch_size=64, refresh_every=5, hands_per_refresh=20)
    return tr.qnet, bk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--torch-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--hands", type=int, default=120)
    ap.add_argument("--seed-start", type=int, default=10000,
                    help="held-out eval seed start (disjoint from training)")
    ap.add_argument("--out", default=None, help="write results/pool.json here")
    args = ap.parse_args()

    if not rl._HAVE_TORCH:
        raise SystemExit("torch required for measure_pool")

    qnet, bk = train_generalist(args.steps, args.torch_seed)

    def rl_f(pid, stack):
        return RLBotPlayer(pid, "RL", stack, qnet=qnet, epsilon=0.0,
                           training=False, feature_mode="belief",
                           belief_state=HMMBeliefState(**bk), mc_engine=_mc())

    def myo_f(pid, s):
        return BotPlayer(pid, "Myopic", s, tight_threshold=0.2, aggression=0.5,
                         mc_engine=_mc())

    def tlt_f(pid, s):
        return AdaptiveBotPlayer(pid, "Tilt", s, mode="tilt", mc_engine=_mc())

    def rnd_f(pid, s):
        return AdaptiveBotPlayer(pid, "Random", s, mode="random", mc_engine=_mc())

    def kel_f(pid, s):
        return KellyBotPlayer(pid, "Kelly", s, mc_engine=_mc())

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    # Cross-agent leaderboard: RL vs the {myopic, tilt, random} pool + Kelly.
    roster = {"RL": rl_f, "Myopic": myo_f, "Tilt": tlt_f, "Random": rnd_f,
              "Kelly": kel_f}
    rr = evaluate_roster(roster, seeds, n_hands=args.hands)
    # Attach a bootstrap 95% CI of each agent's mean net chips (the honest
    # uncertainty band behind the leaderboard means).
    for entry in rr.leaderboard:
        entry["ci95"] = bootstrap_ci(rr.per_agent_nets[entry["name"]])

    # Static-personality fitness landscape; RL added as an extra (forces MC).
    tights = [0.2, 0.4, 0.6, 0.8]
    aggrs = [0.3, 0.5, 0.7, 0.9]
    sweep_rr, grid = parameter_sweep(
        tights, aggrs, seeds, n_hands=args.hands, mc_sims=100, fast_mode=False,
        extra_agents={"RL": rl_f})
    best_static = max(grid, key=lambda g: g["mean_net_chips"])
    sweep_board = [e["name"] for e in sweep_rr.leaderboard]
    rl_rank = sweep_board.index("RL") + 1
    rl_mean = {e["name"]: e["mean_net_chips"]
               for e in sweep_rr.leaderboard}["RL"]

    out = {
        "recipe": "belief+sharp+opponent-mix, reward=chips, multi_hand",
        "steps": args.steps, "torch_seed": args.torch_seed,
        "n_seeds": rr.n_seeds, "n_hands": args.hands,
        "leaderboard": rr.leaderboard,
        "per_agent_nets": rr.per_agent_nets,
        "win_matrix": rr.win_matrix,
        "sweep_leaderboard": sweep_rr.leaderboard,
        "grid": grid,
        "rl_rank": rl_rank, "rl_mean": rl_mean,
        "best_static": best_static,
        "n_agents_in_sweep": len(sweep_rr.leaderboard),
    }
    print(f"RL sweep rank {rl_rank}/{len(sweep_board)}  RL mean {rl_mean:+.0f}  "
          f"best static {best_static['mean_net_chips']:+.0f} "
          f"(t{best_static['tight']:.2f}/a{best_static['aggr']:.2f})")
    print("Pool leaderboard:",
          [(e["name"], round(e["mean_net_chips"])) for e in rr.leaderboard])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
