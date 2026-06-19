"""
measure_icm.py
--------------
Phase A5: does a concave ICM prize reward beat a risk-neutral chip reward at a
MULTI-PRIZE 3+ player table?

In a heads-up winner-take-all bust-match the loser just busts, so risk-aversion
has nothing to exploit (RL_HANDOFF §9). With a concave prize ladder (e.g.
50/30/20) survival near the money is worth more than the chips suggest, so an
agent trained on ICM prize-equity-to-go should out-EARN (in PRIZE) a chip-EV
agent that over-gambles — even if they win similar chips.

This script trains two agents that differ ONLY in their reward shape (concave
ICM vs linear chips; both bankroll/multi-hand, identical everything else),
seats them with a myopic baseline at a multi-prize table, and measures each
agent's mean realised tournament prize on held-out seeds via
`evaluate_icm_tournament`. Repeats over several init seeds for robustness and
reports a paired t-test on the per-seed (ICM - chips) prize diff.

Run thread-pinned (small MLP; parallel BLAS only thrashes):
    OMP_NUM_THREADS=1 python -m scripts.measure_icm
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse

import src.rl_agent as rl
from src.icm_eval import evaluate_icm_tournament
from src.player import BotPlayer


def myopic_factory(pid, name, stack, mc, rng):
    return BotPlayer(pid, name, stack, tight_threshold=0.2, aggression=0.5,
                     mc_engine=mc, rng=rng)


def rl_factory(qnet):
    def factory(pid, name, stack, mc, rng):
        return rl.RLBotPlayer(pid, name, stack, qnet=qnet, epsilon=0.0,
                              training=False, mc_engine=mc, rng=rng)
    return factory


def prize_ladder(n_players, stack0, fracs=None):
    total = n_players * stack0
    if fracs is None:
        if n_players == 3:
            fracs = [0.5, 0.3, 0.2]
        elif n_players == 2:
            fracs = [0.6, 0.4]
        else:
            fracs = [0.5, 0.3, 0.2] + [0.0] * (n_players - 3)
    fracs = list(fracs) + [0.0] * max(0, n_players - len(fracs))
    return [total * f for f in fracs[:n_players]]


def train_agent(reward, prize, args, seed):
    kw = dict(n_players=args.n_players, hidden=args.hidden, seed=seed,
              opponent_mode="fixed", multi_hand=True,
              hands_per_episode=args.hands_per_episode, mc_sims=args.mc_sims)
    if reward == "icm":
        kw["icm_prize_structure"] = prize
    else:
        kw["reward_mode"] = "chips"
    trainer = rl.SelfPlayTrainer(**kw)
    trainer.train(args.steps)
    return trainer.qnet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-players", type=int, default=3)
    ap.add_argument("--stack0", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--hands-per-episode", type=int, default=15)
    ap.add_argument("--mc-sims", type=int, default=100)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--eval-seeds", type=int, default=60)
    ap.add_argument("--eval-hands", type=int, default=120)
    ap.add_argument("--eval-seed-start", type=int, default=10000,
                    help="held-out eval seed range start (disjoint from training)")
    ap.add_argument("--prize-fracs", type=float, nargs="+", default=None,
                    help="prize-pool fractions (1st, 2nd, ...); e.g. a 3-player "
                         "bubble is '0.65 0.35 0.0'. Defaults to 50/30/20.")
    args = ap.parse_args()

    if not rl._HAVE_TORCH:
        raise SystemExit("torch required for measure_icm")

    prize = prize_ladder(args.n_players, args.stack0, args.prize_fracs)
    eval_seeds = range(args.eval_seed_start, args.eval_seed_start + args.eval_seeds)
    print(f"Table: {args.n_players} players, prize ladder {prize}, "
          f"pool {sum(prize):.0f}")
    print(f"Train {args.steps} steps/agent; eval {args.eval_seeds} held-out "
          f"tournaments x {args.eval_hands} hands.\n")

    icm_minus_chips_means = []
    for seed in args.train_seeds:
        icm_qnet = train_agent("icm", prize, args, seed)
        chips_qnet = train_agent("chips", prize, args, seed)

        res = evaluate_icm_tournament(
            [rl_factory(icm_qnet), rl_factory(chips_qnet), myopic_factory],
            prize, seeds=eval_seeds, names=["icm", "chips", "myopic"],
            stack0=args.stack0, mc_sims=args.mc_sims, max_hands=args.eval_hands)

        icm_ps = res["icm"]["per_seed"]
        chips_ps = res["chips"]["per_seed"]
        diffs = [a - b for a, b in zip(icm_ps, chips_ps)]
        tt = rl.paired_t_test(diffs)
        icm_minus_chips_means.append(tt["mean"])

        print(f"[init seed {seed}] mean prize  "
              f"icm={res['icm']['mean_prize']:.2f}  "
              f"chips={res['chips']['mean_prize']:.2f}  "
              f"myopic={res['myopic']['mean_prize']:.2f}  "
              f"| icm-chips={tt['mean']:+.2f}  p={tt['p_value']}")

    n = len(icm_minus_chips_means)
    grand = sum(icm_minus_chips_means) / n if n else 0.0
    wins = sum(1 for m in icm_minus_chips_means if m > 0)
    print(f"\nAcross {n} init seeds: ICM beat chips on prize in {wins}/{n}; "
          f"mean (icm-chips) prize = {grand:+.2f}")
    print("(ICM concavity pays iff icm-chips > 0 with low p.)")


if __name__ == "__main__":
    main()
