"""Characterize multi-hand (bankroll, log-utility) RL across episode length / seed.

Reports both the bust-match headline (wins/mean vs myopic) AND a risk signature:
the bust rate of the trained agent (how often it loses its whole stack), which a
growth-optimal log-utility policy should keep lower than a risk-neutral one.
"""
import random, torch
from src.rl_agent import (SelfPlayTrainer, evaluate_vs_baseline, paired_t_test,
                          RLBotPlayer)
from src.monte_carlo import MonteCarloEngine
from src.game import GameEngine
from src.player import BotPlayer


def bust_rate(qnet, n_seeds=50, n_hands=200, mc_sims=100):
    """Fraction of matches in which the RL agent busts to 0."""
    busts = 0
    for seed in range(n_seeds):
        rng = random.Random(seed); mc = MonteCarloEngine(mc_sims, rng=rng)
        rl = RLBotPlayer(1, "RL", 1000, qnet=qnet, epsilon=0.0, training=False,
                         mc_engine=mc, rng=rng)
        myo = BotPlayer(2, "M", 1000, tight_threshold=0.2, aggression=0.5,
                        mc_engine=mc, rng=rng)
        eng = GameEngine([rl, myo], 10, 20, verbose=False, rng=rng)
        for _ in range(n_hands):
            if min(rl.stack, myo.stack) <= 0:
                break
            eng.play_hand()
        if rl.stack <= 0:
            busts += 1
    return busts / n_seeds


def run(multi, hpe, ts, steps=1500):
    torch.manual_seed(ts)
    gamma = 0.99 if multi else 0.97
    tr = SelfPlayTrainer(seed=1, opponent_mode="fixed", mc_sims=100,
                         epsilon_start=1.0, epsilon_end=0.05, gamma=gamma,
                         multi_hand=multi, hands_per_episode=hpe)
    tr.train(steps, batch_size=64, refresh_every=5, hands_per_refresh=20)
    diffs = []
    for start in (0, 50, 100):
        r = evaluate_vs_baseline(tr.qnet, n_seeds=50, n_hands=150, mc_sims=100,
                                 seed_start=start)
        diffs += r["per_seed_diffs"]
    tt = paired_t_test(diffs)
    w = sum(1 for d in diffs if d > 0)
    br = bust_rate(tr.qnet)
    return w, tt, br


print("MULTI-HAND (log-utility, persistent stacks):")
for hpe in (15, 30):
    for ts in (0, 1, 2):
        w, tt, br = run(True, hpe, ts)
        print(f"  hpe={hpe} ts={ts}: wins {w}/150  mean {tt['mean']:+.0f} "
              f"p={tt['p_value']:.4f}  bust_rate {br:.2f}")

print("\nSINGLE-HAND reference (risk-neutral, for bust-rate contrast):")
for ts in (0, 1, 2):
    w, tt, br = run(False, 30, ts)
    print(f"  ts={ts}: wins {w}/150  mean {tt['mean']:+.0f} "
          f"p={tt['p_value']:.4f}  bust_rate {br:.2f}")
