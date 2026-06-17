"""Characterize RL policy quality vs eval noise across torch seeds / budgets."""
import time, torch
from src.rl_agent import SelfPlayTrainer, evaluate_vs_baseline, paired_t_test

def train_eval(steps, hidden, ts):
    torch.manual_seed(ts)
    tr = SelfPlayTrainer(seed=1, hidden=hidden, opponent_mode="fixed",
                         epsilon_start=1.0, epsilon_end=0.05, gamma=0.97)
    tr.train(steps, batch_size=64, refresh_every=5, hands_per_refresh=20)
    # three disjoint eval batches to expose eval noise vs policy quality
    diffs = []
    batch_wins = []
    for start in (0, 50, 100):
        r = evaluate_vs_baseline(tr.qnet, n_seeds=50, n_hands=150, mc_sims=100,
                                 seed_start=start)
        batch_wins.append(r["wins"]); diffs += r["per_seed_diffs"]
    tt = paired_t_test(diffs)
    wins150 = sum(1 for d in diffs if d > 0)
    return batch_wins, wins150, tt

for steps in (1500, 2500):
    for hidden in (64,):
        print(f"\n=== steps={steps} hidden={hidden} ===")
        for ts in (0, 1, 2, 3):
            t0 = time.time()
            bw, w150, tt = train_eval(steps, hidden, ts)
            print(f"  ts={ts}: per-50 wins {bw} | 150-seed wins {w150}/150 "
                  f"mean {tt['mean']:+.0f} t={tt['t']:.2f} p={tt['p_value']:.4f} "
                  f"({time.time()-t0:.0f}s)")
