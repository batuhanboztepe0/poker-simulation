"""
train_rl.py
-----------
Train the Phase C RL agent and report whether it beats the myopic EV baseline.

This is the demo/report driver for docs/RL_HANDOFF.md: it trains an
`RLBotPlayer` (DQN over the 18-dim featurization, TD(0) targets, MC equity),
prints the learning curve, and prints the headline `evaluate_vs_baseline`
number (wins/N and mean chip diff + a paired t-test).

Examples:
    python -m scripts.train_rl --mode fixed --steps 1500
    python -m scripts.train_rl --mode snapshot --steps 1500 --eval-hands 200
"""

import argparse
import time

import torch

from src.rl_agent import (
    SelfPlayTrainer, evaluate_vs_baseline, paired_t_test, save_trainer_checkpoint,
)
from src.adaptive_agent import AdaptiveBotPlayer
from src.player import BotPlayer
from src.opponent_model import HMMBeliefState


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fixed",
                    choices=["fixed", "snapshot", "self"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--torch-seed", type=int, default=0)
    ap.add_argument("--trainer-seed", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--refresh-every", type=int, default=5)
    ap.add_argument("--hands-per-refresh", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--eval-seeds", type=int, default=30)
    ap.add_argument("--eval-hands", type=int, default=120)
    ap.add_argument("--final-seeds", type=int, default=50)
    ap.add_argument("--final-hands", type=int, default=200)
    ap.add_argument("--mc-sims", type=int, default=100)
    ap.add_argument("--multi-hand", action="store_true",
                    help="bankroll episodes (persistent stacks, log-utility reward)")
    ap.add_argument("--hands-per-episode", type=int, default=15)
    ap.add_argument("--icm", action="store_true",
                    help="ICM prize-pool reward mode (implies --multi-hand; "
                         "--n-players sets the field size)")
    ap.add_argument("--n-players", type=int, default=2,
                    help="number of players at the table (default 2)")
    ap.add_argument("--extended-features", action="store_true",
                    help="use the extended feature vector (horizon appended)")
    ap.add_argument("--save", default=None,
                    help="path to save a reloadable checkpoint (weights + "
                         "feature_mode + learning-curve history) for the dashboard")
    ap.add_argument("--opponent", default="myopic",
                    choices=["myopic", "random", "tilt"],
                    help="fixed-mode opponent: myopic EV bot, or an adaptive bot "
                         "with random or PnL-driven tilt aggression (tilt implies "
                         "--multi-hand)")
    ap.add_argument("--opponent-mix", action="store_true",
                    help="rotate the opponent over {myopic, tilt, random} per "
                         "episode (domain randomization; implies --multi-hand) — "
                         "trains a generalist that resists overfitting to one foe")
    ap.add_argument("--belief", action="store_true",
                    help="belief-conditioned policy: feed the opponent belief "
                         "(posterior_mean, p_tilted) into phi(s) as a feature")
    ap.add_argument("--reward-mode", default="log", choices=["log", "chips"],
                    help="multi-hand reward: log-utility (risk-averse) or chips "
                         "(RISK-NEUTRAL chip delta — gambles into a spewing foe)")
    ap.add_argument("--belief-sharp", action="store_true",
                    help="use the tuned, sharper-detecting HMM belief "
                         "(mu_normal=0.25, mu_tilted=0.92, recover=0.05)")
    ap.add_argument("--no-belief-pnl", action="store_true",
                    help="ablation: disable the hand-boundary PnL->tilt belief "
                         "trigger (revert to aggression-emission-only detection)")
    ap.add_argument("--tilt-bonus", type=float, default=0.0,
                    help="gain-only reward multiplier (1 + tilt_bonus*p_tilted) to "
                         "press the edge vs a detected-tilted opponent (needs --belief)")
    args = ap.parse_args()

    torch.manual_seed(args.torch_seed)
    # ICM prize-pool reward runs as bankroll (multi-hand) episodes; a tilt
    # opponent also needs persistent stacks to accumulate the losses that tilt it.
    multi_hand = (args.multi_hand or args.icm or (args.opponent == "tilt")
                  or args.opponent_mix)
    # Multi-hand bankroll episodes are longer-horizon, so nudge gamma up.
    gamma = 0.99 if multi_hand else 0.97

    extra_kwargs = {}
    if args.icm:
        total_chips = args.n_players * 1000  # stack0=1000
        if args.n_players == 2:
            prize_structure = [total_chips * 0.6, total_chips * 0.4]
        elif args.n_players == 3:
            prize_structure = [total_chips * 0.5, total_chips * 0.3,
                               total_chips * 0.2]
        else:
            fracs = [0.5, 0.3, 0.2] + [0.0] * max(0, args.n_players - 3)
            fracs = fracs[:args.n_players]
            s = sum(fracs) or 1.0
            prize_structure = [total_chips * f / s for f in fracs]
        extra_kwargs["icm_prize_structure"] = prize_structure
    # Feature mode: horizon and/or belief append-ons (opt-in).
    if args.belief and args.extended_features:
        fmode = "full"
    elif args.belief:
        fmode = "belief"
    elif args.extended_features:
        fmode = "horizon"
    else:
        fmode = None
    if fmode:
        extra_kwargs["extended_features"] = True
        extra_kwargs["feature_mode"] = fmode
    belief_kwargs = {}
    if args.belief:
        if args.belief_sharp:
            belief_kwargs = dict(mu_normal=0.25, mu_tilted=0.92, recover=0.05)
        # Stored in the checkpoint so the dashboard/eval rebuild the EXACT
        # detector (incl. whether the PnL->tilt trigger is live).
        belief_kwargs["use_pnl"] = not args.no_belief_pnl
        extra_kwargs["learner_belief_factory"] = (
            lambda bk=belief_kwargs: HMMBeliefState(**bk))
    if multi_hand:
        extra_kwargs["reward_mode"] = args.reward_mode
    if args.tilt_bonus:
        extra_kwargs["tilt_reward_bonus"] = args.tilt_bonus
        # FOOTGUN: the PnL->tilt belief feed and the gain-only tilt-bonus are
        # SUBSTITUTES for the tilt edge, not complements. Poker is zero-sum, so a
        # learner win == the opponent's loss spikes p_tilted exactly when the
        # learner just won big; the bonus then amplifies the learner's OWN big
        # wins (a reward distortion) and the policy collapses to ~break-even.
        # Measured: PnL+bonus tilt +34 vs PnL+no-bonus +533 / no-PnL+bonus +567.
        # Use one or the other.
        if args.belief and not args.no_belief_pnl:
            print("  [WARN] --tilt-bonus with the PnL->tilt belief feed live is "
                  "DESTRUCTIVE (zero-sum coupling corrupts the reward). Use "
                  "EITHER --tilt-bonus (add --no-belief-pnl) OR the PnL feed "
                  "with no --tilt-bonus. See RL_HANDOFF §10.")

    if args.opponent_mix:
        if args.mode != "fixed":
            print("  [note] --opponent-mix only applies to --mode fixed.")

        def _myopic_f(pid, s, mc, rng):
            return BotPlayer(pid, "Myopic", s, tight_threshold=0.2,
                             aggression=0.5, mc_engine=mc, rng=rng)

        def _tilt_f(pid, s, mc, rng):
            return AdaptiveBotPlayer(pid, "Tilt", s, mode="tilt",
                                     mc_engine=mc, rng=rng)

        def _rand_f(pid, s, mc, rng):
            return AdaptiveBotPlayer(pid, "Random", s, mode="random",
                                     mc_engine=mc, rng=rng)
        extra_kwargs["opponent_factories"] = [_myopic_f, _tilt_f, _rand_f]
    elif args.opponent != "myopic":
        if args.mode != "fixed":
            print(f"  [note] --opponent {args.opponent} only applies to "
                  f"--mode fixed; ignored for --mode {args.mode}.")

        def _opp_factory(pid, stack, mc, rng, _mode=args.opponent):
            return AdaptiveBotPlayer(pid, _mode.capitalize(), stack,
                                     mode=_mode, mc_engine=mc, rng=rng)
        extra_kwargs["opponent_factory"] = _opp_factory

    trainer = SelfPlayTrainer(
        n_players=args.n_players, hidden=args.hidden, seed=args.trainer_seed,
        opponent_mode=args.mode, mc_sims=args.mc_sims,
        epsilon_start=1.0, epsilon_end=0.05, gamma=gamma, snapshot_every=300,
        multi_hand=multi_hand, hands_per_episode=args.hands_per_episode,
        **extra_kwargs,
    )

    print(f"Training mode={args.mode} opponent={args.opponent} "
          f"steps={args.steps} hidden={args.hidden} "
          f"torch_seed={args.torch_seed} ...")
    t0 = time.time()
    losses = trainer.train(
        args.steps, batch_size=args.batch_size,
        refresh_every=args.refresh_every,
        hands_per_refresh=args.hands_per_refresh,
        eval_every=(None if args.belief else args.eval_every),
        eval_seeds=args.eval_seeds,
        eval_hands=args.eval_hands, eval_mc_sims=args.mc_sims,
    )
    train_dt = time.time() - t0

    if args.belief:
        # The vs-myopic headline / learning curve use the BASE feature vector and
        # would mismatch a belief-conditioned net; evaluate this checkpoint in the
        # dashboard "Evaluation / Parameter sweep" tab instead.
        print(f"\nBelief-conditioned policy trained "
              f"(train {train_dt:.1f}s, final loss {losses[-1]:.4f}). "
              f"Evaluate it in the dashboard 'Evaluation' tab.")
    else:
        print(f"\nLearning curve (eval vs myopic, "
              f"{args.eval_seeds} seeds x {args.eval_hands} hands):")
        for h in trainer.history:
            print(f"  step {h['step']:5d}:  wins {h['wins']:2d}/{h['n_seeds']}  "
                  f"mean_diff {h['mean_chip_diff']:+8.0f}")
        final = evaluate_vs_baseline(
            trainer.qnet, n_seeds=args.final_seeds, n_hands=args.final_hands,
            mc_sims=args.mc_sims)
        tt = paired_t_test(final["per_seed_diffs"])
        print(f"\nHEADLINE  ({args.final_seeds} seeds x {args.final_hands} hands):")
        print(f"  wins           : {final['wins']}/{final['n_seeds']}")
        print(f"  mean chip diff : {final['mean_chip_diff']:+.1f}")
        print(f"  paired t-test  : t={tt['t']:.2f}  p={tt['p_value']:.4f}  "
              f"(n={tt['n']})")
        print(f"  train time     : {train_dt:.1f}s  (final loss {losses[-1]:.4f})")

    if args.save:
        save_trainer_checkpoint(trainer, args.save,
                                meta={"mode": args.mode, "steps": args.steps,
                                      "belief_kwargs": belief_kwargs})
        print(f"  saved checkpoint -> {args.save}")


if __name__ == "__main__":
    main()
