"""
test_evaluation.py
------------------
Tests for the Monte-Carlo PnL evaluation layer (src/evaluation.py), the
drawdown helpers (src/analytics.py), the new PnL chart factories
(app/charts.py), and RL checkpoint save/load (src/rl_agent.py).

Uses fast_mode (no Monte Carlo) so the structural / determinism / conservation
checks stay quick; strategy quality is not under test here.
"""

import sys
import os
import unittest

import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.player import BotPlayer
from src.kelly_agent import KellyBotPlayer
from src.monte_carlo import MonteCarloEngine
from src.simulation import simulate_session
from src.adaptive_agent import AdaptiveBotPlayer
from src.evaluation import (
    evaluate_matchup, evaluate_roster, parameter_sweep,
    MatchupResult, RosterResult, bootstrap_ci,
)
from src.analytics import drawdown_curve, max_drawdown
from src.stats import binomial_sign_test
from app.charts import (
    pnl_distribution_figure, paired_diff_figure, learning_curve_figure,
    equity_drawdown_figure, pnl_box_figure, parameter_heatmap_figure,
    ab_grouped_bar_figure, ab_heatmap_figure, icm_edge_figure,
    forest_plot_figure, exploitability_curve_figure,
)


def _myopic(pid, stack):
    return BotPlayer(pid, "Myopic", stack, mc_engine=MonteCarloEngine(100))


def _kelly(pid, stack):
    return KellyBotPlayer(pid, "Kelly", stack, mc_engine=MonteCarloEngine(100))


SEEDS = [0, 1, 2, 3]
_N_HANDS = 40


class TestMatchup(unittest.TestCase):
    def test_structure_paired_and_zero_sum(self):
        mr = evaluate_matchup(_myopic, _kelly, "Myopic", "Kelly", SEEDS,
                              n_hands=_N_HANDS, fast_mode=True)
        self.assertIsInstance(mr, MatchupResult)
        self.assertEqual(len(mr.diffs), len(SEEDS))
        # diff is the paired difference net_a - net_b ...
        for d, a, b in zip(mr.diffs, mr.net_a, mr.net_b):
            self.assertEqual(d, a - b)
        # ... and heads-up is zero-sum (chip conservation): net_a + net_b == 0.
        for a, b in zip(mr.net_a, mr.net_b):
            self.assertEqual(a + b, 0)
        self.assertEqual(mr.wins_a + mr.wins_b + mr.ties, len(SEEDS))
        self.assertIn("p_value", mr.t_test)

    def test_determinism(self):
        m1 = evaluate_matchup(_myopic, _kelly, "Myopic", "Kelly", SEEDS,
                              n_hands=_N_HANDS, fast_mode=True)
        m2 = evaluate_matchup(_myopic, _kelly, "Myopic", "Kelly", SEEDS,
                              n_hands=_N_HANDS, fast_mode=True)
        self.assertEqual(m1.diffs, m2.diffs)
        self.assertEqual(m1.net_a, m2.net_a)


class TestBinomialSignTest(unittest.TestCase):
    """Exact sign test for the headline bust-match win count (src.stats)."""

    def test_headline_32_of_50_not_significant(self):
        # 32 wins / 18 losses (the n=50 headline) -> exact two-sided p ~ 0.065,
        # NOT significant at 0.05. The correct test for binary bust outcomes;
        # paired_t on the +/-2000 spread understates p.
        r = binomial_sign_test([2000] * 32 + [-2000] * 18)
        self.assertEqual((r["wins"], r["losses"], r["ties"], r["n"]),
                         (32, 18, 0, 50))
        self.assertAlmostEqual(r["p_value"], 0.0649, places=3)
        self.assertGreater(r["p_value"], 0.05)

    def test_strong_majority_significant(self):
        # The same 64% win rate at n=200 IS resolved (p < 0.001).
        self.assertLess(binomial_sign_test([1] * 128 + [-1] * 72)["p_value"],
                        0.001)

    def test_only_sign_matters_and_symmetric(self):
        # Magnitudes are irrelevant; win/loss swap is symmetric.
        self.assertEqual(binomial_sign_test([5] * 12 + [-5] * 8)["p_value"],
                         binomial_sign_test([1] * 12 + [-9999] * 8)["p_value"])
        self.assertEqual(binomial_sign_test([1] * 12 + [-1] * 8)["p_value"],
                         binomial_sign_test([1] * 8 + [-1] * 12)["p_value"])

    def test_ties_dropped_and_degenerate(self):
        r = binomial_sign_test([1, -1, 0, 0, 1, -1])
        self.assertEqual((r["wins"], r["losses"], r["ties"], r["n"]),
                         (2, 2, 2, 4))
        self.assertEqual(r["p_value"], 1.0)            # even split -> p = 1
        self.assertIsNone(binomial_sign_test([0, 0, 0])["p_value"])


class TestVarianceReduction(unittest.TestCase):
    def test_bootstrap_ci_deterministic_and_brackets_mean(self):
        vals = [10, -5, 3, 0, 8, -2, 12, 4]
        c1 = bootstrap_ci(vals, n_resamples=2000, seed=1)
        c2 = bootstrap_ci(vals, n_resamples=2000, seed=1)
        self.assertEqual(c1, c2)  # seeded -> reproducible
        self.assertAlmostEqual(c1["mean"], sum(vals) / len(vals))
        self.assertLessEqual(c1["lo"], c1["mean"])
        self.assertLessEqual(c1["mean"], c1["hi"])

    def test_bootstrap_ci_degenerate_and_significance(self):
        # All-equal -> zero-width CI at the value.
        deg = bootstrap_ci([7, 7, 7, 7], n_resamples=500, seed=0)
        self.assertEqual((deg["lo"], deg["hi"]), (7.0, 7.0))
        # A clearly-positive sample's 95% CI excludes 0.
        pos = bootstrap_ci([100] * 15 + [60, 140], n_resamples=3000, seed=0)
        self.assertGreater(pos["lo"], 0)
        # Higher spread -> wider CI for the same mean ~0.
        tight = bootstrap_ci([-1, 1] * 10, n_resamples=3000, seed=0)
        wide = bootstrap_ci([-100, 100] * 10, n_resamples=3000, seed=0)
        self.assertGreater(wide["hi"] - wide["lo"], tight["hi"] - tight["lo"])

    def test_mirror_zero_sum_deterministic_antisymmetric(self):
        seeds = [0, 1, 2, 3]
        m1 = evaluate_matchup(_myopic, _kelly, "M", "K", seeds,
                              n_hands=_N_HANDS, fast_mode=True, mirror=True)
        m2 = evaluate_matchup(_myopic, _kelly, "M", "K", seeds,
                              n_hands=_N_HANDS, fast_mode=True, mirror=True)
        self.assertEqual(m1.diffs, m2.diffs)  # deterministic
        for a, b in zip(m1.net_a, m1.net_b):  # duplicate still chip-conserving
            self.assertAlmostEqual(a + b, 0.0)
        # Swapping A/B negates every per-seed diff exactly (mirror is symmetric).
        swapped = evaluate_matchup(_kelly, _myopic, "K", "M", seeds,
                                   n_hands=_N_HANDS, fast_mode=True, mirror=True)
        for d, ds in zip(m1.diffs, swapped.diffs):
            self.assertAlmostEqual(d, -ds)

    def test_luck_adjusted_conserves_engages_unbiased(self):
        # Aggressive bots create all-in confrontations -> the all-in EV control
        # variate engages; the luck-adjusted nets must still be zero-sum
        # (conservation), and the adjusted mean must track the raw mean (the
        # control variate is unbiased; raw and adj share the same realised game
        # because the EV engine never touches the game deck).
        def aggr(pid, s):
            return BotPlayer(pid, "Ag", s, tight_threshold=0.1, aggression=0.9,
                             mc_engine=MonteCarloEngine(100))
        seeds = list(range(24))
        raw = evaluate_matchup(_myopic, aggr, "M", "A", seeds, n_hands=40)
        adj = evaluate_matchup(_myopic, aggr, "M", "A", seeds, n_hands=40,
                               luck_adjusted=True)
        for a, b in zip(adj.net_a, adj.net_b):       # zero-sum conservation
            self.assertAlmostEqual(a + b, 0.0, places=6)
        changed = sum(1 for r, a in zip(raw.diffs, adj.diffs)
                      if abs(r - a) > 1e-6)
        self.assertGreater(changed, 0)               # all-ins -> engaged
        sd = (sum((d - raw.mean_diff) ** 2 for d in raw.diffs)
              / len(raw.diffs)) ** 0.5
        self.assertLess(abs(adj.mean_diff - raw.mean_diff), 0.5 * sd)  # unbiased

    def test_luck_adjusted_default_off_identity(self):
        # track_allin_ev=False (default) records no adjustment, so the
        # luck-adjusted net equals the raw net exactly (byte-identical baseline).
        from src.simulation import simulate_session
        players = [BotPlayer(1, "A", 1000, mc_engine=MonteCarloEngine(100)),
                   BotPlayer(2, "B", 1000, mc_engine=MonteCarloEngine(100))]
        res = simulate_session(players, n_hands=40, seed=3)
        self.assertEqual(res.allin_ev_adjust, {})
        for pid in (1, 2):
            self.assertEqual(res.net_chips_luck_adjusted(pid),
                             res.net_chips(pid))


class TestRoster(unittest.TestCase):
    def setUp(self):
        self.roster = {"Myopic": _myopic, "Kelly": _kelly}

    def test_roster_structure(self):
        rr = evaluate_roster(self.roster, [0, 1, 2], n_hands=_N_HANDS,
                             fast_mode=True)
        self.assertIsInstance(rr, RosterResult)
        self.assertEqual({e["name"] for e in rr.leaderboard},
                         {"Myopic", "Kelly"})
        vals = [e["mean_net_chips"] for e in rr.leaderboard]
        self.assertEqual(vals, sorted(vals, reverse=True))  # ranked desc
        self.assertEqual(len(rr.per_agent_nets["Myopic"]), 3)  # one per seed
        wm = rr.win_matrix
        self.assertLessEqual(wm["Myopic"]["Kelly"] + wm["Kelly"]["Myopic"], 3)

    def test_determinism(self):
        r1 = evaluate_roster(self.roster, [0, 1, 2], n_hands=_N_HANDS,
                             fast_mode=True)
        r2 = evaluate_roster(self.roster, [0, 1, 2], n_hands=_N_HANDS,
                             fast_mode=True)
        self.assertEqual(r1.leaderboard, r2.leaderboard)
        self.assertEqual(r1.win_matrix, r2.win_matrix)


class TestDrawdown(unittest.TestCase):
    def _session(self):
        players = [
            BotPlayer(1, "A", 1000, mc_engine=MonteCarloEngine(100)),
            BotPlayer(2, "B", 1000, mc_engine=MonteCarloEngine(100)),
        ]
        return simulate_session(players, n_hands=60, seed=7, fast_mode=True)

    def test_drawdown_properties(self):
        res = self._session()
        dd = drawdown_curve(res, 1)
        self.assertFalse(dd.empty)
        peaks = list(dd["peak"])
        self.assertEqual(peaks, sorted(peaks))  # running max -> non-decreasing
        for _, r in dd.iterrows():
            self.assertGreaterEqual(r["drawdown"], 0)
            self.assertEqual(r["drawdown"], r["peak"] - r["stack"])
        self.assertEqual(max_drawdown(res, 1), max(dd["drawdown"]))


class TestCharts(unittest.TestCase):
    def test_pnl_charts(self):
        self.assertIsInstance(
            pnl_distribution_figure([10, -5, 3, 0], "A", "B"), go.Figure)
        self.assertIsInstance(
            paired_diff_figure([10, -5, 3], "A", "B", [0, 1, 2]), go.Figure)
        hist = [{"step": 100, "wins": 12, "n_seeds": 20, "mean_chip_diff": 50.0},
                {"step": 200, "wins": 14, "n_seeds": 20, "mean_chip_diff": 80.0}]
        self.assertIsInstance(learning_curve_figure(hist), go.Figure)
        self.assertIsInstance(
            pnl_box_figure({"A": [10, -5, 3], "B": [-10, 5, -3]}), go.Figure)
        self.assertIsInstance(parameter_heatmap_figure([
            {"tight": 0.3, "aggr": 0.5, "mean_net_chips": 10.0},
            {"tight": 0.6, "aggr": 0.5, "mean_net_chips": -5.0},
        ]), go.Figure)

    def test_learning_curve_ribbon(self):
        # Snapshots carrying per_seed_diffs draw a SEM ribbon; the call must
        # still return a Figure (and the no-diffs path stays backward-compatible).
        hist = [
            {"step": 100, "wins": 6, "n_seeds": 4, "mean_chip_diff": -500.0,
             "per_seed_diffs": [-2000, 2000, -2000, 2000]},
            {"step": 200, "wins": 14, "n_seeds": 4, "mean_chip_diff": 1500.0,
             "per_seed_diffs": [2000, 2000, -2000, 2000]},
        ]
        self.assertIsInstance(learning_curve_figure(hist, ribbon=True), go.Figure)
        self.assertIsInstance(learning_curve_figure(hist, ribbon=False), go.Figure)

    def test_ab_measurement_charts(self):
        # B2-style action-grid rows (grouped bars: five vs seven per init seed).
        grid_rows = [
            {"grid": "five", "init_seed": 0, "mean": 462.0},
            {"grid": "seven", "init_seed": 0, "mean": 298.0},
            {"grid": "five", "init_seed": 1, "mean": 267.0},
            {"grid": "seven", "init_seed": 1, "mean": 80.0},
        ]
        self.assertIsInstance(
            ab_grouped_bar_figure(grid_rows, group_key="init_seed",
                                  value_key="mean", by_key="grid"), go.Figure)
        # Single-series variant (no by_key).
        self.assertIsInstance(
            ab_grouped_bar_figure(grid_rows, group_key="grid",
                                  value_key="mean"), go.Figure)
        # Melted config×opponent rows for the heatmap.
        melted = [
            {"config": "fixed", "opponent": "myopic", "value": 250.0},
            {"config": "fixed", "opponent": "tilt", "value": 167.0},
            {"config": "snapshot", "opponent": "myopic", "value": 117.0},
            {"config": "snapshot", "opponent": "tilt", "value": 317.0},
        ]
        self.assertIsInstance(
            ab_heatmap_figure(melted, row_key="config", col_key="opponent",
                              value_key="value"), go.Figure)
        # ICM edge rows.
        icm_rows = [
            {"init_seed": 1, "icm_minus_chips": 85.0},
            {"init_seed": 2, "icm_minus_chips": -30.0},
            {"init_seed": 3, "icm_minus_chips": 120.0},
        ]
        self.assertIsInstance(icm_edge_figure(icm_rows), go.Figure)
        # Forest plot: rows with mean + 95% CI bounds (one CI excludes 0, one not).
        self.assertIsInstance(forest_plot_figure([
            {"label": "A vs B", "mean": 560.0, "lo": 0.0, "hi": 1040.0},
            {"label": "C vs D", "mean": -120.0, "lo": -350.0, "hi": 90.0},
        ]), go.Figure)
        # Exploitability curve: avg vs last-iterate over iterations.
        self.assertIsInstance(exploitability_curve_figure([
            {"iters": 10, "avg_exploitability": 0.43,
             "last_iterate_exploitability": 1.27},
            {"iters": 1000, "avg_exploitability": 0.007,
             "last_iterate_exploitability": 0.40},
        ], uniform=4.03), go.Figure)

    def test_drawdown_chart(self):
        players = [
            BotPlayer(1, "A", 1000, mc_engine=MonteCarloEngine(100)),
            BotPlayer(2, "B", 1000, mc_engine=MonteCarloEngine(100)),
        ]
        res = simulate_session(players, n_hands=40, seed=3, fast_mode=True)
        self.assertIsInstance(
            equity_drawdown_figure(drawdown_curve(res, 1), "A"), go.Figure)


class TestCheckpoint(unittest.TestCase):
    def test_save_load_roundtrip(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        import tempfile
        from src.rl_agent import QNetwork, save_checkpoint, load_checkpoint

        net = QNetwork(input_dim=18, hidden=32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ck.pt")
            save_checkpoint(net, path, hidden=32, input_dim=18,
                            feature_mode="base",
                            history=[{"step": 1, "wins": 3, "n_seeds": 5,
                                      "mean_chip_diff": 2.0}])
            net2, ckpt = load_checkpoint(path)
            self.assertEqual(ckpt["input_dim"], 18)
            self.assertEqual(ckpt["feature_mode"], "base")
            self.assertEqual(len(ckpt["history"]), 1)
            for k in net.state_dict():
                self.assertTrue(torch.equal(net.state_dict()[k],
                                            net2.state_dict()[k]))


class TestAdaptiveAgent(unittest.TestCase):
    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            AdaptiveBotPlayer(1, "x", 1000, mode="nope")

    def test_random_mode_varies_within_range(self):
        import random as _r
        p = AdaptiveBotPlayer(1, "R", 1000, mode="random", aggr_lo=0.1,
                              aggr_hi=0.9, rng=_r.Random(0))
        seen = set()
        for _ in range(25):
            p.reset_for_hand()
            self.assertGreaterEqual(p.aggression, 0.1)
            self.assertLessEqual(p.aggression, 0.9)
            seen.add(round(p.aggression, 6))
        self.assertGreater(len(seen), 1)  # genuinely resampled

    def test_tilt_triggers_on_big_loss(self):
        # An 800-chip loss gives transition_prob = min(1, 0.04+1.5*0.8) = 1.0,
        # so the regime switch is certain regardless of the rng draw.
        p = AdaptiveBotPlayer(1, "T", 1000, mode="tilt", base_aggression=0.3,
                              tilt_aggression=0.95, tilt_tight=0.1, recover=0.0)
        p._prev_stack = 1000
        p.stack = 200
        p.reset_for_hand()
        self.assertTrue(p.is_tilted)
        self.assertEqual(p.aggression, 0.95)
        self.assertEqual(p.tight_threshold, 0.1)

    def test_adaptive_session_conserves_and_deterministic(self):
        def roster():
            return [
                AdaptiveBotPlayer(1, "Tilt", 1000, mode="tilt"),
                AdaptiveBotPlayer(2, "Rand", 1000, mode="random"),
                BotPlayer(3, "S", 1000, tight_threshold=0.5, aggression=0.5),
            ]
        r1 = simulate_session(roster(), n_hands=40, seed=5, fast_mode=True)
        self.assertEqual(sum(r1.starting_stacks.values()),
                         sum(r1.final_stacks.values()))
        r2 = simulate_session(roster(), n_hands=40, seed=5, fast_mode=True)
        self.assertEqual(r1.final_stacks, r2.final_stacks)


class TestParameterSweep(unittest.TestCase):
    def test_grid_shape_and_leaderboard(self):
        rr, grid = parameter_sweep([0.3, 0.6], [0.4, 0.8], seeds=[0, 1],
                                   n_hands=30, fast_mode=True)
        self.assertEqual(len(grid), 4)            # 2 x 2
        self.assertEqual(len(rr.leaderboard), 4)  # one agent per cell
        for cell in grid:
            self.assertIn("mean_net_chips", cell)

    def test_extra_agents_join_roster_not_grid(self):
        from src.kelly_agent import KellyBotPlayer
        extra = {"Kelly": lambda pid, s: KellyBotPlayer(pid, "Kelly", s)}
        rr, grid = parameter_sweep([0.3, 0.6], [0.5], seeds=[0, 1], n_hands=30,
                                   fast_mode=True, extra_agents=extra)
        self.assertIn("Kelly", {e["name"] for e in rr.leaderboard})
        self.assertEqual(len(grid), 2)  # Kelly is in the roster, not the grid

    def test_determinism(self):
        r1, _ = parameter_sweep([0.3, 0.6], [0.4, 0.8], seeds=[0, 1],
                                n_hands=30, fast_mode=True)
        r2, _ = parameter_sweep([0.3, 0.6], [0.4, 0.8], seeds=[0, 1],
                                n_hands=30, fast_mode=True)
        self.assertEqual(r1.leaderboard, r2.leaderboard)


class TestTrainAgainstAdaptive(unittest.TestCase):
    def test_opponent_factory_uses_adaptive_and_trains(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from src.rl_agent import SelfPlayTrainer

        def opp(pid, stack, mc, rng):
            return AdaptiveBotPlayer(pid, "Tilt", stack, mode="tilt",
                                     mc_engine=mc, rng=rng)

        tr = SelfPlayTrainer(opponent_mode="fixed", multi_hand=True,
                             hands_per_episode=8, mc_sims=100, seed=0,
                             opponent_factory=opp)
        # the frozen opponent is the adaptive bot we supplied, not a myopic bot
        self.assertIsInstance(tr.opponents[0], AdaptiveBotPlayer)
        losses = tr.train(15, hands_per_refresh=16)
        self.assertEqual(len(losses), 15)
        self.assertTrue(all(loss >= 0 for loss in losses))


class TestBeliefMixTraining(unittest.TestCase):
    def _torch_or_skip(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_belief_feature_and_chips_reward(self):
        self._torch_or_skip()
        from src.rl_agent import SelfPlayTrainer
        from src.opponent_model import HMMBeliefState
        tr = SelfPlayTrainer(opponent_mode="fixed", multi_hand=True,
                             hands_per_episode=8, mc_sims=100, seed=0,
                             extended_features=True, feature_mode="belief",
                             learner_belief_factory=lambda: HMMBeliefState(),
                             reward_mode="chips")
        learner = tr.learners[0]
        self.assertEqual(tr.qnet.net[0].in_features, 20)   # 18 + 2 belief floats
        self.assertIsNotNone(learner.belief_state)
        self.assertIs(learner._belief, learner.belief_state)
        self.assertFalse(learner.use_belief_equity)        # belief is feature-only
        self.assertEqual(tr.reward_mode, "chips")
        self.assertEqual(tr.reward_clip, 1.0)              # risk-neutral chip scale
        losses = tr.train(12, hands_per_refresh=16)
        self.assertEqual(len(losses), 12)

    def test_opponent_rotation_uses_pool(self):
        self._torch_or_skip()
        from src.rl_agent import SelfPlayTrainer
        from src.player import BotPlayer

        def myo(pid, s, mc, rng):
            return BotPlayer(pid, "M", s, mc_engine=mc, rng=rng)

        def tilt(pid, s, mc, rng):
            return AdaptiveBotPlayer(pid, "T", s, mode="tilt", mc_engine=mc, rng=rng)

        tr = SelfPlayTrainer(opponent_mode="fixed", multi_hand=True,
                             hands_per_episode=8, mc_sims=100, seed=0,
                             opponent_factories=[myo, tilt])
        seen = set()
        for _ in range(12):
            tr._reseat_rotating_opponents()
            seen.add(type(tr.opponents[0]).__name__)
        self.assertEqual(seen, {"BotPlayer", "AdaptiveBotPlayer"})  # both rotated in
        losses = tr.train(10, hands_per_refresh=16)
        self.assertEqual(len(losses), 10)

    def test_tilt_reward_bonus_trains(self):
        self._torch_or_skip()
        from src.rl_agent import SelfPlayTrainer
        from src.opponent_model import HMMBeliefState
        tr = SelfPlayTrainer(opponent_mode="fixed", multi_hand=True,
                             hands_per_episode=8, mc_sims=100, seed=0,
                             extended_features=True, feature_mode="belief",
                             learner_belief_factory=lambda: HMMBeliefState(
                                 mu_tilted=0.92, recover=0.05),
                             reward_mode="chips", tilt_reward_bonus=0.6)
        self.assertEqual(tr.tilt_reward_bonus, 0.6)
        losses = tr.train(10, hands_per_refresh=16)
        self.assertEqual(len(losses), 10)


class TestRolloutFreeCheck(unittest.TestCase):
    def test_never_folds_a_free_check(self):
        # Regression: a free-check spot (call_amount == 0) must never be FOLDED
        # (folding for free forfeits the pot). The old argmax tie-break picked
        # FOLD over CHECK and the rollout bled chips, badly vs adaptive bots.
        import random
        from src.stochastic_control import RolloutPolicy
        from src.card import Card
        # Seed the MC engine too (not just the policy): MonteCarloEngine falls
        # back to the GLOBAL random module when no rng is passed, which made this
        # assertion depend on prior tests' consumption of global random (the
        # "seed everything" non-negotiable). The free-check-never-fold invariant
        # holds regardless; seeding makes the exact action deterministic.
        pol = RolloutPolicy(MonteCarloEngine(100, rng=random.Random(0)),
                            rng=random.Random(0))
        gs = {"pot": 40, "call_amount": 0, "min_raise": 20, "current_bet": 0,
              "community_cards": [Card('K', 's'), Card('Q', 'd'), Card('9', 'c')],
              "round_name": "Flop", "active_player_count": 2, "opponent_ids": [2]}
        action, _ = pol.decide([Card('7', 'h'), Card('2', 'd')], gs, 1000)
        self.assertEqual(action, "check")  # weak hand checks for free, never folds


if __name__ == "__main__":
    unittest.main()
