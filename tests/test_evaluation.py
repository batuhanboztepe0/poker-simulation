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
from src.evaluation import (
    evaluate_matchup, evaluate_roster, MatchupResult, RosterResult,
)
from src.analytics import drawdown_curve, max_drawdown
from app.charts import (
    pnl_distribution_figure, paired_diff_figure, learning_curve_figure,
    equity_drawdown_figure, pnl_box_figure,
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


if __name__ == "__main__":
    unittest.main()
