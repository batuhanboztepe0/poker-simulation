"""
test_tournament.py
------------------
Tests for the Track C agent tournament harness (src/tournament.py).
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.player import BotPlayer
from src.kelly_agent import KellyBotPlayer
from src.stochastic_control import RolloutPolicy, RolloutBotPlayer
from src.monte_carlo import MonteCarloEngine
from src.tournament import run_tournament, TournamentResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_N_HANDS = 50
_SEEDS = [0, 1]
_STARTING_STACK = 1000


def _myopic_factory(player_id, stack):
    return BotPlayer(player_id, "Myopic", stack)


def _kelly_factory(player_id, stack):
    return KellyBotPlayer(player_id, "Kelly", stack)


def _rollout_factory(player_id, stack):
    mc = MonteCarloEngine(n_simulations=100)
    pol = RolloutPolicy(mc)
    return RolloutBotPlayer(player_id, "Rollout", stack, rollout_policy=pol)


_ROSTER_2 = {
    "Myopic": _myopic_factory,
    "Kelly": _kelly_factory,
}

_ROSTER_3 = {
    "Myopic": _myopic_factory,
    "Kelly": _kelly_factory,
    "Rollout": _rollout_factory,
}


# ---------------------------------------------------------------------------
# Test 1: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):

    def test_determinism(self):
        """Same roster + seeds must produce identical win_matrix and leaderboard."""
        kwargs = dict(
            roster=_ROSTER_3,
            seeds=_SEEDS,
            n_hands=_N_HANDS,
            fast_mode=True,
            starting_stack=_STARTING_STACK,
        )
        r1 = run_tournament(**kwargs)
        r2 = run_tournament(**kwargs)
        self.assertEqual(r1.win_matrix, r2.win_matrix)
        self.assertEqual(r1.leaderboard, r2.leaderboard)


# ---------------------------------------------------------------------------
# Test 2: Chip conservation
# ---------------------------------------------------------------------------

class TestChipConservation(unittest.TestCase):

    def test_chip_conservation_direct(self):
        """simulate_session must conserve chips for each pair."""
        from src.simulation import simulate_session

        pairs = [
            ("Myopic", _myopic_factory, "Kelly", _kelly_factory),
            ("Myopic", _myopic_factory, "Rollout", _rollout_factory),
            ("Kelly", _kelly_factory, "Rollout", _rollout_factory),
        ]
        for name_a, fac_a, name_b, fac_b in pairs:
            with self.subTest(pair=(name_a, name_b)):
                p_a = fac_a(1, _STARTING_STACK)
                p_b = fac_b(2, _STARTING_STACK)
                total_before = p_a.stack + p_b.stack
                result = simulate_session(
                    [p_a, p_b], n_hands=_N_HANDS, seed=7,
                    fast_mode=True,
                )
                total_after = sum(result.final_stacks.values())
                self.assertEqual(total_after, total_before,
                                 f"{name_a} vs {name_b}: "
                                 f"before={total_before}, after={total_after}")

    def test_chip_conservation_via_tournament(self):
        """run_tournament asserts conservation; no exception should be raised."""
        result = run_tournament(
            roster=_ROSTER_3,
            seeds=_SEEDS,
            n_hands=_N_HANDS,
            fast_mode=True,
            starting_stack=_STARTING_STACK,
        )
        self.assertIsInstance(result, TournamentResult)


# ---------------------------------------------------------------------------
# Test 3: Leaderboard valid ranking
# ---------------------------------------------------------------------------

class TestLeaderboardValidRanking(unittest.TestCase):

    def setUp(self):
        self.result = run_tournament(
            roster=_ROSTER_3,
            seeds=_SEEDS,
            n_hands=_N_HANDS,
            fast_mode=True,
            starting_stack=_STARTING_STACK,
        )

    def test_leaderboard_is_sorted_descending(self):
        lb = self.result.leaderboard
        self.assertTrue(len(lb) > 0, "Leaderboard must be non-empty")
        for i in range(len(lb) - 1):
            self.assertGreaterEqual(
                lb[i]["mean_net_chips"], lb[i + 1]["mean_net_chips"],
                "Leaderboard must be sorted by mean_net_chips descending"
            )

    def test_leaderboard_all_names_once(self):
        lb = self.result.leaderboard
        names = [entry["name"] for entry in lb]
        self.assertEqual(len(names), len(set(names)), "Duplicate names in leaderboard")
        self.assertEqual(set(names), set(_ROSTER_3.keys()),
                         "Leaderboard must contain exactly the roster names")

    def test_leaderboard_entry_keys(self):
        required = {"name", "mean_net_chips", "n_matches"}
        for entry in self.result.leaderboard:
            self.assertTrue(
                required.issubset(entry.keys()),
                f"Entry missing keys: {required - entry.keys()}"
            )


# ---------------------------------------------------------------------------
# Test 4: Runs without torch
# ---------------------------------------------------------------------------

class TestRunsWithoutTorch(unittest.TestCase):

    def test_runs_without_torch(self):
        """Blocking torch must not raise ImportError; RL agent absent from matrix."""
        import importlib

        # Back up real torch and block it.
        real_torch = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]

        try:
            # Re-import tournament with torch blocked.
            import src.tournament as tm
            importlib.reload(tm)

            # Mark a factory as requiring torch.
            def rl_factory(pid, stack):  # pragma: no cover
                raise RuntimeError("should not be called")
            rl_factory.requires_torch = True

            roster_no_rl = dict(_ROSTER_2)
            roster_with_rl = dict(_ROSTER_2)
            roster_with_rl["RL"] = rl_factory

            # Should complete without error; RL agent excluded.
            result = tm.run_tournament(
                roster=roster_with_rl,
                seeds=_SEEDS,
                n_hands=_N_HANDS,
                fast_mode=True,
                starting_stack=_STARTING_STACK,
            )
            # RL must not appear in win_matrix.
            self.assertNotIn("RL", result.win_matrix)
            # A note about torch / RL must appear.
            combined = " ".join(result.notes).lower()
            self.assertTrue("torch" in combined or "rl" in combined or "rlagent" in combined.replace(" ", ""),
                            f"Expected a torch/RL note, got: {result.notes}")
        finally:
            # Restore torch.
            if real_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = real_torch


# ---------------------------------------------------------------------------
# Test 5: Win matrix shape
# ---------------------------------------------------------------------------

class TestWinMatrixShape(unittest.TestCase):

    def setUp(self):
        self.seeds = _SEEDS
        self.result = run_tournament(
            roster=_ROSTER_3,
            seeds=self.seeds,
            n_hands=_N_HANDS,
            fast_mode=True,
            starting_stack=_STARTING_STACK,
        )

    def test_win_matrix_has_all_directed_pairs(self):
        names = list(_ROSTER_3.keys())
        n = len(names)
        wm = self.result.win_matrix
        for a in names:
            for b in names:
                if a != b:
                    self.assertIn(a, wm, f"{a} missing from win_matrix")
                    self.assertIn(b, wm[a], f"{b} missing from win_matrix[{a}]")
        # Total directed-pair entries: N*(N-1)
        total = sum(len(v) for v in wm.values())
        self.assertEqual(total, n * (n - 1))

    def test_win_matrix_sum_constraint(self):
        """win_matrix[A][B] + win_matrix[B][A] <= len(seeds) (ties possible)."""
        wm = self.result.win_matrix
        names = list(_ROSTER_3.keys())
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i != j:
                    self.assertLessEqual(
                        wm[a][b] + wm[b][a], len(self.seeds),
                        f"Sum constraint violated for ({a}, {b})"
                    )


# ---------------------------------------------------------------------------
# Test 6: Kuhn CFR excluded
# ---------------------------------------------------------------------------

class TestKuhnCFRExcluded(unittest.TestCase):

    def test_kuhn_cfr_note_content(self):
        result = run_tournament(
            roster=_ROSTER_2,
            seeds=_SEEDS,
            n_hands=_N_HANDS,
            fast_mode=True,
            starting_stack=_STARTING_STACK,
        )
        note = result.kuhn_cfr_note
        self.assertIn("-1/18", note, "kuhn_cfr_note must reference -1/18")
        self.assertIn("excluded", note, "kuhn_cfr_note must mention 'excluded'")


if __name__ == "__main__":
    unittest.main()
