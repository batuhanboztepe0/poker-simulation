"""
test_simulation.py
-------------------
Tests for Phase 1: the self-play simulation harness and pure stats.

Covers:
    - Seeded reproducibility (same seed -> identical session).
    - Chip conservation across a long run.
    - Stat correctness on a hand-crafted synthetic session.
    - Throughput budget (1000 hands < 10 s in fast_mode).

Run with: python -m pytest tests/test_simulation.py -v
"""

import sys
import os
import time
import random
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.card import Card
from src.player import BotPlayer
from src.monte_carlo import MonteCarloEngine
from src.events import (
    HandEvent, EVENT_ACTION, EVENT_SHOWDOWN, EVENT_HAND_END,
    STREET_SHOWDOWN, STREET_HAND_END,
)
from src.simulation import (
    simulate_session, SessionResult, HandRecord, PlayerSnapshot,
)
from src.stats import (
    compute_vpip, compute_aggression_frequency, compute_showdown_win_rate,
    chip_ev_per_hand, hands_played, session_summary,
)


def make_bots(n, stack=1000, mc=None, tight=0.3, aggression=0.5):
    return [
        BotPlayer(i, f"B{i}", stack, tight, aggression, mc)
        for i in range(1, n + 1)
    ]


# ===========================================================================
# Reproducibility
# ===========================================================================

class TestReproducibility:
    def test_same_seed_identical_session(self):
        r1 = simulate_session(make_bots(4), n_hands=40, seed=2024, fast_mode=True)
        r2 = simulate_session(make_bots(4), n_hands=40, seed=2024, fast_mode=True)
        for pid in (1, 2, 3, 4):
            assert r1.winnings_sequence(pid) == r2.winnings_sequence(pid)
        assert r1.final_stacks == r2.final_stacks

    def test_same_seed_identical_with_mc(self):
        mc1 = MonteCarloEngine(n_simulations=120)
        mc2 = MonteCarloEngine(n_simulations=120)
        r1 = simulate_session(make_bots(3, mc=mc1), n_hands=25, seed=77)
        r2 = simulate_session(make_bots(3, mc=mc2), n_hands=25, seed=77)
        assert r1.final_stacks == r2.final_stacks
        for pid in (1, 2, 3):
            assert r1.winnings_sequence(pid) == r2.winnings_sequence(pid)

    def test_different_seeds_diverge(self):
        r1 = simulate_session(make_bots(4), n_hands=40, seed=1, fast_mode=True)
        r2 = simulate_session(make_bots(4), n_hands=40, seed=2, fast_mode=True)
        assert r1.final_stacks != r2.final_stacks


# ===========================================================================
# Chip conservation + structure
# ===========================================================================

class TestSessionIntegrity:
    def test_chip_conservation_1000_hands(self):
        players = make_bots(6, stack=1000)
        total_before = sum(p.stack for p in players)
        result = simulate_session(players, n_hands=1000, seed=5, fast_mode=True)
        assert sum(result.final_stacks.values()) == total_before

    def test_result_structure(self):
        result = simulate_session(make_bots(4), n_hands=20, seed=9, fast_mode=True)
        assert isinstance(result, SessionResult)
        assert result.n_hands <= 20
        assert len(result.players) == 4
        assert all(isinstance(h, HandRecord) for h in result.hands)
        # events were harvested per hand
        assert all(len(h.events) > 0 for h in result.hands)

    def test_all_events_flattened(self):
        result = simulate_session(make_bots(3), n_hands=10, seed=9, fast_mode=True)
        flat = result.all_events()
        assert sum(len(h.events) for h in result.hands) == len(flat)

    def test_fast_mode_strips_mc(self):
        mc = MonteCarloEngine(n_simulations=200)
        players = make_bots(4, mc=mc)
        simulate_session(players, n_hands=5, seed=1, fast_mode=True)
        assert all(p.mc_engine is None for p in players)

    def test_stops_when_one_player_left(self):
        # Tiny stacks at a big blind force eliminations well before n_hands.
        players = make_bots(4, stack=40)
        result = simulate_session(players, n_hands=500, seed=3, fast_mode=True)
        # Session ended (early or at cap) with chips conserved.
        assert sum(result.final_stacks.values()) == 4 * 40


# ===========================================================================
# Throughput
# ===========================================================================

class TestThroughput:
    def test_1000_hands_fast_mode_under_10s(self):
        players = make_bots(6, stack=100000)  # big stacks -> few eliminations
        start = time.time()
        result = simulate_session(players, n_hands=1000, seed=1, fast_mode=True)
        elapsed = time.time() - start
        assert elapsed < 10.0, f"1000 hands took {elapsed:.2f}s (budget 10s)"
        assert result.n_hands > 0


# ===========================================================================
# Stat correctness on a synthetic session
# ===========================================================================

def _synthetic_session():
    """
    Two hand-crafted hands with known actions/showdowns:

    Hand 1: P1 raises pre-flop, P2 calls, P3 folds; P1 raises flop, P2 calls;
            showdown P1 vs P2, P1 wins 180.
    Hand 2: P1 folds pre-flop, P2 raises, P3 calls; showdown P2 vs P3, P3 wins 100.
    """
    h1 = [
        HandEvent(1, "Pre-Flop", EVENT_ACTION, player_id=1, action="raise", amount=40),
        HandEvent(1, "Pre-Flop", EVENT_ACTION, player_id=2, action="call", amount=40),
        HandEvent(1, "Pre-Flop", EVENT_ACTION, player_id=3, action="fold", amount=0),
        HandEvent(1, "Flop", EVENT_ACTION, player_id=1, action="raise", amount=50),
        HandEvent(1, "Flop", EVENT_ACTION, player_id=2, action="call", amount=50),
        HandEvent(1, STREET_SHOWDOWN, EVENT_SHOWDOWN, payload={"contenders": [1, 2]}),
        HandEvent(1, STREET_HAND_END, EVENT_HAND_END, payload={"winnings": {1: 180}}),
    ]
    h2 = [
        HandEvent(2, "Pre-Flop", EVENT_ACTION, player_id=1, action="fold", amount=0),
        HandEvent(2, "Pre-Flop", EVENT_ACTION, player_id=2, action="raise", amount=40),
        HandEvent(2, "Pre-Flop", EVENT_ACTION, player_id=3, action="call", amount=40),
        HandEvent(2, STREET_SHOWDOWN, EVENT_SHOWDOWN, payload={"contenders": [2, 3]}),
        HandEvent(2, STREET_HAND_END, EVENT_HAND_END, payload={"winnings": {3: 100}}),
    ]
    return SessionResult(
        hands=[HandRecord(1, {1: 180}, h1), HandRecord(2, {3: 100}, h2)],
        players=[
            PlayerSnapshot(1, "P1", 1000, 1180, 0.3, 0.5),
            PlayerSnapshot(2, "P2", 1000, 820, 0.3, 0.5),
            PlayerSnapshot(3, "P3", 1000, 1000, 0.3, 0.5),
        ],
        starting_stacks={1: 1000, 2: 1000, 3: 1000},
        final_stacks={1: 1180, 2: 820, 3: 1000},
        seed=0, small_blind=10, big_blind=20,
    )


class TestStats:
    def setup_method(self):
        self.s = _synthetic_session()

    def test_vpip(self):
        # P1: raise then fold -> 1/2; P2: call then raise -> 2/2; P3: fold then call -> 1/2
        assert compute_vpip(self.s, 1) == pytest.approx(0.5)
        assert compute_vpip(self.s, 2) == pytest.approx(1.0)
        assert compute_vpip(self.s, 3) == pytest.approx(0.5)

    def test_aggression_frequency(self):
        # P1: 2 aggressive, 0 passive -> 1.0
        # P2: 1 aggressive (h2 raise), 2 passive (h1 call x2) -> 1/3
        # P3: 0 aggressive, 1 passive (h2 call) -> 0.0
        assert compute_aggression_frequency(self.s, 1) == pytest.approx(1.0)
        assert compute_aggression_frequency(self.s, 2) == pytest.approx(1 / 3)
        assert compute_aggression_frequency(self.s, 3) == pytest.approx(0.0)

    def test_showdown_win_rate(self):
        # P1: 1 showdown, 1 win -> 1.0; P2: 2 showdowns, 0 wins -> 0.0;
        # P3: 1 showdown, 1 win -> 1.0
        assert compute_showdown_win_rate(self.s, 1) == pytest.approx(1.0)
        assert compute_showdown_win_rate(self.s, 2) == pytest.approx(0.0)
        assert compute_showdown_win_rate(self.s, 3) == pytest.approx(1.0)

    def test_chip_ev_per_hand(self):
        # P1 net +180 over 2 hands played -> 90
        assert chip_ev_per_hand(self.s, 1) == pytest.approx(90.0)
        # P2 net -180 over 2 hands -> -90
        assert chip_ev_per_hand(self.s, 2) == pytest.approx(-90.0)

    def test_hands_played(self):
        assert hands_played(self.s, 1) == 2
        assert hands_played(self.s, 2) == 2
        assert hands_played(self.s, 3) == 2

    def test_session_summary_keys(self):
        summary = session_summary(self.s)
        assert set(summary.keys()) == {1, 2, 3}
        for pid in (1, 2, 3):
            for key in ("vpip", "aggression_frequency", "showdown_win_rate",
                        "chip_ev_per_hand", "net_chips", "hands_played"):
                assert key in summary[pid]


class TestStatsOnRealSession:
    def test_stats_in_valid_ranges(self):
        result = simulate_session(make_bots(4), n_hands=60, seed=4, fast_mode=True)
        summary = session_summary(result)
        for pid, row in summary.items():
            assert 0.0 <= row["vpip"] <= 1.0
            assert 0.0 <= row["aggression_frequency"] <= 1.0
            assert 0.0 <= row["showdown_win_rate"] <= 1.0
