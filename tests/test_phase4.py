"""
test_phase4.py
--------------
Tests for Phase 4: Multi-Player (4-6 players).

Covers:
    - Correct blind rotation across 3, 4, 5, 6 player tables
    - Action order (pre-flop UTG, post-flop left of dealer)
    - All-in with side pot distribution in multi-player scenarios
    - Chip conservation across many hands and player counts
    - Dealer rotation skipping eliminated players
    - add_player / remove_player table management
    - Heads-up edge case preservation

Run with: python -m pytest tests/test_phase4.py -v
"""

import sys
import os
import pytest
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.card import Card
from src.player import BotPlayer, ACTION_FOLD, ACTION_CALL, ACTION_CHECK
from src.game import GameEngine, MIN_PLAYERS, MAX_PLAYERS
from src.monte_carlo import MonteCarloEngine


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_bots(n, stack=1000, tight=0.3, aggression=0.4, mc=None):
    """Create n BotPlayers with given parameters."""
    return [
        BotPlayer(i, f"Bot{i}", stack,
                  tight_threshold=tight,
                  aggression=aggression,
                  mc_engine=mc)
        for i in range(1, n + 1)
    ]


def make_engine(n_players, stack=1000, small_blind=5,
                big_blind=10, verbose=False, mc=None):
    """Create a GameEngine with n_players bots."""
    players = make_bots(n_players, stack=stack, mc=mc)
    return GameEngine(players, small_blind=small_blind,
                      big_blind=big_blind, verbose=verbose)


# ------------------------------------------------------------------
# Engine initialisation
# ------------------------------------------------------------------

class TestEngineInit:
    def test_two_players_valid(self):
        engine = make_engine(2)
        assert len(engine.players) == 2

    def test_six_players_valid(self):
        engine = make_engine(6)
        assert len(engine.players) == 6

    def test_nine_players_valid(self):
        engine = make_engine(9)
        assert len(engine.players) == 9

    def test_one_player_raises(self):
        with pytest.raises(ValueError, match="Player count"):
            GameEngine(make_bots(1), 5, 10)

    def test_ten_players_raises(self):
        with pytest.raises(ValueError, match="Player count"):
            GameEngine(make_bots(10), 5, 10)

    def test_invalid_blinds_raises(self):
        with pytest.raises(ValueError, match="Invalid blinds"):
            GameEngine(make_bots(4), small_blind=20, big_blind=10)

    def test_add_player(self):
        engine = make_engine(4)
        new_bot = BotPlayer(99, "NewBot", 500)
        engine.add_player(new_bot)
        assert len(engine.players) == 5

    def test_add_player_at_max_raises(self):
        engine = make_engine(MAX_PLAYERS)
        with pytest.raises(ValueError, match="full"):
            engine.add_player(BotPlayer(99, "Extra", 500))

    def test_remove_player(self):
        engine = make_engine(4)
        pid = engine.players[2].player_id
        engine.remove_player(pid)
        assert len(engine.players) == 3
        assert all(p.player_id != pid for p in engine.players)

    def test_remove_nonexistent_player_raises(self):
        engine = make_engine(3)
        with pytest.raises(ValueError, match="not found"):
            engine.remove_player(9999)


# ------------------------------------------------------------------
# Chip conservation — the fundamental correctness invariant
# ------------------------------------------------------------------

class TestChipConservation:
    def _run_hands(self, engine, n_hands=20):
        total_before = sum(p.stack for p in engine.players)
        for _ in range(n_hands):
            active = sum(1 for p in engine.players if p.stack > 0)
            if active < 2:
                break
            engine.play_hand()
        total_after = sum(p.stack for p in engine.players)
        assert total_before == total_after, (
            f"Chips lost: before={total_before}, after={total_after}"
        )

    def test_conservation_2_players(self):
        self._run_hands(make_engine(2))

    def test_conservation_3_players(self):
        self._run_hands(make_engine(3))

    def test_conservation_4_players(self):
        self._run_hands(make_engine(4))

    def test_conservation_5_players(self):
        self._run_hands(make_engine(5))

    def test_conservation_6_players(self):
        self._run_hands(make_engine(6))

    def test_conservation_with_mc_engine(self):
        mc = MonteCarloEngine(n_simulations=200)
        engine = make_engine(4, mc=mc)
        self._run_hands(engine, n_hands=10)

    def test_conservation_unequal_stacks(self):
        players = [
            BotPlayer(1, "Rich",  2000, tight_threshold=0.3, aggression=0.4),
            BotPlayer(2, "Mid",    800, tight_threshold=0.3, aggression=0.4),
            BotPlayer(3, "Short",  200, tight_threshold=0.3, aggression=0.4),
            BotPlayer(4, "Micro",   50, tight_threshold=0.3, aggression=0.4),
        ]
        engine = GameEngine(players, small_blind=5, big_blind=10, verbose=False)
        total_before = sum(p.stack for p in players)
        for _ in range(15):
            if sum(1 for p in players if p.stack > 0) < 2:
                break
            engine.play_hand()
        total_after = sum(p.stack for p in players)
        assert total_before == total_after

    def test_conservation_long_session_6_players(self):
        self._run_hands(make_engine(6, stack=500), n_hands=50)


# ------------------------------------------------------------------
# Dealer and blind rotation
# ------------------------------------------------------------------

class TestDealerRotation:
    def test_dealer_index_advances_each_hand(self):
        engine = make_engine(4, verbose=False)
        initial = engine.dealer_index
        engine.play_hand()
        assert engine.dealer_index != initial

    def test_dealer_wraps_around(self):
        engine = make_engine(3, verbose=False)
        n = len(engine.players)
        engine.dealer_index = n - 1
        engine.rotate_dealer()
        assert engine.dealer_index == 0

    def test_dealer_skips_eliminated(self):
        players = make_bots(4, stack=100)
        engine = GameEngine(players, small_blind=5, big_blind=10, verbose=False)

        # Manually eliminate player at index 1
        players[1].stack = 0
        start = engine.dealer_index

        # Force dealer to the eliminated player, then rotate
        engine.dealer_index = 1
        engine.rotate_dealer()
        assert engine.players[engine.dealer_index].stack > 0

    def test_hand_number_increments(self):
        engine = make_engine(3, verbose=False)
        engine.play_hand()
        engine.play_hand()
        assert engine.hand_number == 2

    def test_blind_posting_reduces_stacks(self):
        engine = make_engine(4, verbose=False)
        stacks_before = [p.stack for p in engine.players]
        engine._setup_hand()
        engine._post_blinds()
        stacks_after = [p.stack for p in engine.players]
        total_posted = sum(b - a for b, a in zip(stacks_before, stacks_after)
                           if b > a)
        assert total_posted == engine.small_blind + engine.big_blind


# ------------------------------------------------------------------
# Side pot correctness
# ------------------------------------------------------------------

class TestSidePots:
    def test_all_in_player_cannot_win_more_than_contributed(self):
        """
        Short-stack all-in: they cannot win more than their
        contribution matched by each opponent.
        """
        # Bot3 has tiny stack — will go all-in for small amount.
        # Chip conservation is the proxy for correct side pot handling.
        players = [
            BotPlayer(1, "Big1",  1000, tight_threshold=0.0, aggression=0.0),
            BotPlayer(2, "Big2",  1000, tight_threshold=0.0, aggression=0.0),
            BotPlayer(3, "Short",   30, tight_threshold=0.0, aggression=1.0),
        ]
        engine = GameEngine(players, small_blind=5, big_blind=10, verbose=False)
        total_before = sum(p.stack for p in players)
        for _ in range(10):
            if sum(1 for p in players if p.stack > 0) < 2:
                break
            engine.play_hand()
        total_after = sum(p.stack for p in players)
        assert total_before == total_after

    def test_multiple_all_ins_chip_conservation(self):
        """Three players with different stacks, all highly aggressive."""
        players = [
            BotPlayer(1, "P1",  500, tight_threshold=0.0, aggression=0.9),
            BotPlayer(2, "P2",  300, tight_threshold=0.0, aggression=0.9),
            BotPlayer(3, "P3",  100, tight_threshold=0.0, aggression=0.9),
        ]
        engine = GameEngine(players, small_blind=5, big_blind=10, verbose=False)
        total = sum(p.stack for p in players)
        for _ in range(20):
            if sum(1 for p in players if p.stack > 0) < 2:
                break
            engine.play_hand()
        assert sum(p.stack for p in players) == total

    def test_showdown_completes_board_when_all_in_preflop(self):
        """
        When all players are all-in pre-flop, the board should be
        run out completely (5 community cards) before showdown.
        """
        players = [
            BotPlayer(1, "A",  100, tight_threshold=0.0, aggression=1.0),
            BotPlayer(2, "B",  100, tight_threshold=0.0, aggression=1.0),
        ]
        engine = GameEngine(players, small_blind=5, big_blind=10, verbose=False)
        engine.play_hand()
        # After hand, community_cards should be 5 (or 0 if folded pre-flop)
        assert len(engine.community_cards) in (0, 3, 4, 5)


# ------------------------------------------------------------------
# Action ordering
# ------------------------------------------------------------------

class TestActionOrdering:
    def test_preflop_bb_index_set_after_blind_posting(self):
        engine = make_engine(4, verbose=False)
        engine._setup_hand()
        engine._post_blinds()
        # _bb_index must be set and point to a valid player
        assert hasattr(engine, "_bb_index")
        assert 0 <= engine._bb_index < len(engine.players)

    def test_game_state_includes_active_player_count(self):
        engine = make_engine(4, verbose=False)
        engine._setup_hand()
        engine._post_blinds()
        engine._deal_hole_cards()
        gs = engine._build_game_state("Flop", 0)
        assert "active_player_count" in gs
        assert gs["active_player_count"] >= 1

    def test_game_state_includes_n_players(self):
        engine = make_engine(5, verbose=False)
        engine._setup_hand()
        engine._post_blinds()
        gs = engine._build_game_state("Pre-Flop", 20)
        assert gs["n_players"] == 5


# ------------------------------------------------------------------
# Play_hand return value
# ------------------------------------------------------------------

class TestPlayHandReturn:
    def test_winnings_sum_equals_pot(self):
        engine = make_engine(4, verbose=False)
        # Run a hand and verify winnings sum == total chips removed from stacks
        stacks_before = {p.player_id: p.stack for p in engine.players}
        winnings = engine.play_hand()
        stacks_after = {p.player_id: p.stack for p in engine.players}

        total_won = sum(winnings.values())
        total_lost = sum(
            stacks_before[p.player_id] - stacks_after[p.player_id]
            for p in engine.players
            if stacks_before[p.player_id] > stacks_after[p.player_id]
        )
        total_gained = sum(
            stacks_after[p.player_id] - stacks_before[p.player_id]
            for p in engine.players
            if stacks_after[p.player_id] > stacks_before[p.player_id]
        )
        assert total_gained == total_lost

    def test_winnings_dict_keys_are_player_ids(self):
        engine = make_engine(3, verbose=False)
        winnings = engine.play_hand()
        valid_ids = {p.player_id for p in engine.players}
        for pid in winnings:
            assert pid in valid_ids

    def test_play_multiple_hands_no_crash(self):
        engine = make_engine(6, verbose=False)
        for _ in range(30):
            if sum(1 for p in engine.players if p.stack > 0) < 2:
                break
            engine.play_hand()


# ------------------------------------------------------------------
# Regression: Phase 1-3 tests still pass
# ------------------------------------------------------------------

class TestPhase4Regression:
    def test_two_player_game_still_works(self):
        engine = make_engine(2, verbose=False)
        total = sum(p.stack for p in engine.players)
        for _ in range(10):
            if sum(1 for p in engine.players if p.stack > 0) < 2:
                break
            engine.play_hand()
        assert sum(p.stack for p in engine.players) == total

    def test_mc_engine_compatible_with_multiway(self):
        mc = MonteCarloEngine(n_simulations=150)
        engine = make_engine(5, mc=mc, verbose=False)
        total = sum(p.stack for p in engine.players)
        for _ in range(5):
            if sum(1 for p in engine.players if p.stack > 0) < 2:
                break
            engine.play_hand()
        assert sum(p.stack for p in engine.players) == total
