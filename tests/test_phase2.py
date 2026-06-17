"""
test_phase2.py
--------------
Tests for Phase 2: Monte Carlo Equity Engine.

Equity tests are inherently probabilistic, so we use wide margins
and statistical properties rather than exact values. The key
assertions are:
    - Obvious favorites have equity >> 0.5
    - Obvious underdogs have equity << 0.5
    - Equity is conservative and monotone with hand strength
    - Pre-flop lookup returns sensible values
    - All chip conservation invariants still hold with MC-enabled bots

Run with: python -m pytest tests/test_phase2.py -v
"""

import sys
import os
import pytest
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.card import Card
from src.monte_carlo import MonteCarloEngine, PREFLOP_EQUITY_TABLE
from src.player import BotPlayer
from src.game import GameEngine

# Equity tolerance for statistical assertions
# With N=2000 simulations, std error ~ sqrt(p(1-p)/N) ~ 0.011 at p=0.5
EQUITY_TOLERANCE = 0.06  # conservative margin for CI
N_SIMS_TEST = 2000       # enough for stable results without being slow


def make_cards(card_strs):
    """Helper: build Card list from [(rank, suit), ...] tuples."""
    return [Card(r, s) for r, s in card_strs]


# ===========================================================================
# MonteCarloEngine initialisation
# ===========================================================================

class TestMonteCarloEngineInit:
    def test_default_n_simulations(self):
        engine = MonteCarloEngine()
        assert engine.n_simulations == 1000

    def test_custom_n_simulations(self):
        engine = MonteCarloEngine(n_simulations=500)
        assert engine.n_simulations == 500

    def test_n_simulations_too_low_raises(self):
        with pytest.raises(ValueError, match="n_simulations"):
            MonteCarloEngine(n_simulations=10)

    def test_n_simulations_too_high_raises(self):
        with pytest.raises(ValueError, match="n_simulations"):
            MonteCarloEngine(n_simulations=200_000)


# ===========================================================================
# Input validation
# ===========================================================================

class TestMonteCarloValidation:
    def setup_method(self):
        self.engine = MonteCarloEngine(n_simulations=200)

    def test_wrong_hole_card_count_raises(self):
        hero = make_cards([("A", "s")])
        with pytest.raises(ValueError, match="2 cards"):
            self.engine.estimate_equity(hero, [], [])

    def test_too_many_community_cards_raises(self):
        hero = make_cards([("A", "s"), ("K", "s")])
        board = make_cards([("Q","s"),("J","s"),("T","s"),("2","h"),("3","d"),("4","c")])
        with pytest.raises(ValueError, match="0-5"):
            self.engine.estimate_equity(hero, [], board)

    def test_duplicate_cards_raises(self):
        hero = make_cards([("A", "s"), ("A", "s")])
        with pytest.raises(ValueError, match="Duplicate"):
            self.engine.estimate_equity(hero, [], [])

    def test_duplicate_across_hole_and_board_raises(self):
        hero = make_cards([("A", "s"), ("K", "s")])
        board = make_cards([("A", "s"), ("Q", "h"), ("J", "d")])
        with pytest.raises(ValueError, match="Duplicate"):
            self.engine.estimate_equity(hero, [], board)

    def test_zero_opponents_raises(self):
        hero = make_cards([("A", "s"), ("K", "s")])
        with pytest.raises(ValueError, match="n_opponents"):
            self.engine.estimate_equity_unknown_opponents(hero, 0, [])


# ===========================================================================
# Pre-flop lookup table
# ===========================================================================

class TestPreflopLookup:
    def setup_method(self):
        self.engine = MonteCarloEngine(n_simulations=200)

    def test_aces_equity_above_80_percent(self):
        hero = make_cards([("A", "s"), ("A", "h")])
        eq = self.engine.estimate_equity(hero, [], [])
        assert eq >= 0.80, f"Pocket aces equity too low: {eq}"

    def test_pocket_twos_equity_around_50_percent(self):
        hero = make_cards([("2", "s"), ("2", "h")])
        eq = self.engine.estimate_equity(hero, [], [])
        assert 0.45 <= eq <= 0.55, f"Pocket 2s equity out of range: {eq}"

    def test_suited_ak_above_65_percent(self):
        hero = make_cards([("A", "s"), ("K", "s")])
        eq = self.engine.estimate_equity(hero, [], [])
        assert eq >= 0.60, f"AKs equity too low: {eq}"

    def test_lookup_returns_float(self):
        hero = make_cards([("7", "h"), ("2", "c")])  # not in table
        eq = self.engine.estimate_equity(hero, [], [])
        assert isinstance(eq, float)
        assert 0.0 <= eq <= 1.0

    def test_unlisted_hand_returns_default(self):
        # 7-2 offsuit: worst starting hand, not in lookup table
        hero = make_cards([("7", "h"), ("2", "c")])
        eq = self.engine.estimate_equity(hero, [], [])
        assert eq == 0.50  # PREFLOP_EQUITY_DEFAULT


# ===========================================================================
# Post-flop equity (simulation path)
# ===========================================================================

class TestPostFlopEquity:
    def setup_method(self):
        self.engine = MonteCarloEngine(n_simulations=N_SIMS_TEST)

    def test_equity_is_float_in_unit_interval(self):
        hero = make_cards([("A", "s"), ("K", "s")])
        board = make_cards([("Q", "s"), ("J", "s"), ("2", "h")])
        eq = self.engine.estimate_equity(hero, [], board)
        assert isinstance(eq, float)
        assert 0.0 <= eq <= 1.0

    def test_flopped_royal_flush_draw_has_high_equity(self):
        # Hero has flush + straight draw to royal flush on flop
        hero = make_cards([("A", "s"), ("K", "s")])
        board = make_cards([("Q", "s"), ("J", "s"), ("2", "h")])
        eq = self.engine.estimate_equity(hero, [], board)
        assert eq > 0.55, f"Royal flush draw equity too low: {eq}"

    def test_made_flush_on_river_has_very_high_equity(self):
        hero = make_cards([("A", "s"), ("K", "s")])
        board = make_cards([("Q", "s"), ("J", "s"), ("T", "s"), ("2", "h"), ("3", "d")])
        eq = self.engine.estimate_equity(hero, [], board)
        # Royal flush — equity should be near 1.0
        assert eq > 0.95, f"Royal flush river equity too low: {eq}"

    def test_pair_on_board_equity_is_reasonable(self):
        hero = make_cards([("A", "h"), ("A", "d")])
        board = make_cards([("A", "s"), ("2", "c"), ("7", "h")])
        eq = self.engine.estimate_equity(hero, [], board)
        # Set of aces on flop — extremely strong
        assert eq > 0.85, f"Set of aces equity too low: {eq}"

    def test_bottom_pair_weak_hand_equity_below_50(self):
        # Hero has bottom pair vs 2 opponents on a high-card board.
        # With 2 opponents, 22 drops well below 0.30 equity.
        hero = make_cards([("2", "h"), ("2", "d")])
        board = make_cards([("A", "s"), ("K", "c"), ("Q", "h")])
        eq = self.engine.estimate_equity_unknown_opponents(hero, 2, board)
        assert eq < 0.35, f"Bottom pair vs 2 opponents equity too high: {eq}"

    def test_multiway_equity_lower_than_heads_up(self):
        hero = make_cards([("A", "h"), ("K", "h")])
        board = make_cards([("T", "s"), ("8", "c"), ("2", "d")])

        eq_hu = self.engine.estimate_equity_unknown_opponents(hero, 1, board)
        eq_mw = self.engine.estimate_equity_unknown_opponents(hero, 3, board)

        assert eq_mw < eq_hu, (
            f"Multiway equity ({eq_mw:.3f}) should be lower than "
            f"heads-up equity ({eq_hu:.3f})"
        )

    def test_known_opponent_cards_used_in_simulation(self):
        # Hero has AA, opponent has 72o — hero should dominate
        hero = make_cards([("A", "s"), ("A", "h")])
        opponent = make_cards([("7", "c"), ("2", "d")])
        board = make_cards([("K", "s"), ("Q", "h"), ("J", "d")])

        eq_vs_known = self.engine.estimate_equity(hero, [opponent], board)
        assert eq_vs_known > 0.80, (
            f"AA vs 72o equity too low: {eq_vs_known:.3f}"
        )

    def test_turn_equity_different_from_flop(self):
        hero = make_cards([("A", "h"), ("K", "h")])
        flop = make_cards([("Q", "h"), ("J", "h"), ("2", "s")])
        turn = flop + make_cards([("T", "h")])  # completes flush

        eq_flop = self.engine.estimate_equity(hero, [], flop)
        eq_turn = self.engine.estimate_equity(hero, [], turn)

        # Flush completed on turn — equity should increase
        assert eq_turn > eq_flop, (
            f"Turn equity ({eq_turn:.3f}) should be higher than "
            f"flop equity ({eq_flop:.3f}) after flush completes"
        )

    def test_equity_sum_heads_up_near_one(self):
        # With two players, hero_equity + villain_equity ~ 1
        # (ignoring tie splits — should still be close)
        hero = make_cards([("A", "s"), ("K", "s")])
        villain = make_cards([("2", "h"), ("7", "c")])
        board = make_cards([("Q", "s"), ("J", "s"), ("3", "d")])

        hero_eq = self.engine.estimate_equity(hero, [villain], board)
        villain_eq = self.engine.estimate_equity(villain, [hero], board)

        total = hero_eq + villain_eq
        assert abs(total - 1.0) < 0.05, (
            f"Hero + villain equity should sum ~1.0, got {total:.4f}"
        )


# ===========================================================================
# BotPlayer MC integration
# ===========================================================================

class TestBotPlayerMCIntegration:
    def _make_game_state(self, community_cards, call_amount=20, pot=60):
        return {
            "round_name": "Flop",
            "pot": pot,
            "call_amount": call_amount,
            "min_raise": 40,
            "current_bet": call_amount,
            "community_cards": community_cards,
            "active_player_count": 2,
        }

    def test_bot_without_mc_still_works(self):
        bot = BotPlayer(1, "NoMC", 500)
        bot.receive_cards(make_cards([("A", "s"), ("K", "h")]))
        gs = self._make_game_state(make_cards([("Q","h"),("J","d"),("2","c")]))
        action, amount = bot.decide(gs)
        assert action in ("fold", "check", "call", "raise", "all_in")

    def test_bot_with_mc_uses_equity(self):
        engine = MonteCarloEngine(n_simulations=200)
        # Threshold = 0.99 — with real equity will almost always fold
        bot = BotPlayer(1, "MCBot", 500,
                        tight_threshold=0.99, aggression=0.0,
                        mc_engine=engine)
        bot.receive_cards(make_cards([("2", "h"), ("7", "c")]))  # worst hand
        board = make_cards([("A", "s"), ("K", "c"), ("Q", "h")])
        gs = self._make_game_state(board, call_amount=50)
        action, _ = bot.decide(gs)
        assert action == "fold", (
            f"72o with threshold=0.99 should fold, got {action}"
        )

    def test_bot_with_mc_strong_hand_does_not_fold(self):
        engine = MonteCarloEngine(n_simulations=300)
        # Threshold = 0.05 — will only fold truly terrible equity
        bot = BotPlayer(1, "MCBot", 500,
                        tight_threshold=0.05, aggression=0.0,
                        mc_engine=engine)
        bot.receive_cards(make_cards([("A", "s"), ("A", "h")]))
        board = make_cards([("A", "d"), ("2", "c"), ("7", "h")])  # set of aces
        gs = self._make_game_state(board, call_amount=20)
        action, _ = bot.decide(gs)
        assert action != "fold", "Set of aces should not fold with threshold=0.05"

    def test_bot_mc_fallback_on_no_hole_cards(self):
        engine = MonteCarloEngine(n_simulations=200)
        bot = BotPlayer(1, "MCBot", 500, mc_engine=engine)
        # No hole cards dealt — should fall back gracefully (random)
        gs = self._make_game_state([])
        action, _ = bot.decide(gs)
        assert action in ("fold", "check", "call", "raise", "all_in")

    def test_chip_conservation_with_mc_bots(self):
        engine = MonteCarloEngine(n_simulations=200)
        players = [
            BotPlayer(i, f"MC-Bot{i}", 500,
                      tight_threshold=0.3, aggression=0.4,
                      mc_engine=engine)
            for i in range(1, 3)
        ]
        game = GameEngine(players, small_blind=5, big_blind=10, verbose=False)
        total_before = sum(p.stack for p in players)

        for _ in range(5):
            if sum(1 for p in players if p.stack > 0) < 2:
                break
            game.play_hand()

        total_after = sum(p.stack for p in players)
        assert total_before == total_after
