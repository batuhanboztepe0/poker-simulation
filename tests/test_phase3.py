"""
test_phase3.py
--------------
Tests for Phase 3: Pot Odds + EV Decision Making.

Pure function tests for ev_calculator are deterministic and exact.
BotPlayer integration tests verify rational decision-making under
controlled equity inputs (MC engine mocked where needed).

Run with: python -m pytest tests/test_phase3.py -v
"""

import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ev_calculator import (
    pot_odds,
    ev_call,
    ev_raise,
    should_call,
    optimal_raise_size,
    ev_summary,
    RAISE_SIZE_OPTIONS,
)
from src.card import Card
from src.player import BotPlayer
from src.game import GameEngine


# ===========================================================================
# pot_odds
# ===========================================================================

class TestPotOdds:
    def test_standard_case(self):
        # call 100 into pot 200 -> 100/300 = 0.3333
        assert pot_odds(100, 200) == pytest.approx(1/3, abs=1e-4)

    def test_zero_call_returns_zero(self):
        assert pot_odds(0, 500) == 0.0

    def test_half_pot_call(self):
        # call 100 into pot 100 -> 100/200 = 0.5
        assert pot_odds(100, 100) == pytest.approx(0.5, abs=1e-4)

    def test_large_overbet(self):
        # call 1000 into pot 100 -> 1000/1100 ~ 0.909
        assert pot_odds(1000, 100) == pytest.approx(1000/1100, abs=1e-4)

    def test_negative_call_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            pot_odds(-10, 100)

    def test_negative_pot_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            pot_odds(10, -100)

    def test_zero_pot_with_call_raises(self):
        with pytest.raises(ValueError, match="pot cannot be 0"):
            pot_odds(50, 0)

    def test_result_always_in_unit_interval(self):
        for call in [1, 10, 100, 999]:
            for pot in [1, 10, 100, 1000]:
                result = pot_odds(call, pot)
                assert 0.0 <= result <= 1.0


# ===========================================================================
# ev_call
# ===========================================================================

class TestEvCall:
    def test_break_even_call(self):
        # equity = pot_odds exactly -> EV = 0
        call, pot = 100, 200
        eq = pot_odds(call, pot)  # 1/3
        assert ev_call(eq, pot, call) == pytest.approx(0.0, abs=0.02)

    def test_profitable_call(self):
        # equity 0.6 > pot_odds 0.333 -> positive EV
        result = ev_call(0.6, 200, 100)
        assert result > 0

    def test_losing_call(self):
        # equity 0.2 < pot_odds 0.5 -> negative EV
        result = ev_call(0.2, 100, 100)
        assert result < 0

    def test_zero_call_amount(self):
        assert ev_call(0.5, 200, 0) == 0.0

    def test_full_equity_call(self):
        # equity = 1.0 -> EV = pot (we win everything)
        assert ev_call(1.0, 200, 100) == pytest.approx(200.0, abs=1e-3)

    def test_zero_equity_call(self):
        # equity = 0 -> EV = -call (we lose the call amount)
        assert ev_call(0.0, 200, 100) == pytest.approx(-100.0, abs=1e-3)

    def test_invalid_equity_raises(self):
        with pytest.raises(ValueError, match="equity"):
            ev_call(1.5, 100, 50)

    def test_negative_call_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ev_call(0.5, 100, -10)


# ===========================================================================
# ev_raise
# ===========================================================================

class TestEvRaise:
    def test_positive_ev_raise_high_equity(self):
        # Strong hand raising into small pot
        result = ev_raise(0.8, 100, 80)
        assert result > 0

    def test_negative_ev_raise_low_equity(self):
        # Weak hand raising large amount
        result = ev_raise(0.1, 100, 500)
        assert result < 0

    def test_zero_raise_raises(self):
        with pytest.raises(ValueError, match="positive"):
            ev_raise(0.5, 100, 0)

    def test_invalid_equity_raises(self):
        with pytest.raises(ValueError, match="equity"):
            ev_raise(-0.1, 100, 50)

    def test_ev_raise_monotone_in_equity(self):
        # Higher equity -> higher EV for same raise
        ev_low = ev_raise(0.3, 200, 100)
        ev_high = ev_raise(0.7, 200, 100)
        assert ev_high > ev_low


# ===========================================================================
# should_call
# ===========================================================================

class TestShouldCall:
    def test_call_when_equity_exceeds_pot_odds(self):
        # pot_odds = 100/300 = 0.333; equity = 0.5 -> call
        assert should_call(0.5, 200, 100) is True

    def test_fold_when_equity_below_pot_odds(self):
        # pot_odds = 100/200 = 0.5; equity = 0.3 -> fold
        assert should_call(0.3, 100, 100) is False

    def test_break_even_equity_calls(self):
        # equity exactly equals pot_odds -> EV = 0, still call (non-negative)
        call, pot = 100, 200
        eq = pot_odds(call, pot)
        assert should_call(eq, pot, call) is True

    def test_zero_call_always_calls(self):
        # Free to act -> always continue
        assert should_call(0.0, 200, 0) is True
        assert should_call(1.0, 200, 0) is True

    def test_all_in_very_good_pot_odds(self):
        # Call 10 into pot 1000 -> pot_odds = 10/1010 ~ 0.0099
        # Any equity > 1% should call
        assert should_call(0.05, 1000, 10) is True

    def test_call_required_for_huge_overbet(self):
        # Call 900 into pot 100: pot_odds = 0.9
        # Need 90% equity to call
        assert should_call(0.85, 100, 900) is False
        assert should_call(0.95, 100, 900) is True


# ===========================================================================
# optimal_raise_size
# ===========================================================================

class TestOptimalRaiseSize:
    def test_returns_value_in_valid_range(self):
        size = optimal_raise_size(0.7, 200, 40, 400)
        assert 40 <= size <= 400

    def test_min_equals_max_returns_that_value(self):
        size = optimal_raise_size(0.6, 100, 50, 50)
        assert size == 50

    def test_high_equity_prefers_larger_raise(self):
        # With very high equity, larger raises have higher EV
        size_strong = optimal_raise_size(0.9, 200, 20, 400)
        size_weak   = optimal_raise_size(0.3, 200, 20, 400)
        assert size_strong >= size_weak

    def test_invalid_min_raise_raises(self):
        with pytest.raises(ValueError, match="min_raise"):
            optimal_raise_size(0.5, 100, 0, 200)

    def test_max_less_than_min_raises(self):
        with pytest.raises(ValueError, match="max_raise"):
            optimal_raise_size(0.5, 100, 100, 50)

    def test_result_is_integer(self):
        size = optimal_raise_size(0.65, 150, 30, 300)
        assert isinstance(size, int)


# ===========================================================================
# ev_summary
# ===========================================================================

class TestEvSummary:
    def test_returns_all_keys(self):
        result = ev_summary(0.6, 200, 100, 40, 400)
        for key in ("fold", "call", "raise", "pot_odds",
                    "equity", "optimal_raise_size", "best_action"):
            assert key in result

    def test_fold_ev_always_zero(self):
        result = ev_summary(0.7, 200, 100, 40, 400)
        assert result["fold"] == 0.0

    def test_best_action_is_fold_when_equity_very_low(self):
        # 5% equity calling half pot -> definitely fold
        result = ev_summary(0.05, 200, 100, 40, 400)
        assert result["best_action"] == "fold"

    def test_best_action_raise_with_strong_hand_no_bet(self):
        # 85% equity, nothing to call -> raise should dominate
        result = ev_summary(0.85, 200, 0, 40, 400)
        assert result["best_action"] == "raise"

    def test_equity_in_result_matches_input(self):
        result = ev_summary(0.55, 100, 50, 20, 200)
        assert result["equity"] == pytest.approx(0.55, abs=1e-4)


# ===========================================================================
# BotPlayer EV-driven decisions (mocked equity)
# ===========================================================================

class TestBotPlayerEVDecisions:
    """
    These tests mock the MC engine to return controlled equity values,
    isolating the EV decision logic from simulation randomness.
    """

    def _make_bot(self, equity, tight_threshold=0.0, aggression=0.0):
        """Create a BotPlayer with a mocked MC engine returning fixed equity."""
        mc = MagicMock()
        mc.estimate_equity_unknown_opponents.return_value = equity
        bot = BotPlayer(
            player_id=1, name="TestBot", stack=1000,
            tight_threshold=tight_threshold,
            aggression=aggression,
            mc_engine=mc,
        )
        bot.receive_cards([Card("A", "s"), Card("K", "h")])
        return bot

    def _game_state(self, call_amount=100, pot=200, min_raise=40):
        return {
            "round_name": "Flop",
            "pot": pot,
            "call_amount": call_amount,
            "min_raise": min_raise,
            "current_bet": call_amount,
            "community_cards": [Card("Q","h"), Card("J","d"), Card("2","c")],
            "active_player_count": 2,
        }

    def test_folds_when_ev_negative(self):
        # equity=0.2, pot_odds=100/300=0.333 -> EV negative -> fold
        bot = self._make_bot(equity=0.2, aggression=0.0)
        action, _ = bot.decide(self._game_state(call_amount=100, pot=200))
        assert action == "fold"

    def test_calls_when_ev_positive(self):
        # equity=0.6, pot_odds=0.333 -> EV positive -> call
        bot = self._make_bot(equity=0.6, aggression=0.0)
        action, amount = bot.decide(self._game_state(call_amount=100, pot=200))
        assert action == "call"
        assert amount == 100

    def test_checks_when_no_bet_and_not_aggressive(self):
        bot = self._make_bot(equity=0.5, aggression=0.0)
        action, _ = bot.decide(self._game_state(call_amount=0, pot=200))
        assert action == "check"

    def test_raises_when_aggressive_and_no_bet(self):
        bot = self._make_bot(equity=0.7, aggression=1.0)
        action, amount = bot.decide(self._game_state(call_amount=0, pot=200))
        assert action == "raise"
        assert amount >= 40  # at least min_raise

    def test_tight_threshold_folds_despite_positive_ev(self):
        # equity=0.4 > pot_odds=0.333, but tight_threshold=0.5 -> fold
        bot = self._make_bot(equity=0.4, tight_threshold=0.5, aggression=0.0)
        action, _ = bot.decide(self._game_state(call_amount=100, pot=200))
        assert action == "fold"

    def test_all_in_when_call_exceeds_stack(self):
        bot = self._make_bot(equity=0.8, aggression=0.0)
        # call_amount > stack -> all_in
        gs = self._game_state(call_amount=2000, pot=500)
        action, amount = bot.decide(gs)
        assert action == "all_in"
        assert amount == bot.stack

    def test_raise_amount_is_valid(self):
        bot = self._make_bot(equity=0.75, aggression=1.0)
        gs = self._game_state(call_amount=0, pot=200, min_raise=40)
        action, amount = bot.decide(gs)
        assert action == "raise"
        assert amount >= 40
        assert amount <= bot.stack

    def test_ev_breakdown_returns_dict(self):
        bot = self._make_bot(equity=0.55)
        gs = self._game_state(call_amount=100, pot=200)
        breakdown = bot.ev_breakdown(gs)
        assert "best_action" in breakdown
        assert "pot_odds" in breakdown
        assert "equity" in breakdown

    def test_break_even_equity_calls(self):
        # equity exactly at pot_odds -> EV=0, should still call
        call, pot = 100, 200
        eq = pot_odds(call, pot)  # ~0.3333
        bot = self._make_bot(equity=eq, tight_threshold=0.0, aggression=0.0)
        action, _ = bot.decide(self._game_state(call_amount=call, pot=pot))
        assert action == "call"


# ===========================================================================
# GameEngine integration: chip conservation with Phase 3 bots
# ===========================================================================

class TestPhase3GameIntegration:
    def _make_engine(self, n_sims=200):
        from src.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(n_simulations=n_sims)
        players = [
            BotPlayer(i, f"EVBot{i}", 500,
                      tight_threshold=0.3,
                      aggression=0.4,
                      mc_engine=mc)
            for i in range(1, 3)
        ]
        return GameEngine(players, small_blind=5, big_blind=10, verbose=False)

    def test_chip_conservation_phase3(self):
        engine = self._make_engine()
        total_before = sum(p.stack for p in engine.players)
        for _ in range(10):
            if sum(1 for p in engine.players if p.stack > 0) < 2:
                break
            engine.play_hand()
        total_after = sum(p.stack for p in engine.players)
        assert total_before == total_after

    def test_phase1_bots_still_work(self):
        # Ensure Phase 1 bots (no MC, no EV) still run without errors
        players = [
            BotPlayer(i, f"P1Bot{i}", 500,
                      tight_threshold=0.4,
                      aggression=0.5)
            for i in range(1, 3)
        ]
        engine = GameEngine(players, small_blind=5, big_blind=10, verbose=False)
        total_before = sum(p.stack for p in players)
        for _ in range(5):
            if sum(1 for p in players if p.stack > 0) < 2:
                break
            engine.play_hand()
        total_after = sum(p.stack for p in players)
        assert total_before == total_after


# ===========================================================================
# A1: respect_pot_odds flag — close the +EV-fold leak in the tight gate
# ===========================================================================

class TestRespectPotOdds:
    """
    The tight_threshold style filter normally hard-folds every sub-threshold
    hand facing a bet, even one the pot prices in as a +EV call. The opt-in
    `respect_pot_odds` flag folds such a hand only when calling is ALSO -EV.
    Default-off keeps the baseline byte-identical.
    """

    def _make_bot(self, equity, tight_threshold, respect_pot_odds,
                  aggression=0.0):
        mc = MagicMock()
        mc.estimate_equity_unknown_opponents.return_value = equity
        bot = BotPlayer(
            player_id=1, name="PotOddsBot", stack=1000,
            tight_threshold=tight_threshold,
            aggression=aggression,
            mc_engine=mc,
            respect_pot_odds=respect_pot_odds,
        )
        bot.receive_cards([Card("A", "s"), Card("K", "h")])
        return bot

    def _game_state(self, call_amount=100, pot=200, min_raise=40):
        return {
            "round_name": "Flop",
            "pot": pot,
            "call_amount": call_amount,
            "min_raise": min_raise,
            "current_bet": call_amount,
            "community_cards": [Card("Q", "h"), Card("J", "d"), Card("2", "c")],
            "active_player_count": 2,
        }

    def test_default_off_hard_folds_plus_ev_marginal(self):
        # equity=0.4 > pot_odds=0.333 (a +EV call) but < tight_threshold=0.5.
        # Default behavior: the style filter still hard-folds it (the leak).
        bot = self._make_bot(equity=0.4, tight_threshold=0.5,
                             respect_pot_odds=False)
        action, _ = bot.decide(self._game_state(call_amount=100, pot=200))
        assert action == "fold"

    def test_respect_pot_odds_calls_plus_ev_marginal(self):
        # Same spot, flag on: the +EV marginal it used to fold is now called.
        bot = self._make_bot(equity=0.4, tight_threshold=0.5,
                             respect_pot_odds=True)
        action, amount = bot.decide(self._game_state(call_amount=100, pot=200))
        assert action == "call"
        assert amount == 100

    def test_respect_pot_odds_still_folds_junk(self):
        # equity=0.2 < pot_odds=0.333 and < tight_threshold: calling is -EV,
        # so junk below the pot-odds line is still folded with the flag on.
        bot = self._make_bot(equity=0.2, tight_threshold=0.5,
                             respect_pot_odds=True)
        action, _ = bot.decide(self._game_state(call_amount=100, pot=200))
        assert action == "fold"

    def test_respect_pot_odds_boundary_calls(self):
        # equity exactly at pot_odds (break-even) and below tight_threshold:
        # equity >= pot_odds, so the flag lets it call rather than hard-fold.
        call, pot = 100, 200
        eq = pot_odds(call, pot)  # ~0.3333
        bot = self._make_bot(equity=eq, tight_threshold=0.5,
                             respect_pot_odds=True)
        action, _ = bot.decide(self._game_state(call_amount=call, pot=pot))
        assert action == "call"

    def test_flag_does_not_affect_free_action(self):
        # No bet to call: the tight gate never applied; flag is a no-op here.
        bot = self._make_bot(equity=0.2, tight_threshold=0.5,
                             respect_pot_odds=True)
        action, _ = bot.decide(self._game_state(call_amount=0, pot=200))
        assert action == "check"

    def test_chip_conservation_with_flag_on(self):
        # The flag only changes fold/call routing, never chip accounting.
        from src.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(n_simulations=200)
        players = [
            BotPlayer(i, f"PO{i}", 500,
                      tight_threshold=0.5, aggression=0.4,
                      mc_engine=mc, respect_pot_odds=True)
            for i in range(1, 3)
        ]
        engine = GameEngine(players, small_blind=5, big_blind=10, verbose=False)
        total_before = sum(p.stack for p in engine.players)
        for _ in range(10):
            if sum(1 for p in engine.players if p.stack > 0) < 2:
                break
            engine.play_hand()
        total_after = sum(p.stack for p in engine.players)
        assert total_before == total_after
