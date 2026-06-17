"""
test_phase_a.py
---------------
Tests for Phase A: fold equity & strategic bluffing.

DoD (ROADMAP §9):
    - pure-bluff break-even matches b/(pot+b) to 1e-6,
    - p_fold=0 path is byte-identical to the baseline,
    - the fold-equity bot picks LARGER raises than a p_fold=0 bot,
    - a low-equity/high-p_fold spot raises while low-p_fold folds
      (bluffing is EV-gated, not random).

Run with: python -m pytest tests/test_phase_a.py -v
"""

import sys
import os
import random
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.card import Card
from src.player import BotPlayer
from src.game import GameEngine
import src.fold_equity as fe
from src.fold_equity import FoldEquityModel, regularized_incomplete_beta
from src.opponent_model import BeliefState
from src.ev_calculator import (
    ev_raise, ev_raise_with_fold_equity, optimal_raise_size,
    optimal_raise_size_with_fold_equity, gto_bluff_ratio,
    minimum_defense_frequency, breakeven_pfold,
)


def make_cards(card_strs):
    return [Card(r, s) for r, s in card_strs]


# Controllable fold models (duck-typed for BotPlayer).
class ConstFoldModel:
    def __init__(self, p):
        self.p = p

    def estimate_p_fold(self, opponent_ids, raise_amount, pot, n_opponents=None):
        return self.p


class OverFoldModel:
    """Opponent that folds MORE than GTO (exploitable by larger bets)."""
    def __init__(self, k):
        self.k = k

    def estimate_p_fold(self, opponent_ids, raise_amount, pot, n_opponents=None):
        return min(0.99, self.k * raise_amount / (pot + raise_amount))


class ZeroFoldModel:
    def estimate_p_fold(self, opponent_ids, raise_amount, pot, n_opponents=None):
        return 0.0


# ===========================================================================
# EV math
# ===========================================================================

class TestFoldEquityMath:
    def test_gto_bluff_ratio(self):
        assert gto_bluff_ratio(100, 200) == pytest.approx(100 / 300, abs=1e-4)

    def test_mdf_is_complement(self):
        b, pot = 100, 200
        assert (gto_bluff_ratio(b, pot)
                + minimum_defense_frequency(b, pot)) == pytest.approx(1.0,
                                                                      abs=1e-4)

    def test_pure_bluff_breakeven_matches_gto_ratio(self):
        for b, pot in [(50, 100), (100, 100), (200, 75), (33, 100)]:
            assert abs(breakeven_pfold(0.0, pot, b)
                       - gto_bluff_ratio(b, pot)) < 1e-6

    def test_pure_bluff_breaks_even_at_gto_pfold(self):
        b, pot = 80, 120
        p = gto_bluff_ratio(b, pot)
        assert ev_raise_with_fold_equity(0.0, pot, b, p) == pytest.approx(
            0.0, abs=1e-3)

    def test_pfold_zero_is_byte_identical_to_baseline(self):
        for eq in (0.1, 0.4, 0.85):
            for b in (20, 50, 200):
                assert (ev_raise_with_fold_equity(eq, 100, b, 0.0)
                        == ev_raise(eq, 100, b))

    def test_optimal_size_pfold_zero_matches_baseline(self):
        zero = lambda b: 0.0
        for eq in (0.2, 0.6, 0.9):
            assert (optimal_raise_size_with_fold_equity(eq, 200, 20, 400, zero)
                    == optimal_raise_size(eq, 200, 20, 400))

    def test_ev_increases_with_pfold(self):
        # When the called-EV is below pot, more folds is strictly better.
        low = ev_raise_with_fold_equity(0.2, 100, 50, 0.1)
        high = ev_raise_with_fold_equity(0.2, 100, 50, 0.8)
        assert high > low

    def test_breakeven_pfold_consistent_with_ev(self):
        # At the break-even fold probability, the raise EV must be ~0 (this is
        # the regression for the corrected (pot+b)(1-eq) denominator).
        for eq in (0.1, 0.2, 0.3):
            for pot, b in [(100, 50), (200, 120), (150, 200)]:
                p = breakeven_pfold(eq, pot, b)
                if p > 0:
                    assert ev_raise_with_fold_equity(eq, pot, b, p) == pytest.approx(
                        0.0, abs=0.05)

    def test_breakeven_pfold_monotone_in_equity(self):
        # Higher equity needs less fold equity to break even.
        hi_eq = breakeven_pfold(0.6, 100, 80)
        lo_eq = breakeven_pfold(0.1, 100, 80)
        assert hi_eq < lo_eq

    def test_overfolding_opponent_favors_larger_raise(self):
        pfold = lambda b: min(0.99, 2.0 * b / (100 + b))
        size = optimal_raise_size_with_fold_equity(0.3, 100, 20, 400, pfold)
        # GTO/no-fold-equity baseline always picks the minimum.
        assert size > optimal_raise_size(0.3, 100, 20, 400)


# ===========================================================================
# Regularized incomplete beta
# ===========================================================================

class TestIncompleteBeta:
    def test_uniform_cdf(self):
        # I_x(1, 1) = x
        for x in (0.1, 0.5, 0.9):
            assert regularized_incomplete_beta(1, 1, x) == pytest.approx(x,
                                                                         abs=1e-9)

    def test_symmetric_midpoint(self):
        assert regularized_incomplete_beta(2, 2, 0.5) == pytest.approx(0.5,
                                                                       abs=1e-9)

    def test_monotone_in_x(self):
        vals = [regularized_incomplete_beta(2, 3, x)
                for x in (0.1, 0.3, 0.6, 0.9)]
        assert vals == sorted(vals)

    def test_looser_opponent_has_smaller_cdf(self):
        # Beta(5,2) mass toward 1 -> small CDF at 0.3 (folds less).
        loose = regularized_incomplete_beta(5, 2, 0.3)
        tight = regularized_incomplete_beta(2, 5, 0.3)
        assert loose < tight

    def test_fallback_matches_scipy(self):
        if not fe._HAVE_SCIPY:
            pytest.skip("scipy not installed")
        from scipy.special import betainc
        orig = fe._HAVE_SCIPY
        try:
            fe._HAVE_SCIPY = False  # force pure-Python branch
            for a, b, x in [(2, 3, 0.4), (5, 2, 0.3), (1.5, 4.0, 0.7),
                            (10, 3, 0.25)]:
                assert abs(regularized_incomplete_beta(a, b, x)
                           - float(betainc(a, b, x))) < 1e-9
        finally:
            fe._HAVE_SCIPY = orig


# ===========================================================================
# FoldEquityModel
# ===========================================================================

class TestFoldEquityModel:
    def test_neutral_prior_no_beliefs(self):
        model = FoldEquityModel()
        # single opponent, no belief -> GTO neutral b/(pot+b)
        p = model.estimate_p_fold([1], raise_amount=100, pot=200)
        assert p == pytest.approx(100 / 300, abs=1e-4)

    def test_multiway_is_product(self):
        model = FoldEquityModel()
        p1 = model.estimate_p_fold([1], 100, 200)
        p2 = model.estimate_p_fold([1, 2], 100, 200)
        assert p2 == pytest.approx(p1 * p1, abs=1e-4)
        assert p2 < p1

    def test_looser_belief_folds_less(self):
        loose = BeliefState()
        for _ in range(40):
            loose.update("call", 0, 1)   # high looseness
        tight = BeliefState()
        for _ in range(40):
            tight.update("fold", 0, 1)    # low looseness
        m_loose = FoldEquityModel({1: loose})
        m_tight = FoldEquityModel({1: tight})
        assert (m_loose.estimate_p_fold([1], 100, 200)
                < m_tight.estimate_p_fold([1], 100, 200))

    def test_bigger_bet_folds_more(self):
        tight = BeliefState()
        for _ in range(40):
            tight.update("fold", 0, 1)
        model = FoldEquityModel({1: tight})
        small = model.estimate_p_fold([1], 50, 200)
        big = model.estimate_p_fold([1], 400, 200)
        assert big > small


# ===========================================================================
# BotPlayer integration
# ===========================================================================

class TestFoldEquityBot:
    def _bot(self, equity, fold_model, tight=0.0, aggression=1.0):
        mc = MagicMock()
        mc.estimate_equity_unknown_opponents.return_value = equity
        bot = BotPlayer(1, "FE", 400, tight_threshold=tight,
                        aggression=aggression, mc_engine=mc,
                        fold_equity_model=fold_model)
        bot.receive_cards(make_cards([("K", "d"), ("Q", "c")]))
        return bot

    def _state(self, call_amount, pot=100, min_raise=40):
        return {
            "round_name": "Flop",
            "pot": pot,
            "call_amount": call_amount,
            "min_raise": min_raise,
            "current_bet": call_amount,
            "community_cards": make_cards([("2", "s"), ("7", "h"), ("9", "d")]),
            "active_player_count": 2,
            "opponent_ids": [2],
        }

    def test_fold_equity_bot_raises_larger_than_pfold_zero(self):
        # Free action; both raise, but the over-folding model bets bigger.
        bot_fe = self._bot(0.5, OverFoldModel(2.0))
        bot_zero = self._bot(0.5, ZeroFoldModel())
        act_fe, size_fe = bot_fe.decide(self._state(call_amount=0))
        act_zero, size_zero = bot_zero.decide(self._state(call_amount=0))
        assert act_fe == "raise" and act_zero == "raise"
        assert size_fe > size_zero

    def test_high_pfold_bluffs_low_equity(self):
        # Low equity, facing a bet, but opponents fold a lot -> raise (bluff).
        bot = self._bot(0.2, ConstFoldModel(0.85))
        action, _ = bot.decide(self._state(call_amount=50))
        assert action == "raise"

    def test_low_pfold_folds_low_equity(self):
        # Same low equity, but opponents rarely fold -> the bluff is -EV, fold.
        bot = self._bot(0.2, ConstFoldModel(0.1))
        action, _ = bot.decide(self._state(call_amount=50))
        assert action == "fold"

    def test_baseline_bot_unaffected(self):
        # No fold-equity model -> baseline flow: -EV call folds.
        mc = MagicMock()
        mc.estimate_equity_unknown_opponents.return_value = 0.2
        bot = BotPlayer(1, "Base", 400, tight_threshold=0.0, aggression=0.0,
                        mc_engine=mc)
        bot.receive_cards(make_cards([("K", "d"), ("Q", "c")]))
        action, _ = bot.decide(self._state(call_amount=50))
        assert action == "fold"

    def test_chip_conservation_with_fold_equity_bots(self):
        rng = random.Random(3)
        from src.monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(n_simulations=120, rng=rng)
        players = []
        for i in range(1, 4):
            belief = BeliefState()
            players.append(BotPlayer(
                i, f"B{i}", 1000, 0.2, 0.6, mc, rng,
                belief_state=belief,
                fold_equity_model=FoldEquityModel({}),
            ))
        engine = GameEngine(players, 10, 20, verbose=False, rng=rng)
        total = sum(p.stack for p in players)
        for _ in range(30):
            if sum(1 for p in players if p.stack > 0) < 2:
                break
            engine.play_hand()
        assert sum(p.stack for p in players) == total
