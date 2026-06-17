"""
test_phase_d.py
---------------
Tests for Phase D: CFR / GTO equilibrium solver on Kuhn poker.

Kuhn poker has a known closed-form Nash equilibrium, so CFR's average strategy
and game value are exact unit-test targets:
    - player-1 game value = -1/18,
    - player 1 never bets the Queen first,
    - the Jack is bluffed with probability xi in [0, 1/3],
    - the King is bet with probability 3*xi (the 3:1 value/bluff ratio),
    - facing a bet: fold the Jack, always call the King, call the Queen ~1/3.

Run with: python -m pytest tests/test_phase_d.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cfr import KuhnCFR, KUHN_GAME_VALUE


@pytest.fixture(scope="module")
def trained():
    cfr = KuhnCFR()
    value = cfr.train(8000)
    return cfr, value


class TestGameValue:
    def test_converges_to_minus_one_eighteenth(self, trained):
        _cfr, value = trained
        assert abs(value - KUHN_GAME_VALUE) < 0.01

    def test_value_constant_is_correct(self):
        assert KUHN_GAME_VALUE == pytest.approx(-1.0 / 18.0)

    def test_deterministic(self):
        v1 = KuhnCFR().train(500)
        v2 = KuhnCFR().train(500)
        assert v1 == v2  # no randomness -> bit-identical

    def test_all_information_sets_discovered(self, trained):
        cfr, _ = trained
        # 12 Kuhn info sets: 3 cards x {"", "b", "p", "pb"}
        assert len(cfr.strategy_table()) == 12


class TestEquilibriumStrategy:
    def test_queen_never_bets_first(self, trained):
        cfr, _ = trained
        assert cfr.bet_probability("2") < 0.05

    def test_jack_bluff_within_bound(self, trained):
        cfr, _ = trained
        jack = cfr.bet_probability("1")
        assert 0.0 <= jack <= 0.34  # xi in [0, 1/3]

    def test_king_bets_three_times_jack(self, trained):
        cfr, _ = trained
        jack = cfr.bet_probability("1")
        king = cfr.bet_probability("3")
        assert king > jack
        # The Kuhn equilibrium fixes P(bet|K) = 3 * P(bet|J).
        assert king / jack == pytest.approx(3.0, abs=0.6)

    def test_facing_bet_fold_jack_call_king(self, trained):
        cfr, _ = trained
        # "Xb" = facing a bet; the second action bet==call, pass==fold.
        assert cfr.bet_probability("1b") < 0.05   # fold the Jack
        assert cfr.bet_probability("3b") > 0.95   # always call the King

    def test_facing_bet_call_queen_one_third(self, trained):
        cfr, _ = trained
        assert cfr.bet_probability("2b") == pytest.approx(1.0 / 3.0, abs=0.12)

    def test_strategies_are_probability_distributions(self, trained):
        cfr, _ = trained
        for _info_set, dist in cfr.strategy_table().items():
            assert abs(sum(dist) - 1.0) < 1e-9
            assert all(0.0 <= p <= 1.0 for p in dist)


class TestConvergence:
    def test_more_iterations_closer_to_equilibrium(self):
        err_short = abs(KuhnCFR().train(100) - KUHN_GAME_VALUE)
        err_long = abs(KuhnCFR().train(4000) - KUHN_GAME_VALUE)
        assert err_long <= err_short
