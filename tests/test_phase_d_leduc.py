"""
test_phase_d_leduc.py
---------------------
Tests for Phase D: CFR / GTO equilibrium solver on Leduc Hold'em.

Leduc Hold'em has a known numerically-verified Nash game value:
    - player-0 game value = LEDUC_GAME_VALUE ≈ -0.032913
    - P0 is disadvantaged as first mover
    - A pair (private card matches board) warrants more aggressive play in round 2

Run with: python -m pytest tests/test_phase_d_leduc.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.leduc_cfr import LeducCFR, LEDUC_GAME_VALUE


@pytest.fixture(scope="module")
def trained():
    cfr = LeducCFR()
    value = cfr.train(1000)
    return cfr, value


class TestGameValue:
    def test_converges_to_expected_value(self, trained):
        _cfr, value = trained
        assert abs(value - LEDUC_GAME_VALUE) < 0.005

    def test_value_constant_is_negative(self):
        # P0 is disadvantaged as first mover
        assert LEDUC_GAME_VALUE < 0

    def test_deterministic(self):
        v1 = LeducCFR().train(200)
        v2 = LeducCFR().train(200)
        assert v1 == v2  # no RNG -> bit-identical

    def test_all_information_sets_discovered(self, trained):
        cfr, _ = trained
        # 288 Leduc info sets: 18 round-1, 270 round-2
        assert len(cfr.strategy_table()) == 288


class TestEquilibriumStrategy:
    def test_strategies_are_probability_distributions(self, trained):
        cfr, _ = trained
        for info_set, dist in cfr.strategy_table().items():
            assert abs(sum(dist) - 1.0) < 1e-9, (
                f"Probabilities don't sum to 1.0 at {info_set}: {dist}"
            )
            assert all(0.0 <= p <= 1.0 for p in dist), (
                f"Probability out of [0,1] at {info_set}: {dist}"
            )

    def test_pair_bets_more_than_weaker_hand_in_round2(self, trained):
        cfr, _ = trained
        table = cfr.strategy_table()

        # Collect round-2 info sets for pair vs non-pair situations
        # Round-2 key format: "<private><board><r1_terminal>/<r2_history>"
        # At round-2 start (r2_history == ''), player acts first (CALL or RAISE)
        pair_raise_probs = []
        nonpair_raise_probs = []

        for key, dist in table.items():
            if '/' not in key:
                continue
            # private card is key[0], board card is key[1]
            private = int(key[0])
            board = int(key[1])
            slash_pos = key.index('/')
            r2_hist = key[slash_pos + 1:]
            # Only look at round-2 start positions where the player has CALL or RAISE
            # (i.e., not facing a bet, so available = [CALL, RAISE])
            if r2_hist == '' or (len(r2_hist) > 0 and r2_hist[-1] == 'c' and r2_hist.count('r') == 0):
                # First action in round 2 or after a mutual check — player not facing a bet
                from src.leduc_cfr import _available_actions, RAISE
                avail = _available_actions(r2_hist, in_r2=True)
                if RAISE in avail and 0 in avail:  # [CALL, RAISE] not [FOLD, CALL, RAISE]
                    raise_prob = dist[RAISE]
                    if private == board:
                        pair_raise_probs.append(raise_prob)
                    else:
                        nonpair_raise_probs.append(raise_prob)

        if pair_raise_probs and nonpair_raise_probs:
            avg_pair_raise = sum(pair_raise_probs) / len(pair_raise_probs)
            avg_nonpair_raise = sum(nonpair_raise_probs) / len(nonpair_raise_probs)
            assert avg_pair_raise >= avg_nonpair_raise, (
                f"Expected pairs to raise more: pair={avg_pair_raise:.3f}, "
                f"nonpair={avg_nonpair_raise:.3f}"
            )


class TestConvergence:
    def test_more_iterations_closer_to_equilibrium(self):
        err_short = abs(LeducCFR().train(50) - LEDUC_GAME_VALUE)
        err_long = abs(LeducCFR().train(500) - LEDUC_GAME_VALUE)
        assert err_long <= err_short


class TestExploitability:
    def test_exploitability_is_nonnegative(self, trained):
        cfr, _ = trained
        assert cfr.exploitability() >= 0.0

    def test_exploitability_is_small_after_training(self, trained):
        # Info-set-respecting best response: after 1000 CFR iterations the
        # average strategy is close to the Nash equilibrium, so the most either
        # player can gain by deviating is small. (This is the real
        # "unexploitable benchmark" guarantee; a clairvoyant per-deal best
        # response would never reach zero.)
        cfr, _ = trained
        assert cfr.exploitability() < 0.03

    def test_exploitability_decreases_with_training(self, trained):
        # More iterations -> closer to equilibrium -> strictly less exploitable.
        cfr_long, _ = trained          # 1000 iterations (module fixture)
        cfr_short = LeducCFR()
        cfr_short.train(80)
        assert cfr_long.exploitability() < cfr_short.exploitability()


class TestGeneralExploitability:
    """src/leduc_eval: exact exploitability of an ARBITRARY strategy, and the
    average-vs-last-iterate convergence contrast."""

    def test_tool_matches_cfr_on_its_own_average(self, trained):
        from src.leduc_eval import exploitability_of
        cfr, _ = trained
        # Scoring CFR's own average strategy via the general tool must reproduce
        # CFR's internal exploitability exactly (validates the fake-node reuse).
        assert abs(exploitability_of(cfr.strategy_table())
                   - cfr.exploitability()) < 1e-9

    def test_uniform_is_much_more_exploitable(self, trained):
        from src.leduc_eval import (exploitability_of, uniform_strategy_table)
        cfr, _ = trained
        assert (exploitability_of(uniform_strategy_table())
                > 10 * cfr.exploitability())

    def test_last_iterate_stays_exploitable(self, trained):
        # The greedy last-iterate (the DQN-self-play regime) is far more
        # exploitable than the time-average even though both come from the SAME
        # converged CFR run -- averaging is what reaches Nash.
        from src.leduc_eval import exploitability_of, current_strategy_table
        cfr, _ = trained
        assert (exploitability_of(current_strategy_table(cfr))
                > 5 * cfr.exploitability())


class TestQLearningSelfPlay:
    """src/leduc_q: the DQN-regime tabular Q-learner. Its greedy last-iterate
    does NOT converge to Nash -- a direct measurement of the same non-convergence
    that CFR's last-iterate shows, with an INDEPENDENT value-based learner."""

    def test_greedy_table_valid_and_covers_all_info_sets(self):
        from src.leduc_q import LeducQLearner
        from src.leduc_eval import uniform_strategy_table
        ql = LeducQLearner(seed=0)
        ql.train(2000)
        table = ql.greedy_strategy_table()
        assert set(table) == set(uniform_strategy_table())   # every info-set
        for dist in table.values():
            assert abs(sum(dist) - 1.0) < 1e-9 and all(p >= 0.0 for p in dist)

    def test_deterministic(self):
        from src.leduc_q import LeducQLearner
        from src.leduc_eval import exploitability_of
        a = LeducQLearner(seed=0); a.train(3000)
        b = LeducQLearner(seed=0); b.train(3000)
        assert (exploitability_of(a.greedy_strategy_table())
                == exploitability_of(b.greedy_strategy_table()))

    def test_greedy_last_iterate_does_not_reach_nash(self):
        # CFR's time-average is < 0.03 after 1000 iters; the independent greedy
        # Q-learner's last-iterate stays an order of magnitude more exploitable
        # (averaged over snapshots, to be robust to its oscillation).
        from src.leduc_q import LeducQLearner
        from src.leduc_eval import exploitability_of
        ql, vals = LeducQLearner(seed=0), []
        for _ in range(5):
            ql.train(4000)
            vals.append(exploitability_of(ql.greedy_strategy_table()))
        assert sum(vals) / len(vals) > 0.2
