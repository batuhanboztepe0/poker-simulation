"""
test_icm_eval.py
----------------
Tests for Phase A5: the multi-player ICM tournament evaluation harness
(`src/icm_eval.py`).

DoD:
    - tournament_prizes assigns finish-order payouts and conserves the pool,
    - survivors split the top prizes by ICM equity of their stacks,
    - play_tournament conserves chips across a multi-hand bankroll tournament,
    - evaluate_icm_tournament is seed-deterministic and pool-conserving, and
      unbiased for identical agents (seat rotation cancels position),
    - the harness runs with trained RL agents (torch-guarded).

Run with: python -m pytest tests/test_icm_eval.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.rl_agent as rl
from src.player import BotPlayer
from src.icm_eval import (
    tournament_prizes,
    play_tournament,
    evaluate_icm_tournament,
)
from src.monte_carlo import MonteCarloEngine

PRIZE = [50.0, 30.0, 20.0]


def _myopic_factory(pid, name, stack, mc, rng):
    return BotPlayer(pid, name, stack, tight_threshold=0.2, aggression=0.5,
                     mc_engine=mc, rng=rng)


# ===========================================================================
# tournament_prizes — finish-order payouts (pure, no torch)
# ===========================================================================

class TestTournamentPrizes:
    def test_all_live_equal_stacks_split_pool_evenly(self):
        out = tournament_prizes([1, 2, 3], {1: 1000, 2: 1000, 3: 1000}, [], PRIZE)
        assert out[1] == pytest.approx(100 / 3)
        assert out[2] == pytest.approx(100 / 3)
        assert out[3] == pytest.approx(100 / 3)

    def test_full_bustout_distinguishes_second_and_third(self):
        # 3 busted first (last place=20), 2 busted second (2nd=30), 1 survives (50).
        out = tournament_prizes([1, 2, 3], {1: 3000, 2: 0, 3: 0}, [3, 2], PRIZE)
        assert out == {1: 50.0, 2: 30.0, 3: 20.0}

    def test_partial_bustout_survivors_split_top_prizes_by_icm(self):
        # 3 busted (gets last place=20); 1 and 2 still live with equal stacks
        # split the top two prizes (50, 30) by ICM -> 40 each.
        out = tournament_prizes([1, 2, 3], {1: 1500, 2: 1500, 3: 0}, [3], PRIZE)
        assert out[3] == 20.0
        assert out[1] == pytest.approx(40.0)
        assert out[2] == pytest.approx(40.0)

    def test_single_survivor_with_chip_lead_among_two_live(self):
        # No bustouts (hand cap reached): all three split by ICM of final stacks.
        out = tournament_prizes([1, 2, 3], {1: 2000, 2: 700, 3: 300}, [], PRIZE)
        # Chip leader earns the most; pool conserved; ordering follows stacks.
        assert out[1] > out[2] > out[3]
        assert sum(out.values()) == pytest.approx(100.0)

    @pytest.mark.parametrize("final,bust", [
        ({1: 3000, 2: 0, 3: 0}, [2, 3]),
        ({1: 1500, 2: 1500, 3: 0}, [3]),
        ({1: 1200, 2: 1000, 3: 800}, []),
        ({1: 0, 2: 0, 3: 3000}, [1, 2]),
    ])
    def test_prize_pool_always_conserved(self, final, bust):
        out = tournament_prizes([1, 2, 3], final, bust, PRIZE)
        assert sum(out.values()) == pytest.approx(sum(PRIZE))


# ===========================================================================
# play_tournament — chip conservation across a multi-hand bankroll tournament
# ===========================================================================

class TestPlayTournament:
    def test_chip_conservation_over_tournament(self):
        import random
        rng = random.Random(3)
        mc = MonteCarloEngine(n_simulations=100, rng=rng)
        players = [_myopic_factory(i, f"M{i}", 1000, mc, rng) for i in (1, 2, 3)]
        play_tournament(players, PRIZE, max_hands=60, rng=rng)
        # Chips are neither created nor destroyed: survivors hold the busted
        # players' chips, total == n * stack0.
        assert sum(p.stack for p in players) == 3 * 1000

    def test_prizes_returned_for_every_seat(self):
        import random
        rng = random.Random(5)
        mc = MonteCarloEngine(n_simulations=100, rng=rng)
        players = [_myopic_factory(i, f"M{i}", 1000, mc, rng) for i in (1, 2, 3)]
        prizes = play_tournament(players, PRIZE, max_hands=60, rng=rng)
        assert set(prizes) == {1, 2, 3}
        assert sum(prizes.values()) == pytest.approx(sum(PRIZE))


# ===========================================================================
# evaluate_icm_tournament — seeded, paired, seat-rotated
# ===========================================================================

class TestEvaluateIcmTournament:
    def test_deterministic_across_runs(self):
        a = evaluate_icm_tournament([_myopic_factory] * 3, PRIZE,
                                    seeds=range(4), mc_sims=100, max_hands=40)
        b = evaluate_icm_tournament([_myopic_factory] * 3, PRIZE,
                                    seeds=range(4), mc_sims=100, max_hands=40)
        assert a == b

    def test_per_seed_pool_conserved(self):
        names = ["X", "Y", "Z"]
        res = evaluate_icm_tournament([_myopic_factory] * 3, PRIZE,
                                      seeds=range(6), names=names,
                                      mc_sims=100, max_hands=40)
        # Each seed's three prizes sum to the pool, so the agents' per-seed
        # totals line up to sum(PRIZE) seed by seed.
        per_seed_totals = [sum(res[n]["per_seed"][k] for n in names)
                           for k in range(6)]
        assert all(t == pytest.approx(sum(PRIZE)) for t in per_seed_totals)

    def test_identical_agents_mean_prizes_sum_to_pool(self):
        res = evaluate_icm_tournament([_myopic_factory] * 3, PRIZE,
                                      seeds=range(12), mc_sims=100, max_hands=40)
        total = sum(v["mean_prize"] for v in res.values())
        assert total == pytest.approx(sum(PRIZE))


# ===========================================================================
# End-to-end with trained RL agents (torch-guarded, bounded)
# ===========================================================================

def _rl_factory(qnet):
    def factory(pid, name, stack, mc, rng):
        return rl.RLBotPlayer(pid, name, stack, qnet=qnet, epsilon=0.0,
                              training=False, mc_engine=mc, rng=rng)
    return factory


@pytest.mark.skipif(not rl._HAVE_TORCH, reason="torch not installed")
class TestIcmEvalWithRLAgents:
    def test_harness_runs_and_conserves_pool_with_rl_agents(self):
        total = 3 * 1000
        prize = [total * 0.5, total * 0.3, total * 0.2]

        icm = rl.SelfPlayTrainer(
            n_players=3, hidden=32, seed=1, opponent_mode="fixed",
            multi_hand=True, hands_per_episode=10,
            icm_prize_structure=prize, mc_sims=100)
        icm.train(20)

        chips = rl.SelfPlayTrainer(
            n_players=3, hidden=32, seed=1, opponent_mode="fixed",
            multi_hand=True, hands_per_episode=10,
            reward_mode="chips", mc_sims=100)
        chips.train(20)

        res = evaluate_icm_tournament(
            [_rl_factory(icm.qnet), _rl_factory(chips.qnet), _myopic_factory],
            prize, seeds=range(4), names=["icm", "chips", "myopic"],
            mc_sims=100, max_hands=40)

        assert set(res) == {"icm", "chips", "myopic"}
        assert sum(v["mean_prize"] for v in res.values()) == pytest.approx(sum(prize))
        for v in res.values():
            assert v["mean_prize"] >= 0.0
