"""
test_phase_c.py
---------------
Tests for Phase C: bankroll-aware stochastic control (Kelly -> ICM -> rollout
-> RL).

DoD (ROADMAP §11):
    - Kelly sizing matches the analytic toy bet (+/-5%),
    - ICM matches a known 3-player example,
    - marginal chip value strictly decreasing,
    - KellyBotPlayer beats the myopic baseline over many seeds,
    - chip conservation with Kelly bots,
    - non-torch modules import without torch,
    - SelfPlayTrainer.train(...) completes well under the time budget.

Run with: python -m pytest tests/test_phase_c.py -v
"""

import sys
import os
import time
import random
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.card import Card
from src.player import BotPlayer
from src.game import GameEngine
from src.monte_carlo import MonteCarloEngine
from src.ev_calculator import kelly_fraction, kelly_raise_size
from src.icm import icm_equity, marginal_chip_value
from src.featurizer import featurize, FEATURE_DIM
from src.kelly_agent import KellyBotPlayer
from src.stochastic_control import RolloutPolicy, RolloutBotPlayer
import src.rl_agent as rl
from src.rl_agent import (
    ReplayBuffer, legal_action_indices, map_action_index, ACTION_NAMES,
    N_ACTIONS,
)


def make_cards(card_strs):
    return [Card(r, s) for r, s in card_strs]


# ===========================================================================
# Kelly sizing
# ===========================================================================

class TestKelly:
    def test_fraction_matches_analytic(self):
        # Even-money coin flip with 60% edge -> f* = 0.2.
        assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2, abs=1e-9)

    def test_raise_size_matches_toy_bet_within_5pct(self):
        size = kelly_raise_size(0.6, 1.0, stack=1000, min_raise=20,
                                max_raise=1000, kelly_scalar=1.0)
        assert abs(size - 200) / 200 <= 0.05

    def test_fractional_kelly_is_smaller(self):
        full = kelly_raise_size(0.6, 1.0, 1000, 20, 1000, kelly_scalar=1.0)
        half = kelly_raise_size(0.6, 1.0, 1000, 20, 1000, kelly_scalar=0.5)
        assert half < full

    def test_no_edge_no_bet(self):
        assert kelly_raise_size(0.3, 1.0, 1000, 20, 1000) == 0


# ===========================================================================
# ICM
# ===========================================================================

class TestICM:
    def test_known_three_player_example(self):
        eq = icm_equity([50, 30, 20], [70, 20, 10])
        assert eq[0] == pytest.approx(43.393, abs=0.01)
        assert eq[1] == pytest.approx(31.750, abs=0.01)
        assert eq[2] == pytest.approx(24.857, abs=0.01)

    def test_equity_sums_to_prize_pool(self):
        eq = icm_equity([40, 35, 25], [100, 60, 40])
        assert sum(eq) == pytest.approx(200.0, abs=1e-6)

    def test_chip_leader_has_most_equity(self):
        eq = icm_equity([60, 25, 15], [70, 20, 10])
        assert eq[0] > eq[1] > eq[2]

    def test_equal_stacks_equal_equity(self):
        eq = icm_equity([100, 100, 100], [70, 20, 10])
        assert eq[0] == pytest.approx(eq[1]) == pytest.approx(eq[2])

    def test_zero_stack_player_no_prize_leak(self):
        # Busted players occupy the bottom places and take those prizes, so the
        # pool is conserved (regression for the prize-leak bug).
        eq = icm_equity([1000, 0, 300], [700, 300, 100])
        assert sum(eq) == pytest.approx(1100.0, abs=1e-6)
        assert eq[1] == pytest.approx(100.0, abs=1e-6)  # busted -> 3rd prize
        eq2 = icm_equity([1000, 0, 0, 500], [500, 300, 200, 100])
        assert sum(eq2) == pytest.approx(1100.0, abs=1e-6)
        # two busted players split the bottom two prizes: (200 + 100) / 2 = 150
        assert eq2[1] == pytest.approx(150.0)
        assert eq2[2] == pytest.approx(150.0)

    def test_marginal_chip_value_strictly_decreasing(self):
        low = marginal_chip_value([20, 40, 40], [70, 20, 10], 0)
        mid = marginal_chip_value([50, 40, 40], [70, 20, 10], 0)
        high = marginal_chip_value([90, 40, 40], [70, 20, 10], 0)
        assert low > mid > high


# ===========================================================================
# Featurizer
# ===========================================================================

class TestFeaturizer:
    def _state(self):
        return {
            "round_name": "Flop",
            "pot": 200,
            "call_amount": 50,
            "min_raise": 40,
            "current_bet": 50,
            "community_cards": make_cards([("A", "s"), ("K", "c"), ("2", "h")]),
            "active_player_count": 3,
            "n_players": 4,
            "opponent_ids": [2, 3],
            "all_stacks": {1: 1000, 2: 800, 3: 1200, 4: 0},
        }

    def test_dimension(self):
        hero = BotPlayer(1, "H", 1000)
        feats = featurize(self._state(), hero, equity=0.6)
        assert len(feats) == FEATURE_DIM == 18
        assert all(isinstance(x, float) for x in feats)

    def test_deterministic(self):
        hero = BotPlayer(1, "H", 1000)
        s = self._state()
        assert featurize(s, hero, equity=0.6) == featurize(s, hero, equity=0.6)

    def test_equity_feature_first(self):
        hero = BotPlayer(1, "H", 1000)
        feats = featurize(self._state(), hero, equity=0.73)
        assert feats[0] == pytest.approx(0.73)


# ===========================================================================
# KellyBotPlayer
# ===========================================================================

class TestKellyBot:
    def test_more_conservative_all_in_than_myopic(self):
        # Marginal all-in: equity above pot odds but inside the ruin margin.
        gs = {
            "round_name": "Flop", "pot": 100, "call_amount": 100,
            "min_raise": 40, "current_bet": 100,
            "community_cards": make_cards([("2", "s"), ("7", "h"), ("9", "d")]),
            "active_player_count": 2, "opponent_ids": [2],
        }

        def bot(cls, **extra):
            mc = MagicMock()
            mc.estimate_equity_unknown_opponents.return_value = 0.52
            b = cls(1, "B", 100, tight_threshold=0.0, aggression=0.5,
                    mc_engine=mc, **extra)
            b.receive_cards(make_cards([("K", "d"), ("Q", "c")]))
            return b

        myopic = bot(BotPlayer)
        kelly = bot(KellyBotPlayer, kelly_scalar=0.5)
        assert myopic.decide(gs)[0] == "all_in"
        assert kelly.decide(gs)[0] == "fold"

    def test_clear_edge_kelly_commits(self):
        gs = {
            "round_name": "Flop", "pot": 100, "call_amount": 100,
            "min_raise": 40, "current_bet": 100,
            "community_cards": make_cards([("2", "s"), ("7", "h"), ("9", "d")]),
            "active_player_count": 2, "opponent_ids": [2],
        }
        mc = MagicMock()
        mc.estimate_equity_unknown_opponents.return_value = 0.80
        kelly = KellyBotPlayer(1, "K", 100, tight_threshold=0.0,
                               aggression=0.5, mc_engine=mc, kelly_scalar=0.5)
        kelly.receive_cards(make_cards([("A", "s"), ("A", "h")]))
        assert kelly.decide(gs)[0] == "all_in"

    def test_beats_myopic_over_many_seeds(self):
        def run_match(seed, n_hands=150):
            rng = random.Random(seed)
            mc = MonteCarloEngine(n_simulations=100, rng=rng)
            kelly = KellyBotPlayer(1, "Kelly", 1000, tight_threshold=0.2,
                                   aggression=0.5, mc_engine=mc, rng=rng,
                                   kelly_scalar=0.5)
            myopic = BotPlayer(2, "Myopic", 1000, tight_threshold=0.2,
                               aggression=0.5, mc_engine=mc, rng=rng)
            engine = GameEngine([kelly, myopic], 10, 20, verbose=False, rng=rng)
            for _ in range(n_hands):
                if sum(1 for p in (kelly, myopic) if p.stack > 0) < 2:
                    break
                engine.play_hand()
            return kelly.stack, myopic.stack

        wins = 0
        chip_diff = 0
        N = 40
        for s in range(N):
            k, m = run_match(s)
            chip_diff += (k - m)
            if k > m:
                wins += 1
        # Measured deterministically at 24/40 with +400 mean diff; assert a
        # safe margin (majority and positive chip EV).
        assert wins >= 22, f"Kelly won only {wins}/{N}"
        assert chip_diff > 0

    def test_chip_conservation(self):
        rng = random.Random(4)
        mc = MonteCarloEngine(n_simulations=100, rng=rng)
        players = [
            KellyBotPlayer(i, f"K{i}", 1000, 0.2, 0.5, mc, rng,
                           kelly_scalar=0.5)
            for i in range(1, 4)
        ]
        engine = GameEngine(players, 10, 20, verbose=False, rng=rng)
        total = sum(p.stack for p in players)
        for _ in range(30):
            if sum(1 for p in players if p.stack > 0) < 2:
                break
            engine.play_hand()
        assert sum(p.stack for p in players) == total


# ===========================================================================
# RolloutPolicy
# ===========================================================================

class TestRollout:
    def test_strong_hand_raises_free(self):
        mc = MonteCarloEngine(n_simulations=200, rng=random.Random(0))
        pol = RolloutPolicy(mc, rng=random.Random(1))
        gs = {
            "round_name": "Flop", "pot": 100, "call_amount": 0,
            "min_raise": 20, "current_bet": 0,
            "community_cards": make_cards([("A", "d"), ("7", "c"), ("2", "h")]),
            "active_player_count": 2, "opponent_ids": [2],
        }
        action, _ = pol.decide(make_cards([("A", "s"), ("A", "h")]), gs, 1000)
        assert action == "raise"

    def test_junk_folds_to_big_bet(self):
        mc = MonteCarloEngine(n_simulations=200, rng=random.Random(0))
        pol = RolloutPolicy(mc, rng=random.Random(1))
        gs = {
            "round_name": "Flop", "pot": 100, "call_amount": 400,
            "min_raise": 40, "current_bet": 400,
            "community_cards": make_cards([("A", "d"), ("K", "c"), ("Q", "h")]),
            "active_player_count": 2, "opponent_ids": [2],
        }
        action, _ = pol.decide(make_cards([("7", "d"), ("2", "c")]), gs, 1000)
        assert action == "fold"

    def test_chip_conservation(self):
        rng = random.Random(2)
        mc = MonteCarloEngine(n_simulations=100, rng=rng)
        players = []
        for i in range(1, 4):
            pol = RolloutPolicy(mc, rng=rng)
            players.append(RolloutBotPlayer(i, f"R{i}", 1000, mc_engine=mc,
                                            rng=rng, rollout_policy=pol))
        engine = GameEngine(players, 10, 20, verbose=False, rng=rng)
        total = sum(p.stack for p in players)
        for _ in range(25):
            if sum(1 for p in players if p.stack > 0) < 2:
                break
            engine.play_hand()
        assert sum(p.stack for p in players) == total


# ===========================================================================
# RL agent (torch-free parts always; torch parts guarded)
# ===========================================================================

class TestRLTorchFree:
    def test_replay_buffer(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.push([float(i)] * FEATURE_DIM, i % N_ACTIONS, float(i))
        assert len(buf) == 3  # capacity bound
        batch = buf.sample(2, random.Random(0))
        assert len(batch) == 2

    def test_action_mapping_legal(self):
        gs = {"call_amount": 0, "pot": 100, "min_raise": 20, "current_bet": 0}
        legal = legal_action_indices(gs, stack=1000)
        assert 0 in legal and 1 in legal
        for idx in legal:
            action, amount = map_action_index(idx, gs, 1000)
            assert action in ("fold", "check", "call", "raise", "all_in")
            assert 0 <= amount <= 1000

    def test_facing_bet_no_check(self):
        gs = {"call_amount": 50, "pot": 100, "min_raise": 40, "current_bet": 50}
        action, amount = map_action_index(1, gs, 1000)
        assert action == "call" and amount == 50

    def test_action_names_count(self):
        assert len(ACTION_NAMES) == N_ACTIONS == 5


@pytest.mark.skipif(not rl._HAVE_TORCH, reason="torch not installed")
class TestRLTorch:
    def test_qnetwork_output_dim(self):
        import torch
        net = rl.QNetwork()
        out = net(torch.zeros(FEATURE_DIM))
        assert out.shape[-1] == N_ACTIONS

    def test_self_play_trainer_completes_fast(self):
        trainer = rl.SelfPlayTrainer(n_players=2, seed=0)
        start = time.time()
        losses = trainer.train(100)
        elapsed = time.time() - start
        assert len(losses) == 100
        assert elapsed < 120.0, f"train(100) took {elapsed:.1f}s"

    def test_rl_bot_plays_and_conserves_chips(self):
        rng = random.Random(0)
        net = rl.QNetwork()
        players = [
            rl.RLBotPlayer(i, f"RL{i}", 1000, qnet=net, epsilon=0.1,
                           rng=random.Random(i))
            for i in range(1, 4)
        ]
        engine = GameEngine(players, 10, 20, verbose=False, rng=rng)
        total = sum(p.stack for p in players)
        for _ in range(20):
            if sum(1 for p in players if p.stack > 0) < 2:
                break
            engine.play_hand()
        assert sum(p.stack for p in players) == total

    def test_multi_hand_episodes_train_and_conserve_chips(self):
        """Bankroll (multi-hand) mode: stacks persist within an episode, the
        reward is a bounded log-utility delta, and chips are conserved every
        hand. Single-hand mode is unaffected (separate default path)."""
        import torch
        torch.manual_seed(0)
        tr = rl.SelfPlayTrainer(
            n_players=2, seed=1, opponent_mode="fixed", mc_sims=100,
            multi_hand=True, hands_per_episode=20, gamma=0.99)
        # Looser clip is auto-selected for log-utility rewards.
        assert tr.reward_clip == 3.0

        orig = tr.engine.play_hand
        n_players = len(tr.players)

        def checked():
            result = orig()
            assert (sum(p.stack for p in tr.players)
                    == n_players * tr.stack0), "chip conservation violated"
            return result

        tr.engine.play_hand = checked
        losses = tr.train(60, batch_size=32, refresh_every=5,
                          hands_per_refresh=20)
        assert len(losses) == 60
        # Every stored reward is a clipped log-utility delta.
        assert tr.buffer._buf, "no transitions collected"
        assert all(abs(t[2]) <= tr.reward_clip + 1e-9 for t in tr.buffer._buf)


@pytest.mark.skipif(not rl._HAVE_TORCH, reason="torch not installed")
class TestRLBeatsBaseline:
    """
    Phase C RL deliverable (docs/RL_HANDOFF.md): a self-trained agent that
    measurably beats the myopic EV baseline.

    Bounded so it runs in the suite: a short fixed-baseline training run, then
    a held-out evaluation. Poker variance is large and the bust-match win count
    has a wide CI, so rather than chase the exact headline number the test
    asserts a robust cluster: the random-init net is a *losing* baseline, and
    the trained policy is +EV, wins a clear majority of held-out matches, and
    improves over its own random init. The headline "X/50 wins, +Y mean chips"
    number is produced by `scripts/train_rl.py` and recorded in docs/PHASES.md.
    Fully seeded -> deterministic, not flaky (measured: random-init 2/40 vs
    trained 27/40 on these seeds).
    """

    def test_trained_agent_beats_myopic_and_improves_over_random_init(self):
        import copy
        import torch

        torch.manual_seed(0)
        trainer = rl.SelfPlayTrainer(
            n_players=2, seed=1, opponent_mode="fixed", mc_sims=100,
            epsilon_start=1.0, epsilon_end=0.05, gamma=0.97)
        random_init = copy.deepcopy(trainer.qnet)
        trainer.train(1200, batch_size=64, refresh_every=5,
                      hands_per_refresh=20)

        # Held-out seed range (disjoint from the 0..49 headline range and from
        # the trainer's own rng streams).
        eval_kwargs = dict(n_seeds=40, n_hands=150, mc_sims=100, seed_start=200)
        base = rl.evaluate_vs_baseline(random_init, **eval_kwargs)
        trained = rl.evaluate_vs_baseline(trainer.qnet, **eval_kwargs)

        # The random-init net must be a LOSING baseline, otherwise "trained > 0"
        # and "trained improves over base" would be vacuous.
        assert base["mean_chip_diff"] < 0, (
            f"random-init baseline not negative ({base['mean_chip_diff']}); "
            "the asserts below would be meaningless")
        # Trained policy is +EV, wins a clear majority of held-out matches, and
        # clearly improved over the untrained net it started from.
        assert trained["mean_chip_diff"] > 0, (
            f"trained mean chip diff {trained['mean_chip_diff']} not positive")
        assert trained["wins"] >= 22, (   # > 55% of 40 held-out matches
            f"trained won only {trained['wins']}/40 held-out matches")
        assert trained["mean_chip_diff"] > base["mean_chip_diff"]
