"""
test_phase_c_deepen.py
----------------------
Tests for the deepened Phase C bankroll-aware RL additions:

  1. TestFeaturizerExtended       - featurize_extended() and FEATURE_DIM_* constants.
  2. TestExtendedActionGrid       - 7-action EXTENDED_ACTION_NAMES / *_ext functions.
  3. TestICMRewardMode            - 3-player ICM reward training (torch-required).
  4. TestHorizonBeliefFeaturesTraining - extended feature training smoke (torch-required).

Run with: python -m pytest tests/test_phase_c_deepen.py -v
"""

import sys
import os
import random
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.player import BotPlayer
from src.card import Card
from src.featurizer import (
    featurize, featurize_extended, FEATURE_DIM,
    FEATURE_DIM_HORIZON, FEATURE_DIM_BELIEF, FEATURE_DIM_FULL,
)
from src.opponent_model import BeliefState
import src.rl_agent as rl
from src.rl_agent import (
    EXTENDED_ACTION_NAMES, N_EXTENDED_ACTIONS,
    legal_action_indices_ext, map_action_index_ext,
    ACTION_NAMES, N_ACTIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cards(card_strs):
    return [Card(r, s) for r, s in card_strs]


def _state():
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


def _hero():
    return BotPlayer(1, "Hero", 1000)


# ---------------------------------------------------------------------------
# 1. TestFeaturizerExtended
# ---------------------------------------------------------------------------

class TestFeaturizerExtended:

    def test_base_unchanged(self):
        """featurize() still returns exactly 18 floats; FEATURE_DIM==18."""
        hero = _hero()
        feats = featurize(_state(), hero, equity=0.6)
        assert len(feats) == 18
        assert FEATURE_DIM == 18
        assert all(isinstance(x, float) for x in feats)

    def test_extended_no_extras_is_base(self):
        """featurize_extended with no horizon/belief equals featurize()."""
        hero = _hero()
        s = _state()
        base = featurize(s, hero, equity=0.6)
        ext = featurize_extended(s, hero, equity=0.6)
        assert len(ext) == 18
        assert ext == base

    def test_horizon_appends_one_feature(self):
        """horizon=(10,50) appends a single 0.2 float at position 18."""
        hero = _hero()
        feats = featurize_extended(_state(), hero, equity=0.6, horizon=(10, 50))
        assert len(feats) == 19
        assert feats[18] == pytest.approx(0.2)

    def test_horizon_zero(self):
        """horizon=(0,50) appends 0.0."""
        hero = _hero()
        feats = featurize_extended(_state(), hero, equity=0.6, horizon=(0, 50))
        assert len(feats) == 19
        assert feats[18] == pytest.approx(0.0)

    def test_horizon_clamp_above_one(self):
        """horizon=(100,1) clamps to 1.0, not >1.0."""
        hero = _hero()
        feats = featurize_extended(_state(), hero, equity=0.6, horizon=(100, 1))
        assert len(feats) == 19
        assert feats[18] == pytest.approx(1.0)

    def test_belief_appends_two_features(self):
        """belief=BeliefState() appends posterior_mean and p_tilted."""
        hero = _hero()
        belief = BeliefState()
        feats = featurize_extended(_state(), hero, equity=0.6, belief=belief)
        assert len(feats) == 20
        assert feats[18] == pytest.approx(belief.posterior_mean())
        assert feats[19] == pytest.approx(belief.p_tilted())

    def test_full_extended_dim(self):
        """Both horizon and belief give 21 floats."""
        hero = _hero()
        belief = BeliefState()
        feats = featurize_extended(_state(), hero, equity=0.6,
                                   horizon=(5, 20), belief=belief)
        assert len(feats) == 21
        assert feats[18] == pytest.approx(5 / 20)

    def test_extended_deterministic(self):
        """Calling twice with same args produces identical vectors."""
        hero = _hero()
        belief = BeliefState()
        s = _state()
        a = featurize_extended(s, hero, equity=0.6, horizon=(3, 15), belief=belief)
        b = featurize_extended(s, hero, equity=0.6, horizon=(3, 15), belief=belief)
        assert a == b

    def test_feature_dim_constants(self):
        """FEATURE_DIM_* constants match spec values."""
        assert FEATURE_DIM_HORIZON == 19
        assert FEATURE_DIM_BELIEF == 20
        assert FEATURE_DIM_FULL == 21


# ---------------------------------------------------------------------------
# 2. TestExtendedActionGrid
# ---------------------------------------------------------------------------

class TestExtendedActionGrid:

    def test_extended_action_count(self):
        assert len(EXTENDED_ACTION_NAMES) == N_EXTENDED_ACTIONS == 7

    def test_default_action_count_unchanged(self):
        assert len(ACTION_NAMES) == N_ACTIONS == 5

    def _gs(self, pot=200, call=0, min_raise=20, current_bet=0):
        return {
            "pot": pot,
            "call_amount": call,
            "min_raise": min_raise,
            "current_bet": current_bet,
        }

    def test_ext_legal_indices_no_bet(self):
        """When call=0 and stack large, fold(0), passive(1), raises(2-5), all-in(6)."""
        gs = self._gs()
        legal = legal_action_indices_ext(gs, stack=1000)
        assert 0 in legal  # fold
        assert 1 in legal  # passive (check)
        assert 2 in legal  # raise_quarter
        assert 3 in legal  # raise_half
        assert 4 in legal  # raise_two_thirds
        assert 5 in legal  # raise_pot
        assert 6 in legal  # all-in

    def test_ext_no_raise_when_too_short(self):
        """Raise indices absent when stack < min_raise."""
        gs = self._gs(call=10, min_raise=50, current_bet=10)
        legal = legal_action_indices_ext(gs, stack=15)
        assert 0 in legal
        assert 1 in legal
        assert 6 in legal  # all-in still present if stack>0
        assert 2 not in legal
        assert 5 not in legal

    def test_ext_map_all_indices(self):
        """All 7 indices map to valid (action, amount) pairs."""
        VALID_ACTIONS = {"fold", "check", "call", "raise", "all_in"}
        gs = self._gs(pot=200, call=0, min_raise=20, current_bet=0)
        stack = 1000
        for idx in range(N_EXTENDED_ACTIONS):
            action, amount = map_action_index_ext(idx, gs, stack)
            assert action in VALID_ACTIONS, f"idx={idx} action={action!r}"
            assert 0 <= amount <= stack, f"idx={idx} amount={amount}"

    def test_ext_raise_quarter_smaller_than_half(self):
        """raise_quarter amount < raise_half amount when pot=200, stack=1000."""
        gs = self._gs(pot=200, call=0, min_raise=20, current_bet=0)
        stack = 1000
        _, quarter_amt = map_action_index_ext(2, gs, stack)
        _, half_amt = map_action_index_ext(3, gs, stack)
        assert quarter_amt < half_amt

    def test_ext_raise_two_thirds_between_half_and_pot(self):
        """two_thirds amount is strictly between half and pot amounts."""
        gs = self._gs(pot=200, call=0, min_raise=20, current_bet=0)
        stack = 1000
        _, half_amt = map_action_index_ext(3, gs, stack)
        _, two_thirds_amt = map_action_index_ext(4, gs, stack)
        _, pot_amt = map_action_index_ext(5, gs, stack)
        assert half_amt < two_thirds_amt < pot_amt


# ---------------------------------------------------------------------------
# 3. TestICMRewardMode  (torch-required)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not rl._HAVE_TORCH, reason="torch not installed")
class TestICMRewardMode:

    def _trainer(self, seed=7):
        return rl.SelfPlayTrainer(
            n_players=3,
            multi_hand=True,
            icm_prize_structure=[1500, 900, 600],
            seed=seed,
            opponent_mode="fixed",
        )

    def test_icm_requires_multi_hand(self):
        """Constructing with icm_prize_structure but multi_hand=False raises ValueError."""
        with pytest.raises(ValueError, match="multi_hand"):
            rl.SelfPlayTrainer(
                n_players=3,
                multi_hand=False,
                icm_prize_structure=[1500, 900, 600],
                seed=0,
            )

    def test_icm_prize_pool_equals_chip_pool(self):
        """sum(icm_prize_structure) == n_players * stack0."""
        trainer = self._trainer()
        total_chips = sum(p.stack for p in trainer.players)
        assert sum(trainer.icm_prize_structure) == pytest.approx(total_chips)

    def test_icm_train_smoke_runs(self):
        """train(50) returns a list of 50 loss floats without error."""
        trainer = self._trainer()
        losses = trainer.train(50, batch_size=16, hands_per_refresh=10)
        assert len(losses) == 50
        assert all(isinstance(l, float) for l in losses)

    def test_icm_rewards_are_clipped(self):
        """All buffered rewards are within the reward_clip bound."""
        trainer = self._trainer()
        trainer.train(30, batch_size=16, hands_per_refresh=10)
        clip = trainer.reward_clip
        for t in trainer.buffer._buf:
            assert abs(t[2]) <= clip + 1e-9, f"reward {t[2]} exceeds clip {clip}"

    def test_icm_reward_scale_matches_chip_scale(self):
        """
        The ICM-equity delta is normalized by sum(prizes)/n_players, which equals
        stack0 because the prize pool equals the chip pool -> the same O(1) scale
        the chips reward uses. (Without this the prize-unit delta saturates the
        clip into a sign-only signal; see test below.)
        """
        trainer = self._trainer()
        assert trainer.icm_reward_scale == pytest.approx(trainer.stack0)

    def test_icm_rewards_not_saturated_by_clip(self):
        """
        Regression for the reward-scaling bug: un-normalized ICM rewards (prize
        units, ~13x the clip) clobbered ~all rewards to +/-clip, training a
        sign-only policy. With normalization the graded signal survives, so only
        a small fraction of nonzero rewards sit at the clip boundary.
        """
        trainer = self._trainer()
        trainer.train(60, batch_size=16, hands_per_refresh=10)
        clip = trainer.reward_clip
        nonzero = [abs(t[2]) for t in trainer.buffer._buf if t[2] != 0.0]
        assert nonzero, "expected some nonzero ICM rewards"
        saturated = sum(1 for r in nonzero if r >= clip - 1e-9)
        # The bug produced >90% saturation; the fix keeps it well below half.
        assert saturated / len(nonzero) < 0.5, (
            f"{saturated}/{len(nonzero)} ICM rewards saturate the clip "
            f"({clip}) -> signal is being clobbered")

    def test_icm_trainer_chip_conservation(self):
        """
        Total chips across all players must be constant after every hand in
        ICM mode.  We wrap _collect_episode to assert conservation.
        """
        trainer = self._trainer(seed=42)
        total_chips = 3 * trainer.stack0
        # Run a collection and verify after each episode that stacks sum correctly.
        original_episode = trainer._collect_episode

        conservation_violated = []

        def patched_episode():
            original_episode()
            s = sum(p.stack for p in trainer.players)
            if s != total_chips:
                conservation_violated.append(s)

        trainer._collect_episode = patched_episode
        trainer._collect(30)
        assert not conservation_violated, (
            f"Chip conservation violated: totals={conservation_violated}")


# ---------------------------------------------------------------------------
# 4. TestHorizonBeliefFeaturesTraining  (torch-required)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not rl._HAVE_TORCH, reason="torch not installed")
class TestHorizonBeliefFeaturesTraining:

    def test_horizon_feature_trainer_smoke(self):
        """SelfPlayTrainer with extended_features=True, feature_mode='horizon' runs."""
        trainer = rl.SelfPlayTrainer(
            n_players=2, multi_hand=True, extended_features=True,
            feature_mode="horizon", seed=5,
        )
        losses = trainer.train(30, batch_size=16)
        assert len(losses) == 30

    def test_extended_qnetwork_input_dim(self):
        """QNetwork first layer input == FEATURE_DIM_HORIZON == 19."""
        import torch.nn as nn
        trainer = rl.SelfPlayTrainer(
            n_players=2, multi_hand=True, extended_features=True,
            feature_mode="horizon", seed=5,
        )
        first_layer = trainer.qnet.net[0]
        assert isinstance(first_layer, nn.Linear)
        assert first_layer.in_features == FEATURE_DIM_HORIZON == 19

    def test_extended_buffer_feature_length(self):
        """After collecting 5 hands, buffer transitions have 19-dim feature vectors."""
        trainer = rl.SelfPlayTrainer(
            n_players=2, multi_hand=True, extended_features=True,
            feature_mode="horizon", seed=5,
        )
        trainer._collect(5)
        assert len(trainer.buffer) > 0
        feat = trainer.buffer._buf[0][0]
        assert len(feat) == FEATURE_DIM_HORIZON

    def test_belief_feature_trainer_smoke(self):
        """feature_mode='belief' trains for 20 steps; buffer has 20-dim features."""
        trainer = rl.SelfPlayTrainer(
            n_players=2, extended_features=True, feature_mode="belief", seed=6,
        )
        # Inject a constant BeliefState into all learners so decide() has
        # something to call posterior_mean() / p_tilted() on.
        belief = BeliefState()
        for b in trainer.learners:
            b._belief = belief
        trainer.train(20, batch_size=16)
        # Check buffer has 20-dim vectors (18 base + 2 belief).
        assert len(trainer.buffer) > 0
        feat = trainer.buffer._buf[0][0]
        assert len(feat) == FEATURE_DIM_BELIEF == 20

    def test_seed_determinism_extended(self):
        """Two trainers with same seed produce identical first-10-loss sequences."""
        def run(seed):
            t = rl.SelfPlayTrainer(
                n_players=2, multi_hand=True, extended_features=True,
                feature_mode="horizon", seed=seed,
            )
            import torch
            torch.manual_seed(seed)
            losses = t.train(10, batch_size=16)
            return losses

        import torch
        torch.manual_seed(42)
        l1 = run(42)
        torch.manual_seed(42)
        l2 = run(42)
        assert l1 == l2, "Training is not deterministic with same seed"
