"""
test_phase3_belief.py
---------------------
Tests for Phase 3: the static Bayesian opponent model.

DoD:
    - prior init sane,
    - updates move the posterior in the right direction,
    - range_sample excludes known cards,
    - protocol / frozen-interface tests,
    - decisions differ between a tight-modeled and loose-modeled opponent.

Run with: python -m pytest tests/test_phase3_belief.py -v
"""

import sys
import os
import inspect
import random
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.card import Card
from src.monte_carlo import MonteCarloEngine
from src.player import BotPlayer, MIN_HANDS_FOR_BELIEF
from src.game import GameEngine
from src.opponent_model import BeliefState, combo_strength


def make_cards(card_strs):
    return [Card(r, s) for r, s in card_strs]


# ===========================================================================
# Frozen interface (protocol)
# ===========================================================================

class TestFrozenInterface:
    def test_required_methods_exist(self):
        b = BeliefState()
        for name in ("update", "update_from_showdown", "posterior_mean",
                     "p_tilted", "range_sample"):
            assert callable(getattr(b, name))

    def test_update_signature(self):
        params = list(inspect.signature(BeliefState.update).parameters)
        assert params == ["self", "action", "n_aggressive", "n_actions",
                          "delta_stack"]

    def test_range_sample_signature(self):
        params = list(inspect.signature(BeliefState.range_sample).parameters)
        assert params == ["self", "n_combos", "exclude_cards", "rng"]


# ===========================================================================
# Prior sanity
# ===========================================================================

class TestPrior:
    def test_prior_aggression_is_half(self):
        assert BeliefState().posterior_mean() == pytest.approx(0.5)

    def test_prior_looseness_is_half(self):
        assert BeliefState().looseness_mean() == pytest.approx(0.5)

    def test_static_model_never_tilted(self):
        assert BeliefState().p_tilted() == 0.0

    def test_custom_prior_mean(self):
        b = BeliefState(prior_aggression=0.8, prior_strength=10)
        assert b.posterior_mean() == pytest.approx(0.8)


# ===========================================================================
# Updates move the posterior in the right direction
# ===========================================================================

class TestUpdates:
    def test_aggressive_actions_raise_aggression_posterior(self):
        b = BeliefState()
        before = b.posterior_mean()
        for _ in range(20):
            b.update("raise", n_aggressive=1, n_actions=1)
        assert b.posterior_mean() > before
        assert b.posterior_mean() > 0.7

    def test_passive_actions_lower_aggression_posterior(self):
        b = BeliefState()
        for _ in range(20):
            b.update("call", n_aggressive=0, n_actions=1)
        assert b.posterior_mean() < 0.3

    def test_folds_lower_looseness(self):
        b = BeliefState()
        before = b.looseness_mean()
        for _ in range(20):
            b.update("fold", n_aggressive=0, n_actions=1)
        assert b.looseness_mean() < before

    def test_voluntary_actions_raise_looseness(self):
        b = BeliefState()
        for _ in range(20):
            b.update("call", n_aggressive=0, n_actions=1)
        assert b.looseness_mean() > 0.7

    def test_n_observations_counts_updates(self):
        b = BeliefState()
        for _ in range(5):
            b.update("call", 0, 1)
        assert b.n_observations == 5

    def test_showdown_weak_hand_raises_looseness(self):
        b = BeliefState()
        before = b.looseness_mean()
        b.update_from_showdown(make_cards([("7", "d"), ("2", "c")]))  # weak
        assert b.looseness_mean() > before
        assert len(b.shown_hands) == 1


# ===========================================================================
# range_sample
# ===========================================================================

class TestRangeSample:
    def test_returns_requested_count_of_pairs(self):
        b = BeliefState()
        rng = random.Random(0)
        hands = b.range_sample(50, exclude_cards=[], rng=rng)
        assert len(hands) == 50
        assert all(len(h) == 2 for h in hands)
        assert all(isinstance(c, Card) for h in hands for c in h)

    def test_excludes_known_cards(self):
        b = BeliefState()
        rng = random.Random(0)
        exclude = make_cards([("A", "s"), ("K", "s"), ("Q", "h"),
                             ("J", "d"), ("2", "c")])
        excl_ints = {c.treys_int for c in exclude}
        hands = b.range_sample(200, exclude_cards=exclude, rng=rng)
        for h in hands:
            assert h[0].treys_int not in excl_ints
            assert h[1].treys_int not in excl_ints
            assert h[0].treys_int != h[1].treys_int

    def test_tight_range_is_narrower_than_loose(self):
        rng = random.Random(0)
        tight = BeliefState()
        for _ in range(40):
            tight.update("fold", 0, 1)
        loose = BeliefState()
        for _ in range(40):
            loose.update("call", 0, 1)
        assert tight.range_fraction() < loose.range_fraction()

    def test_tight_range_holds_stronger_hands(self):
        rng = random.Random(1)
        tight = BeliefState()
        for _ in range(40):
            tight.update("fold", 0, 1)
        loose = BeliefState()
        for _ in range(40):
            loose.update("call", 0, 1)
        tight_hands = tight.range_sample(200, [], rng)
        loose_hands = loose.range_sample(200, [], rng)
        avg_tight = sum(combo_strength(h[0], h[1]) for h in tight_hands) / 200
        avg_loose = sum(combo_strength(h[0], h[1]) for h in loose_hands) / 200
        assert avg_tight > avg_loose


# ===========================================================================
# Belief-conditioned equity: tight vs loose
# ===========================================================================

class TestBeliefConditionedEquity:
    def _warm(self, action, n=40):
        b = BeliefState()
        for _ in range(n):
            b.update(action, 1 if action in ("raise", "all_in") else 0, 1)
        return b

    def test_equity_lower_vs_tight_range(self):
        rng = random.Random(0)
        mc = MonteCarloEngine(n_simulations=1500, rng=random.Random(2))
        tight = self._warm("fold")     # narrow strong range
        loose = self._warm("call")     # wide range
        hero = make_cards([("K", "d"), ("Q", "c")])
        eq_tight = mc.estimate_equity_vs_range(
            hero, [tight.range_sample(200, hero, rng)], [])
        eq_loose = mc.estimate_equity_vs_range(
            hero, [loose.range_sample(200, hero, rng)], [])
        assert eq_tight < eq_loose


# ===========================================================================
# BotPlayer integration
# ===========================================================================

class SpyMC(MonteCarloEngine):
    """MC engine that records belief-path invocations."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.vs_range_calls = 0

    def estimate_equity_vs_range(self, *a, **k):
        self.vs_range_calls += 1
        return super().estimate_equity_vs_range(*a, **k)


class TestBotIntegration:
    def _flop_state(self):
        return {
            "round_name": "Flop",
            "pot": 100,
            "call_amount": 40,
            "min_raise": 40,
            "current_bet": 40,
            "community_cards": make_cards([("2", "s"), ("7", "h"), ("9", "d")]),
            "active_player_count": 2,
            "opponent_ids": [2],
        }

    def test_cold_belief_uses_unknown_path(self):
        mc = SpyMC(n_simulations=200, rng=random.Random(3))
        belief = BeliefState()  # 0 observations -> below threshold
        bot = BotPlayer(1, "Cold", 1000, mc_engine=mc,
                        belief_state=belief, rng=random.Random(9))
        bot.receive_cards(make_cards([("A", "s"), ("K", "h")]))
        bot._estimate_equity(self._flop_state())
        assert mc.vs_range_calls == 0

    def test_warm_belief_uses_range_path(self):
        mc = SpyMC(n_simulations=200, rng=random.Random(3))
        belief = BeliefState()
        for _ in range(MIN_HANDS_FOR_BELIEF + 2):
            belief.update("call", 0, 1)
        bot = BotPlayer(1, "Warm", 1000, mc_engine=mc,
                        belief_state=belief, rng=random.Random(9))
        bot.receive_cards(make_cards([("A", "s"), ("K", "h")]))
        bot._estimate_equity(self._flop_state())
        assert mc.vs_range_calls == 1

    def test_decisions_differ_tight_vs_loose(self):
        mc = MonteCarloEngine(n_simulations=1200, rng=random.Random(4))
        tight = BeliefState()
        loose = BeliefState()
        for _ in range(40):
            tight.update("fold", 0, 1)
            loose.update("call", 0, 1)

        gs = {
            "round_name": "Flop",
            "pot": 100,
            "call_amount": 40,
            "min_raise": 40,
            "current_bet": 40,
            "community_cards": make_cards([("2", "s"), ("7", "h"), ("9", "d")]),
            "active_player_count": 2,
            "opponent_ids": [2],
        }
        hero = make_cards([("K", "d"), ("Q", "c")])

        bot_t = BotPlayer(1, "T", 1000, mc_engine=mc,
                          belief_state=tight, rng=random.Random(1))
        bot_t.receive_cards(list(hero))
        bot_l = BotPlayer(2, "L", 1000, mc_engine=mc,
                          belief_state=loose, rng=random.Random(1))
        bot_l.receive_cards(list(hero))

        eq_tight = bot_t._estimate_equity(gs)
        eq_loose = bot_l._estimate_equity(gs)
        # Facing a tight (strong) range, hero's equity is lower.
        assert eq_tight < eq_loose

    def test_belief_off_is_baseline(self):
        # A bot with no belief never touches the range path.
        mc = SpyMC(n_simulations=200, rng=random.Random(3))
        bot = BotPlayer(1, "Plain", 1000, mc_engine=mc, rng=random.Random(9))
        bot.receive_cards(make_cards([("A", "s"), ("K", "h")]))
        bot._estimate_equity(self._flop_state())
        assert mc.vs_range_calls == 0


# ===========================================================================
# Chip conservation with belief-driven bots
# ===========================================================================

class TestBeliefSessionIntegrity:
    def test_chip_conservation_with_belief_bots(self):
        rng = random.Random(5)
        mc = MonteCarloEngine(n_simulations=120, rng=rng)
        players = [
            BotPlayer(i, f"B{i}", 1000, 0.3, 0.5, mc, rng,
                      belief_state=BeliefState())
            for i in range(1, 4)
        ]
        engine = GameEngine(players, 10, 20, verbose=False, rng=rng)
        total = sum(p.stack for p in players)
        for _ in range(30):
            if sum(1 for p in players if p.stack > 0) < 2:
                break
            engine.play_hand()
        assert sum(p.stack for p in players) == total

    def test_beliefs_accumulate_observations(self):
        rng = random.Random(6)
        mc = MonteCarloEngine(n_simulations=120, rng=rng)
        beliefs = [BeliefState() for _ in range(3)]
        players = [
            BotPlayer(i, f"B{i}", 1000, 0.3, 0.5, mc, rng,
                      belief_state=beliefs[i - 1])
            for i in range(1, 4)
        ]
        engine = GameEngine(players, 10, 20, verbose=False, rng=rng)
        for _ in range(15):
            if sum(1 for p in players if p.stack > 0) < 2:
                break
            engine.play_hand()
        # Each bot observed opponents' actions over 15 hands.
        assert any(b.n_observations > 0 for b in beliefs)


# ===========================================================================
# A4: per-opponent belief dict (belief_factory) — unblocks 3+ player modeling
# ===========================================================================

class TestPerOpponentBeliefDict:
    """
    belief_factory makes BotPlayer keep a PER-OPPONENT belief dict:
    observe_action routes by actor_id, observe_hand_result by player_id, and
    equity conditions on each opponent's own range. Default (no factory) keeps
    the single-belief heads-up path byte-identical.
    """

    def test_observe_action_routes_by_actor_id(self):
        bot = BotPlayer(1, "Hero", 1000,
                        belief_factory=lambda oid: BeliefState())
        for _ in range(5):
            bot.observe_action({"actor_id": 2, "action": "raise",
                                "is_aggressive": True})
            bot.observe_action({"actor_id": 3, "action": "fold",
                                "is_aggressive": False})
        assert set(bot.beliefs.keys()) == {2, 3}
        # The aggressive actor's posterior is higher than the folder's.
        assert bot.beliefs[2].posterior_mean() > bot.beliefs[3].posterior_mean()
        assert bot.beliefs[2].n_observations == 5
        assert bot.beliefs[3].n_observations == 5

    def test_observe_hand_result_routes_by_player_id(self):
        from src.opponent_model import HMMBeliefState
        bot = BotPlayer(1, "Hero", 1000,
                        belief_factory=lambda oid: HMMBeliefState())
        # Opponent 2 lost big (tilts); opponent 3 won (stays calm).
        bot.observe_hand_result({"hand_id": 1, "player_id": 2,
                                 "delta_stack": -600})
        bot.observe_hand_result({"hand_id": 1, "player_id": 3,
                                 "delta_stack": +600})
        assert bot.beliefs[2].p_tilted() > bot.beliefs[3].p_tilted()

    def test_lazy_creation(self):
        created = []

        def factory(oid):
            created.append(oid)
            return BeliefState()

        bot = BotPlayer(1, "Hero", 1000, belief_factory=factory)
        assert bot.beliefs == {}
        bot.observe_action({"actor_id": 7, "action": "call",
                            "is_aggressive": False})
        assert created == [7]
        # Re-observing the same id reuses the belief (no second creation).
        bot.observe_action({"actor_id": 7, "action": "call",
                            "is_aggressive": False})
        assert created == [7]
        assert bot.beliefs[7].n_observations == 2

    def test_single_belief_path_unchanged_without_factory(self):
        b = BeliefState()
        bot = BotPlayer(1, "Hero", 1000, belief_state=b)
        bot.observe_action({"actor_id": 2, "action": "raise",
                            "is_aggressive": True})
        bot.observe_hand_result({"hand_id": 1, "player_id": 2,
                                 "delta_stack": -600})
        assert bot.beliefs == {}          # per-opponent dict untouched
        assert b.n_observations == 1      # single belief updated as before

    def test_per_opponent_equity_uses_one_range_per_opponent(self):
        from unittest.mock import MagicMock
        mc = MagicMock()
        mc.estimate_equity_vs_range.return_value = 0.42
        bot = BotPlayer(1, "Hero", 1000, mc_engine=mc,
                        belief_factory=lambda oid: BeliefState(),
                        rng=random.Random(0))
        bot.receive_cards([Card("A", "s"), Card("K", "h")])
        for _ in range(MIN_HANDS_FOR_BELIEF):       # warm both opponents
            bot.observe_action({"actor_id": 2, "action": "raise",
                                "is_aggressive": True})
            bot.observe_action({"actor_id": 3, "action": "fold",
                                "is_aggressive": False})
        gs = {"round_name": "Flop", "community_cards": [],
              "opponent_ids": [2, 3], "active_player_count": 3}
        eq = bot._estimate_equity(gs)
        assert eq == 0.42
        mc.estimate_equity_vs_range.assert_called_once()
        opponent_ranges = mc.estimate_equity_vs_range.call_args.args[1]
        assert len(opponent_ranges) == 2            # one range per opponent
        mc.estimate_equity_unknown_opponents.assert_not_called()

    def test_falls_back_to_unknown_when_any_opponent_cold(self):
        from unittest.mock import MagicMock
        mc = MagicMock()
        mc.estimate_equity_unknown_opponents.return_value = 0.33
        bot = BotPlayer(1, "Hero", 1000, mc_engine=mc,
                        belief_factory=lambda oid: BeliefState(),
                        rng=random.Random(0))
        bot.receive_cards([Card("A", "s"), Card("K", "h")])
        for _ in range(MIN_HANDS_FOR_BELIEF):       # warm opponent 2 only
            bot.observe_action({"actor_id": 2, "action": "raise",
                                "is_aggressive": True})
        gs = {"round_name": "Flop", "community_cards": [],
              "opponent_ids": [2, 3], "active_player_count": 3}
        eq = bot._estimate_equity(gs)
        assert eq == 0.33
        mc.estimate_equity_vs_range.assert_not_called()
        mc.estimate_equity_unknown_opponents.assert_called_once()

    def test_three_player_integration_builds_per_opponent_beliefs(self):
        rng = random.Random(7)
        mc = MonteCarloEngine(n_simulations=100, rng=rng)
        hero = BotPlayer(1, "Hero", 1000, tight_threshold=0.3, aggression=0.4,
                         mc_engine=mc, rng=rng,
                         belief_factory=lambda oid: BeliefState())
        others = [BotPlayer(i, f"B{i}", 1000, 0.3, 0.4, mc, rng)
                  for i in (2, 3)]
        players = [hero] + others
        engine = GameEngine(players, small_blind=10, big_blind=20,
                            verbose=False, rng=rng)
        total_before = sum(p.stack for p in players)
        for _ in range(15):
            if sum(1 for p in players if p.stack > 0) < 2:
                break
            engine.play_hand()
        assert total_before == sum(p.stack for p in players)   # chip conservation
        # Hero built a SEPARATE belief per opponent it observed; never itself.
        assert len(hero.beliefs) >= 1
        assert hero.player_id not in hero.beliefs
        assert set(hero.beliefs).issubset({2, 3})
