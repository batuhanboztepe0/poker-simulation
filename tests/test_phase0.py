"""
test_phase0.py
--------------
Tests for Phase 0: Foundation (seeding, events/observers, opponent-observable
state, and the correctness fixes).

Definition of Done (ROADMAP §5):
    - same seed -> identical hand-by-hand outcomes,
    - event_log populated,
    - chip conservation over 50 seeded hands at 2/4/6/9 players.

Plus targeted regression tests for each correctness fix:
    - MC pre-flop lookup gated on a single opponent,
    - MC multi-way tie credits 1/k,
    - equity-aware _should_raise,
    - pre-flop action starts UTG (left of BB), BB acts last,
    - remove_player preserves dealer seat identity,
    - ev_breakdown guards a busted player.

Run with: python -m pytest tests/test_phase0.py -v
"""

import sys
import os
import random
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.card import Card
from src.monte_carlo import MonteCarloEngine
from src.player import (
    BotPlayer, Player, ACTION_RAISE, ACTION_CHECK, ACTION_CALL, ACTION_ALL_IN,
)
from src.game import GameEngine, ROUND_PRE_FLOP, ROUND_FLOP
from src.events import (
    HandEvent, EVENT_ACTION, EVENT_STREET, EVENT_HAND_END, EVENT_SHOWDOWN,
)


def make_cards(card_strs):
    return [Card(r, s) for r, s in card_strs]


def build_seeded_session(seed, n_players=4, use_mc=True, n_sims=120,
                         stack=1000, tight=0.3, aggression=0.5):
    """
    Build a fully seeded session: one rng shared across deck, MC, and bots.

    Returns (engine, bots).
    """
    rng = random.Random(seed)
    mc = MonteCarloEngine(n_simulations=n_sims, rng=rng) if use_mc else None
    bots = [
        BotPlayer(i, f"B{i}", stack, tight, aggression, mc, rng)
        for i in range(1, n_players + 1)
    ]
    engine = GameEngine(bots, small_blind=10, big_blind=20,
                        verbose=False, rng=rng)
    return engine, bots


def run_seeded_session(seed, n_players=4, n_hands=30, use_mc=True, n_sims=120):
    """Run a seeded session and return (per_hand_winnings, final_stacks)."""
    engine, bots = build_seeded_session(seed, n_players, use_mc, n_sims)
    per_hand = []
    for _ in range(n_hands):
        if sum(1 for b in bots if b.stack > 0) < 2:
            break
        per_hand.append(dict(engine.play_hand()))
    final_stacks = {b.player_id: b.stack for b in bots}
    return per_hand, final_stacks


# ===========================================================================
# Seeded reproducibility (DoD)
# ===========================================================================

class TestReproducibility:
    def test_same_seed_identical_with_mc(self):
        a = run_seeded_session(12345, n_players=4, n_hands=30, use_mc=True)
        b = run_seeded_session(12345, n_players=4, n_hands=30, use_mc=True)
        assert a == b

    def test_same_seed_identical_fast_mode(self):
        a = run_seeded_session(999, n_players=6, n_hands=40, use_mc=False)
        b = run_seeded_session(999, n_players=6, n_hands=40, use_mc=False)
        assert a == b

    def test_different_seeds_diverge(self):
        a = run_seeded_session(1, n_players=4, n_hands=30)
        b = run_seeded_session(2, n_players=4, n_hands=30)
        # Two independent seeds producing identical 30-hand histories is
        # astronomically unlikely.
        assert a != b

    def test_same_seed_identical_event_logs(self):
        e1, _ = build_seeded_session(42)
        e2, _ = build_seeded_session(42)
        for _ in range(10):
            e1.play_hand()
            e2.play_hand()
        log1 = [ev.to_dict() for ev in e1.event_log]
        log2 = [ev.to_dict() for ev in e2.event_log]
        assert log1 == log2


# ===========================================================================
# Chip conservation over 50 seeded hands at 2/4/6/9 players (DoD)
# ===========================================================================

class TestChipConservation:
    @pytest.mark.parametrize("n_players", [2, 4, 6, 9])
    def test_conservation_50_hands(self, n_players):
        engine, bots = build_seeded_session(7, n_players=n_players, n_sims=120)
        total_before = sum(b.stack for b in bots)
        for _ in range(50):
            if sum(1 for b in bots if b.stack > 0) < 2:
                break
            engine.play_hand()
        assert sum(b.stack for b in bots) == total_before


# ===========================================================================
# Event log + observers
# ===========================================================================

class TestEventLog:
    def test_event_log_populated(self):
        engine, _ = build_seeded_session(3, n_players=3, use_mc=False)
        engine.play_hand()
        assert len(engine.event_log) > 0
        assert all(isinstance(e, HandEvent) for e in engine.event_log)

    def test_event_types_present(self):
        engine, _ = build_seeded_session(3, n_players=3, use_mc=False)
        engine.play_hand()
        types = {e.event_type for e in engine.event_log}
        assert EVENT_STREET in types
        assert EVENT_ACTION in types
        assert EVENT_HAND_END in types

    def test_events_tagged_with_hand_id(self):
        engine, _ = build_seeded_session(3, n_players=3, use_mc=False)
        engine.play_hand()
        engine.play_hand()
        hand_ids = {e.hand_id for e in engine.event_log}
        assert hand_ids == {1, 2}

    def test_observer_receives_every_event(self):
        received = []
        engine, _ = build_seeded_session(5, n_players=3, use_mc=False)
        engine.add_observer(received.append)
        engine.play_hand()
        assert received == engine.event_log

    def test_observer_passed_in_constructor(self):
        received = []
        rng = random.Random(11)
        bots = [BotPlayer(i, f"B{i}", 1000, 0.3, 0.5, None, rng)
                for i in range(1, 3)]
        engine = GameEngine(bots, 10, 20, verbose=False, rng=rng,
                            observers=[received.append])
        engine.play_hand()
        assert len(received) > 0

    def test_hand_end_payload_has_winnings(self):
        engine, _ = build_seeded_session(8, n_players=3, use_mc=False)
        winnings = engine.play_hand()
        end_events = [e for e in engine.event_log
                      if e.event_type == EVENT_HAND_END]
        assert len(end_events) == 1
        assert end_events[0].payload["winnings"] == winnings

    def test_action_event_amount_recorded(self):
        engine, _ = build_seeded_session(8, n_players=4, use_mc=False)
        engine.play_hand()
        action_events = [e for e in engine.event_log
                         if e.event_type == EVENT_ACTION]
        assert len(action_events) > 0
        for e in action_events:
            assert e.amount is not None
            assert e.player_id is not None


# ===========================================================================
# Opponent-observable state + post-action callback
# ===========================================================================

class RecordingPlayer(BotPlayer):
    """Bot that records every observation it receives from the engine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations = []
        self.seen_states = []

    def decide(self, game_state):
        self.seen_states.append(game_state)
        return super().decide(game_state)

    def observe_action(self, observation):
        self.observations.append(observation)


class TestOpponentObservableState:
    def test_game_state_exposes_opponent_ids(self):
        engine, _ = build_seeded_session(3, n_players=4, use_mc=False)
        engine._setup_hand()
        engine._post_blinds()
        engine._deal_hole_cards()
        actor = engine.players[0]
        gs = engine._build_game_state(ROUND_FLOP, 0, actor=actor)
        assert "opponent_ids" in gs
        assert actor.player_id not in gs["opponent_ids"]
        assert gs["acting_player_id"] == actor.player_id

    def test_opponent_ids_exclude_folded(self):
        engine, _ = build_seeded_session(3, n_players=4, use_mc=False)
        engine._setup_hand()
        engine._post_blinds()
        engine._deal_hole_cards()
        engine.players[1].is_folded = True
        actor = engine.players[0]
        gs = engine._build_game_state(ROUND_FLOP, 0, actor=actor)
        assert engine.players[1].player_id not in gs["opponent_ids"]

    def test_observe_action_callback_fires(self):
        rng = random.Random(21)
        players = [
            RecordingPlayer(i, f"R{i}", 1000, 0.3, 0.5, None, rng)
            for i in range(1, 4)
        ]
        engine = GameEngine(players, 10, 20, verbose=False, rng=rng)
        engine.play_hand()
        # Each recording player should have observed opponents' actions, and
        # never its own action (engine excludes the actor).
        for p in players:
            for obs in p.observations:
                assert obs["actor_id"] != p.player_id
        # At least one player observed something.
        assert any(p.observations for p in players)

    def test_decide_receives_opponent_ids(self):
        rng = random.Random(22)
        players = [
            RecordingPlayer(i, f"R{i}", 1000, 0.3, 0.5, None, rng)
            for i in range(1, 4)
        ]
        engine = GameEngine(players, 10, 20, verbose=False, rng=rng)
        engine.play_hand()
        for p in players:
            for gs in p.seen_states:
                assert "opponent_ids" in gs
                assert p.player_id not in gs["opponent_ids"]


# ===========================================================================
# Correctness fix: MC pre-flop lookup gated on a single opponent
# ===========================================================================

class TestMCPreflopMultiwayGate:
    def test_aces_lookup_only_for_single_opponent(self):
        mc = MonteCarloEngine(n_simulations=3000, rng=random.Random(0))
        aa = make_cards([("A", "s"), ("A", "h")])
        eq_hu = mc.estimate_equity_unknown_opponents(aa, 1, [])
        eq_5way = mc.estimate_equity_unknown_opponents(aa, 5, [])
        # Heads-up uses the lookup table (~0.85). Five-way must simulate and
        # fall well below the heads-up value (true equity ~0.49).
        assert eq_hu >= 0.80
        assert eq_5way < 0.65
        assert eq_5way < eq_hu

    def test_multiway_monotonic_decline(self):
        mc = MonteCarloEngine(n_simulations=2000, rng=random.Random(1))
        kk = make_cards([("K", "s"), ("K", "h")])
        eq1 = mc.estimate_equity_unknown_opponents(kk, 1, [])
        eq3 = mc.estimate_equity_unknown_opponents(kk, 3, [])
        eq6 = mc.estimate_equity_unknown_opponents(kk, 6, [])
        assert eq1 > eq3 > eq6


# ===========================================================================
# Correctness fix: multi-way tie credits 1/k
# ===========================================================================

class TestMultiwayTie:
    def test_three_way_board_tie_credits_one_third(self):
        # A royal flush on the board: every player plays the board -> 3-way tie.
        board = make_cards([("A", "s"), ("K", "s"), ("Q", "s"),
                            ("J", "s"), ("T", "s")])
        hero = make_cards([("2", "h"), ("3", "h")])
        opp1 = make_cards([("4", "h"), ("5", "h")])
        opp2 = make_cards([("6", "h"), ("7", "h")])
        mc = MonteCarloEngine(n_simulations=200, rng=random.Random(0))
        eq = mc.estimate_equity(hero, [opp1, opp2], board)
        assert abs(eq - 1.0 / 3.0) < 1e-3

    def test_heads_up_board_tie_credits_one_half(self):
        board = make_cards([("A", "s"), ("K", "s"), ("Q", "s"),
                            ("J", "s"), ("T", "s")])
        hero = make_cards([("2", "h"), ("3", "h")])
        opp = make_cards([("4", "h"), ("5", "h")])
        mc = MonteCarloEngine(n_simulations=200, rng=random.Random(0))
        eq = mc.estimate_equity(hero, [opp], board)
        assert abs(eq - 0.5) < 1e-3


# ===========================================================================
# Correctness fix: equity-aware _should_raise
# ===========================================================================

class TestEquityAwareRaise:
    def _bot(self, equity, aggression):
        mc = MagicMock()
        mc.estimate_equity_unknown_opponents.return_value = equity
        bot = BotPlayer(1, "B", 500, tight_threshold=0.0,
                        aggression=aggression, mc_engine=mc)
        bot.receive_cards(make_cards([("2", "h"), ("3", "d")]))
        return bot

    def _free_state(self, pot=100, min_raise=20):
        return {
            "round_name": "Flop",
            "pot": pot,
            "call_amount": 0,
            "min_raise": min_raise,
            "current_bet": 0,
            "community_cards": make_cards([("A", "s"), ("K", "c"), ("Q", "h")]),
            "active_player_count": 2,
        }

    def test_low_equity_never_raises_despite_full_aggression(self):
        # 5% equity: every raise size is -EV, so even aggression=1.0 checks.
        bot = self._bot(equity=0.05, aggression=1.0)
        action, _ = bot.decide(self._free_state())
        assert action == "check"

    def test_high_equity_raises_at_full_aggression(self):
        bot = self._bot(equity=0.9, aggression=1.0)
        action, _ = bot.decide(self._free_state())
        assert action == "raise"

    def test_should_raise_method_is_equity_gated(self):
        bot = self._bot(equity=0.05, aggression=1.0)
        assert bot._should_raise(0.05, 100, 20, 500) is False
        assert bot._should_raise(0.95, 100, 20, 500) is True


# ===========================================================================
# Correctness fix: pre-flop action starts UTG, BB acts last
# ===========================================================================

class TestActionOrder:
    def _prepared_engine(self, n_players, dealer_index=0):
        engine, _ = build_seeded_session(3, n_players=n_players, use_mc=False)
        engine.dealer_index = dealer_index
        engine._setup_hand()
        engine._post_blinds()
        engine._deal_hole_cards()
        return engine

    def test_preflop_starts_utg_bb_last_4players(self):
        engine = self._prepared_engine(4, dealer_index=0)
        # dealer=0, SB=1, BB=2 -> UTG=3, order [3, 0, 1, 2], BB last.
        order = engine._build_action_order(ROUND_PRE_FLOP)
        ids = [p.player_id for p in order]
        bb_id = engine.players[engine._bb_index].player_id
        utg_id = engine.players[3].player_id
        assert ids[0] == utg_id
        assert ids[-1] == bb_id

    def test_preflop_heads_up_sb_first_bb_last(self):
        engine = self._prepared_engine(2, dealer_index=0)
        # Heads-up: dealer(0)=SB acts first pre-flop, BB(1) last.
        order = engine._build_action_order(ROUND_PRE_FLOP)
        ids = [p.player_id for p in order]
        assert ids[0] == engine.players[0].player_id
        assert ids[-1] == engine.players[1].player_id

    def test_postflop_starts_left_of_dealer(self):
        engine = self._prepared_engine(4, dealer_index=0)
        order = engine._build_action_order(ROUND_FLOP)
        ids = [p.player_id for p in order]
        # Post-flop: first to act is left of dealer (SB seat, index 1).
        assert ids[0] == engine.players[1].player_id
        assert ids[-1] == engine.players[0].player_id


# ===========================================================================
# Correctness fix: remove_player preserves dealer seat identity
# ===========================================================================

class TestRemovePlayerDealer:
    def _engine(self):
        rng = random.Random(0)
        players = [BotPlayer(i, f"B{i}", 1000, 0.3, 0.5, None, rng)
                   for i in range(1, 5)]  # ids 1,2,3,4
        return GameEngine(players, 10, 20, verbose=False, rng=rng)

    def test_remove_before_dealer_keeps_dealer_identity(self):
        engine = self._engine()
        engine.dealer_index = 2  # player id 3
        dealer = engine.players[2]
        engine.remove_player(1)  # remove seat before the dealer
        assert engine.players[engine.dealer_index] is dealer

    def test_remove_dealer_advances_button(self):
        engine = self._engine()
        engine.dealer_index = 2  # player id 3
        next_seat = engine.players[3]  # player id 4
        engine.remove_player(3)  # remove the dealer itself
        # Button now points at the seat that followed the dealer.
        assert engine.players[engine.dealer_index] is next_seat

    def test_remove_after_dealer_no_shift(self):
        engine = self._engine()
        engine.dealer_index = 1  # player id 2
        dealer = engine.players[1]
        engine.remove_player(4)  # seat after the dealer
        assert engine.players[engine.dealer_index] is dealer

    def test_remove_last_seat_clamps_dealer(self):
        engine = self._engine()
        engine.dealer_index = 3
        engine.remove_player(4)  # remove the seat the button is on (last)
        assert engine.dealer_index < len(engine.players)


# ===========================================================================
# Correctness fix: ev_breakdown guards a busted player
# ===========================================================================

class LimpPlayer(Player):
    """Always checks when free, calls any bet — deterministic pot tests."""
    def decide(self, game_state):
        call = game_state.get("call_amount", 0)
        if call == 0:
            return ACTION_CHECK, 0
        if call >= self.stack:
            return ACTION_ALL_IN, self.stack
        return ACTION_CALL, call


class TestBlindMechanics:
    """Regression: pre-flop blinds must not be wiped (SB discount, BB option)."""

    def test_three_limpers_pot_is_bb_times_players(self):
        rng = random.Random(0)
        players = [LimpPlayer(i, f"L{i}", 1000) for i in range(1, 4)]
        engine = GameEngine(players, small_blind=10, big_blind=20,
                            verbose=False, rng=rng)
        winnings = engine.play_hand()
        # Everyone in for exactly one big blind -> pot 60 (was 90 with the bug:
        # SB overpaying + BB forced to call its own blind).
        assert sum(winnings.values()) == 3 * 20

    def test_heads_up_sb_completes_bb_checks(self):
        rng = random.Random(0)
        players = [LimpPlayer(1, "A", 1000), LimpPlayer(2, "B", 1000)]
        engine = GameEngine(players, 10, 20, verbose=False, rng=rng)
        winnings = engine.play_hand()
        assert sum(winnings.values()) == 2 * 20


class TestEvBreakdownBustedGuard:
    def test_busted_player_no_raise_in_breakdown(self):
        mc = MagicMock()
        mc.estimate_equity_unknown_opponents.return_value = 0.6
        bot = BotPlayer(1, "Broke", 1, tight_threshold=0.0,
                        aggression=0.0, mc_engine=mc)
        bot.receive_cards(make_cards([("A", "s"), ("K", "h")]))
        # min_raise (40) far exceeds the 1-chip stack -> cannot raise.
        gs = {
            "round_name": "Flop",
            "pot": 200,
            "call_amount": 50,
            "min_raise": 40,
            "current_bet": 50,
            "community_cards": make_cards([("Q", "h"), ("J", "d"), ("2", "c")]),
            "active_player_count": 2,
        }
        breakdown = bot.ev_breakdown(gs)
        assert breakdown["raise"] is None
        assert breakdown["optimal_raise_size"] is None
        assert breakdown["best_action"] in ("fold", "call")


class TestIncompleteAllInMinRaise:
    """
    Regression: an incomplete all-in (a raise below the prior full increment)
    must NOT shrink last_raise_size, so the minimum raise still owed by later
    full raisers stays correct. (game.py ACTION_ALL_IN branch.)
    """

    def _engine(self):
        players = [BotPlayer(i, f"P{i}", 1000) for i in range(1, 4)]
        eng = GameEngine(players, small_blind=10, big_blind=20, verbose=False)
        eng.pot_manager.reset()
        # Simulate "A raised to 60" — a full 40-chip raise over the 20 BB.
        eng.current_bet = 60
        eng.last_raise_size = 40
        return eng

    def test_incomplete_all_in_preserves_last_raise_size(self):
        eng = self._engine()
        b = eng.players[1]
        b.current_bet, b.stack = 0, 80     # all-in for 80 -> a 20-chip raise
        eng._apply_action(b, ACTION_ALL_IN, 80, 60)
        assert b.current_bet == 80 and eng.current_bet == 80
        # 20 < 40 (incomplete) -> last_raise_size unchanged.
        assert eng.last_raise_size == 40
        # So C is owed a full raise to 120, not the buggy 100.
        gs = eng._build_game_state(ROUND_PRE_FLOP, 80, actor=eng.players[2])
        assert gs["min_raise"] == 120

    def test_full_all_in_updates_last_raise_size(self):
        eng = self._engine()
        b = eng.players[1]
        b.current_bet, b.stack = 0, 140    # all-in for 140 -> an 80-chip raise
        eng._apply_action(b, ACTION_ALL_IN, 140, 60)
        assert eng.current_bet == 140
        assert eng.last_raise_size == 80   # full all-in DOES update it
        gs = eng._build_game_state(ROUND_PRE_FLOP, 140, actor=eng.players[2])
        assert gs["min_raise"] == 220
