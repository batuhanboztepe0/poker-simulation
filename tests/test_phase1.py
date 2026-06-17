"""
test_phase1.py
--------------
Unit tests for Phase 1: Core Engine.

Covers: Card, Deck, HandEvaluator, PotManager, Player, GameEngine.
Run with: python -m pytest tests/test_phase1.py -v
"""

import sys
import os
import pytest
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.card import Card, Deck, RANKS, SUITS, DECK_SIZE
from src.hand_evaluator import HandEvaluator, evaluate_hand, hand_label
from src.player import (
    HumanPlayer, BotPlayer,
    ACTION_FOLD, ACTION_CHECK, ACTION_CALL, ACTION_RAISE, ACTION_ALL_IN
)
from src.pot import PotManager, SidePot
from src.game import GameEngine


# ===========================================================================
# Card tests
# ===========================================================================

class TestCard:
    def test_valid_card_creation(self):
        card = Card("A", "s")
        assert card.rank == "A"
        assert card.suit == "s"

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError, match="Invalid rank"):
            Card("X", "s")

    def test_invalid_suit_raises(self):
        with pytest.raises(ValueError, match="Invalid suit"):
            Card("A", "z")

    def test_str_representation(self):
        card = Card("A", "s")
        assert "A" in str(card)
        assert "\u2660" in str(card)

    def test_card_equality(self):
        assert Card("K", "h") == Card("K", "h")
        assert Card("K", "h") != Card("K", "s")

    def test_card_hashable(self):
        cards = {Card("A", "s"), Card("A", "s"), Card("K", "h")}
        assert len(cards) == 2

    def test_treys_int_is_integer(self):
        card = Card("T", "d")
        assert isinstance(card.treys_int, int)


class TestDeck:
    def test_deck_starts_with_52_cards(self):
        deck = Deck()
        assert len(deck) == DECK_SIZE

    def test_deal_reduces_count(self):
        deck = Deck()
        deck.deal()
        assert len(deck) == DECK_SIZE - 1

    def test_deal_many(self):
        deck = Deck()
        cards = deck.deal_many(5)
        assert len(cards) == 5
        assert len(deck) == DECK_SIZE - 5

    def test_deal_many_invalid_count(self):
        deck = Deck()
        with pytest.raises(ValueError):
            deck.deal_many(0)

    def test_deal_many_exceeds_remaining(self):
        deck = Deck()
        with pytest.raises(ValueError, match="Cannot deal"):
            deck.deal_many(DECK_SIZE + 1)

    def test_shuffle_restores_full_deck(self):
        deck = Deck()
        deck.deal_many(10)
        deck.shuffle()
        assert len(deck) == DECK_SIZE

    def test_deck_is_shuffled(self):
        """Statistical test: two shuffles should produce different order."""
        random.seed(None)
        deck1 = Deck()
        deck2 = Deck()
        order1 = [c.treys_int for c in deck1.cards]
        order2 = [c.treys_int for c in deck2.cards]
        # Astronomically unlikely to be identical after separate shuffles
        assert order1 != order2

    def test_remove_card(self):
        deck = Deck()
        target = deck.cards[0]
        deck.remove([target])
        assert len(deck) == DECK_SIZE - 1
        assert target not in deck.cards

    def test_remove_nonexistent_card_raises(self):
        deck = Deck()
        card = deck.deal()  # remove from deck first
        with pytest.raises(ValueError, match="not found"):
            deck.remove([card])

    def test_no_duplicate_cards_in_full_deal(self):
        deck = Deck()
        cards = deck.deal_many(DECK_SIZE)
        treys_ints = [c.treys_int for c in cards]
        assert len(set(treys_ints)) == DECK_SIZE


# ===========================================================================
# HandEvaluator tests
# ===========================================================================

class TestHandEvaluator:
    def setup_method(self):
        self.evaluator = HandEvaluator()

    def _make_cards(self, card_strs):
        return [Card(r, s) for r, s in card_strs]

    def test_royal_flush_beats_straight_flush(self):
        # Royal flush: A K Q J T of spades
        royal = self._make_cards([("A","s"),("K","s")])
        board_royal = self._make_cards([("Q","s"),("J","s"),("T","s"),("2","h"),("3","d")])

        # Straight flush: 9 8 7 6 5 of hearts
        sf = self._make_cards([("9","h"),("8","h")])
        board_sf = self._make_cards([("7","h"),("6","h"),("5","h"),("2","d"),("3","c")])

        royal_score = self.evaluator.evaluate(royal, board_royal)
        sf_score = self.evaluator.evaluate(sf, board_sf)

        assert royal_score < sf_score  # lower = better

    def test_full_house_beats_flush(self):
        fh_hole = self._make_cards([("A","s"),("A","h")])
        fh_board = self._make_cards([("A","d"),("K","s"),("K","h"),("2","c"),("3","d")])

        flush_hole = self._make_cards([("T","s"),("8","s")])
        flush_board = self._make_cards([("6","s"),("4","s"),("2","s"),("K","d"),("Q","h")])

        fh_score = self.evaluator.evaluate(fh_hole, fh_board)
        fl_score = self.evaluator.evaluate(flush_hole, flush_board)

        assert fh_score < fl_score

    def test_hand_label_four_of_a_kind(self):
        # AA hole + A K K K board = Four of a Kind (four aces impossible, use four kings)
        hole = self._make_cards([("K","c"),("K","d")])
        board = self._make_cards([("K","s"),("K","h"),("A","d"),("2","c"),("3","d")])
        score = self.evaluator.evaluate(hole, board)
        assert self.evaluator.hand_label(score) == "Four of a Kind"

    def test_hand_label_full_house(self):
        hole = self._make_cards([("A","s"),("A","h")])
        board = self._make_cards([("K","s"),("K","h"),("K","d"),("2","c"),("3","d")])
        score = self.evaluator.evaluate(hole, board)
        assert self.evaluator.hand_label(score) == "Full House"

    def test_evaluate_requires_2_hole_cards(self):
        board = self._make_cards([("A","s"),("K","s"),("Q","s"),("J","s"),("T","s")])
        with pytest.raises(ValueError, match="2 hole cards"):
            self.evaluator.evaluate(
                self._make_cards([("A","h")]),
                board
            )

    def test_evaluate_requires_3_to_5_community(self):
        hole = self._make_cards([("A","h"),("K","h")])
        with pytest.raises(ValueError, match="3–5"):
            self.evaluator.evaluate(hole, self._make_cards([("Q","h"),("J","h")]))

    def test_best_hand_among_finds_winner(self):
        board = self._make_cards([("A","s"),("A","h"),("A","d"),("K","s"),("K","h")])
        players = {
            1: self._make_cards([("2","c"),("3","d")]),  # Full house (AAAKK)
            2: self._make_cards([("A","c"),("K","d")]),  # Four aces!
        }
        winners, scores = self.evaluator.best_hand_among(players, board)
        assert winners == [2]

    def test_best_hand_among_detects_tie(self):
        board = self._make_cards([("A","s"),("A","h"),("A","d"),("K","s"),("K","h")])
        players = {
            1: self._make_cards([("2","c"),("3","d")]),
            2: self._make_cards([("4","c"),("5","d")]),
        }
        winners, scores = self.evaluator.best_hand_among(players, board)
        assert set(winners) == {1, 2}

    def test_convenience_functions(self):
        hole = [Card("A","s"), Card("K","s")]
        board = [Card("Q","s"), Card("J","s"), Card("T","s"), Card("2","h"), Card("3","d")]
        score = evaluate_hand(hole, board)
        assert isinstance(score, int)
        label = hand_label(score)
        assert label == "Royal Flush"


# ===========================================================================
# PotManager tests
# ===========================================================================

class TestPotManager:
    def setup_method(self):
        self.pot = PotManager()

    def test_total_pot_zero_initially(self):
        assert self.pot.total_pot() == 0

    def test_add_contributions(self):
        self.pot.add_contribution(1, 100)
        self.pot.add_contribution(2, 200)
        assert self.pot.total_pot() == 300

    def test_negative_contribution_raises(self):
        with pytest.raises(ValueError):
            self.pot.add_contribution(1, -10)

    def test_simple_pot_no_all_in(self):
        self.pot.add_contribution(1, 100)
        self.pot.add_contribution(2, 100)
        pots = self.pot.calculate_pots()
        assert len(pots) == 1
        assert pots[0].amount == 200
        assert pots[0].eligible_player_ids == {1, 2}

    def test_side_pot_created_for_all_in(self):
        # Player 1 all-in for 50, Player 2 bets 100
        self.pot.add_contribution(1, 50)
        self.pot.add_contribution(2, 100)
        pots = self.pot.calculate_pots()
        assert len(pots) == 2
        # Main pot: 50+50=100, eligible: both
        main = pots[0]
        assert main.amount == 100
        assert 1 in main.eligible_player_ids
        assert 2 in main.eligible_player_ids
        # Side pot: remaining 50, eligible: player 2 only
        side = pots[1]
        assert side.amount == 50
        assert 1 not in side.eligible_player_ids
        assert 2 in side.eligible_player_ids

    def test_folded_player_excluded_from_pots(self):
        self.pot.add_contribution(1, 100)
        self.pot.add_contribution(2, 100)
        self.pot.mark_folded(1)
        pots = self.pot.calculate_pots()
        assert 1 not in pots[0].eligible_player_ids

    def test_distribute_simple(self):
        self.pot.add_contribution(1, 100)
        self.pot.add_contribution(2, 100)

        def resolver(ids):
            return [1]

        result = self.pot.distribute(resolver)
        assert result[1] == 200

    def test_distribute_split_pot(self):
        self.pot.add_contribution(1, 100)
        self.pot.add_contribution(2, 100)

        def resolver(ids):
            return [1, 2]

        result = self.pot.distribute(resolver)
        assert result[1] == 100
        assert result[2] == 100

    def test_reset_clears_state(self):
        self.pot.add_contribution(1, 500)
        self.pot.reset()
        assert self.pot.total_pot() == 0


# ===========================================================================
# Player tests
# ===========================================================================

class TestBotPlayer:
    def _make_game_state(self, call_amount=0, pot=100, min_raise=20):
        return {
            "round_name": "Flop",
            "pot": pot,
            "call_amount": call_amount,
            "min_raise": min_raise,
            "current_bet": call_amount,
            "community_cards": [],
            "active_player_count": 2,
        }

    def test_bot_can_check_when_no_bet(self):
        bot = BotPlayer(1, "Bot", 500, tight_threshold=0.0, aggression=0.0)
        action, amount = bot.decide(self._make_game_state(call_amount=0))
        assert action in (ACTION_CHECK, ACTION_RAISE, ACTION_ALL_IN)

    def test_tight_bot_folds_to_bet(self):
        """A tight bot with threshold=1.0 should always fold to a bet."""
        random.seed(42)
        bot = BotPlayer(1, "TightBot", 500, tight_threshold=1.0, aggression=0.0)
        action, _ = bot.decide(self._make_game_state(call_amount=50))
        assert action == ACTION_FOLD

    def test_post_blind_reduces_stack(self):
        bot = BotPlayer(1, "Bot", 500)
        posted = bot.post_blind(20)
        assert posted == 20
        assert bot.stack == 480

    def test_post_blind_all_in_when_insufficient(self):
        bot = BotPlayer(1, "Bot", 10)
        posted = bot.post_blind(20)
        assert posted == 10
        assert bot.stack == 0
        assert bot.is_all_in is True

    def test_reset_for_hand_clears_state(self):
        bot = BotPlayer(1, "Bot", 500)
        bot.post_blind(20)
        bot.is_folded = True
        bot.reset_for_hand()
        assert bot.hole_cards == []
        assert bot.current_bet == 0
        assert bot.is_folded is False

    def test_player_repr(self):
        bot = BotPlayer(99, "TestBot", 1000)
        assert "99" in repr(bot)
        assert "TestBot" in repr(bot)


# ===========================================================================
# GameEngine integration tests
# ===========================================================================

class TestGameEngine:
    def _make_engine(self, verbose=False):
        players = [
            BotPlayer(1, "Bot1", 1000, tight_threshold=0.3, aggression=0.5),
            BotPlayer(2, "Bot2", 1000, tight_threshold=0.3, aggression=0.5),
        ]
        return GameEngine(players, small_blind=10, big_blind=20, verbose=verbose)

    def test_engine_requires_minimum_2_players(self):
        with pytest.raises(ValueError, match="Player count must be"):
            GameEngine([BotPlayer(1, "Solo", 500)], 10, 20)

    def test_engine_requires_valid_blinds(self):
        players = [BotPlayer(1, "A", 500), BotPlayer(2, "B", 500)]
        with pytest.raises(ValueError, match="Invalid blinds"):
            GameEngine(players, small_blind=20, big_blind=10)

    def test_play_hand_returns_winnings_dict(self):
        engine = self._make_engine()
        winnings = engine.play_hand()
        assert isinstance(winnings, dict)

    def test_chips_are_conserved_across_hand(self):
        """Total chips should be identical before and after any hand."""
        engine = self._make_engine()
        total_before = sum(p.stack for p in engine.players)
        engine.play_hand()
        total_after = sum(p.stack for p in engine.players)
        assert total_before == total_after

    def test_chips_conserved_across_multiple_hands(self):
        engine = self._make_engine()
        total_before = sum(p.stack for p in engine.players)
        for _ in range(20):
            if sum(1 for p in engine.players if p.stack > 0) < 2:
                break
            engine.play_hand()
        total_after = sum(p.stack for p in engine.players)
        assert total_before == total_after

    def test_dealer_button_rotates(self):
        engine = self._make_engine()
        initial = engine.dealer_index
        engine.rotate_dealer()
        assert engine.dealer_index != initial or len(engine.players) == 1

    def test_hand_number_increments(self):
        engine = self._make_engine()
        engine.play_hand()
        assert engine.hand_number == 1
        engine.play_hand()
        assert engine.hand_number == 2

    def test_four_player_chips_conserved(self):
        players = [
            BotPlayer(i, f"Bot{i}", 500, tight_threshold=0.3, aggression=0.4)
            for i in range(1, 5)
        ]
        engine = GameEngine(players, small_blind=5, big_blind=10, verbose=False)
        total_before = sum(p.stack for p in players)
        for _ in range(10):
            if sum(1 for p in players if p.stack > 0) < 2:
                break
            engine.play_hand()
        total_after = sum(p.stack for p in players)
        assert total_before == total_after
