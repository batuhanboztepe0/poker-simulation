"""
hand_evaluator.py
-----------------
Wrapper around the treys library for 5- and 7-card hand evaluation.

Provides a clean, simulation-friendly interface. treys uses integer
encoding for O(1) lookup — this module isolates that dependency so
the rest of the engine works with Card objects only.
"""

from treys import Evaluator as TreysEvaluator
from treys import Card as TreysCard


HAND_RANK_LABELS = {
    0: "Royal Flush",      # treys uses 0 for royal flush rank class
    1: "Straight Flush",
    2: "Four of a Kind",
    3: "Full House",
    4: "Flush",
    5: "Straight",
    6: "Three of a Kind",
    7: "Two Pair",
    8: "One Pair",
    9: "High Card",
}

# treys rank score: lower = better. Royal flush = 1, worst high card ~7462
BEST_POSSIBLE_SCORE = 1
WORST_POSSIBLE_SCORE = 7462


class HandEvaluator:
    """
    Evaluates Texas Hold'em hands using the treys library.

    treys scoring convention: lower score = stronger hand.
    This class normalizes that to a consistent interface.
    """

    def __init__(self):
        """Initialize the treys evaluator."""
        self._evaluator = TreysEvaluator()

    def evaluate(self, hole_cards, community_cards):
        """
        Evaluate the best 5-card hand from hole + community cards.

        Args:
            hole_cards (list[Card]): Exactly 2 hole cards.
            community_cards (list[Card]): 3, 4, or 5 community cards.

        Returns:
            int: treys score (lower = better hand). Range: 1–7462.

        Raises:
            ValueError: If card counts are invalid.
        """
        if len(hole_cards) != 2:
            raise ValueError(
                f"Expected exactly 2 hole cards, got {len(hole_cards)}."
            )
        if len(community_cards) not in (3, 4, 5):
            raise ValueError(
                f"Expected 3–5 community cards, got {len(community_cards)}."
            )

        hole_ints = [c.treys_int for c in hole_cards]
        board_ints = [c.treys_int for c in community_cards]

        try:
            score = self._evaluator.evaluate(board_ints, hole_ints)
        except Exception as exc:
            raise RuntimeError(
                f"treys evaluation failed. Verify no duplicate cards. "
                f"Original error: {exc}"
            )
        return score

    def rank_class(self, score):
        """
        Return the hand rank class (1–10) for a treys score.

        Args:
            score (int): treys hand score.

        Returns:
            int: Rank class (1 = Royal Flush, 10 = High Card).
        """
        return self._evaluator.get_rank_class(score)

    def hand_label(self, score):
        """
        Return a human-readable hand label for a given score.

        Args:
            score (int): treys hand score.

        Returns:
            str: e.g. 'Full House', 'Two Pair'.
        """
        rank_cls = self.rank_class(score)
        return HAND_RANK_LABELS.get(rank_cls, "Unknown")

    def best_hand_among(self, players_hole_cards, community_cards):
        """
        Determine the winner(s) among multiple players at showdown.

        Handles ties (split pot scenario).

        Args:
            players_hole_cards (dict): {player_id: [Card, Card]}
            community_cards (list[Card]): Exactly 5 community cards.

        Returns:
            tuple: (winner_ids, scores_dict)
                - winner_ids (list): Player id(s) with the best hand.
                - scores_dict (dict): {player_id: score}
        """
        if len(community_cards) != 5:
            raise ValueError(
                f"Showdown requires exactly 5 community cards, "
                f"got {len(community_cards)}."
            )

        if not players_hole_cards:
            raise ValueError(
                "players_hole_cards must contain at least one player. "
                "All eligible players may have been filtered out."
            )

        scores = {}
        for player_id, hole in players_hole_cards.items():
            scores[player_id] = self.evaluate(hole, community_cards)

        best_score = min(scores.values())  # lower = better in treys
        winners = [pid for pid, s in scores.items() if s == best_score]
        return winners, scores


# Module-level singleton — no need to re-instantiate per hand
_evaluator_instance = HandEvaluator()


def evaluate_hand(hole_cards, community_cards):
    """
    Convenience function: evaluate a hand using the shared evaluator.

    Args:
        hole_cards (list[Card]): 2 hole cards.
        community_cards (list[Card]): 3–5 community cards.

    Returns:
        int: treys score (lower = better).
    """
    return _evaluator_instance.evaluate(hole_cards, community_cards)


def hand_label(score):
    """
    Convenience function: human-readable label for a treys score.

    Args:
        score (int): treys score.

    Returns:
        str: Hand label (e.g. 'Flush').
    """
    return _evaluator_instance.hand_label(score)
