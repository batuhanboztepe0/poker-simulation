"""
card.py
-------
Card and Deck abstractions for Texas Hold'em simulation.

Uses the treys library internally for hand evaluation compatibility.
All card representations are treys integer encoding under the hood,
but the public API works with human-readable rank/suit strings.
"""

import random
from treys import Card as TreysCard


RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["s", "h", "d", "c"]  # spades, hearts, diamonds, clubs

RANK_DISPLAY = {
    "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9",
    "T": "10", "J": "J", "Q": "Q", "K": "K", "A": "A",
}

SUIT_SYMBOLS = {
    "s": "\u2660",  # spades
    "h": "\u2665",  # hearts
    "d": "\u2666",  # diamonds
    "c": "\u2663",  # clubs
}

DECK_SIZE = 52


class Card:
    """
    Represents a single playing card.

    Wraps the treys integer encoding for fast hand evaluation
    while exposing a clean rank/suit interface.
    """

    def __init__(self, rank, suit):
        """
        Initialize a card.

        Args:
            rank (str): One of RANKS (e.g. 'A', 'K', 'T', '2').
            suit (str): One of SUITS ('s', 'h', 'd', 'c').

        Raises:
            ValueError: If rank or suit is invalid.
        """
        if rank not in RANKS:
            raise ValueError(
                f"Invalid rank '{rank}'. Valid ranks: {RANKS}"
            )
        if suit not in SUITS:
            raise ValueError(
                f"Invalid suit '{suit}'. Valid suits: {SUITS}"
            )
        self.rank = rank
        self.suit = suit
        self._treys_int = TreysCard.new(rank + suit)

    @property
    def treys_int(self):
        """Return the treys integer representation for hand evaluation."""
        return self._treys_int

    def __repr__(self):
        return f"Card('{self.rank}{self.suit}')"

    def __str__(self):
        return f"{RANK_DISPLAY[self.rank]}{SUIT_SYMBOLS[self.suit]}"

    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self._treys_int == other._treys_int

    def __hash__(self):
        return hash(self._treys_int)


class Deck:
    """
    A standard 52-card deck.

    Maintains dealt/remaining state. Supports dealing individual
    cards or burning before community card deals (standard poker
    procedure — skipped in simulation mode by default).
    """

    def __init__(self, rng=None):
        """
        Initialize and shuffle a full 52-card deck.

        Args:
            rng (random.Random | None): Random source for shuffling. When
                None, the module-level `random` is used (unseeded, matching
                pre-Phase-0 behavior). Inject a seeded `random.Random(seed)`
                for reproducible sessions.
        """
        self._cards = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        self._dealt = []
        self._rng = rng if rng is not None else random
        self._rng.shuffle(self._cards)

    def shuffle(self):
        """
        Reset the deck (return all dealt cards) and reshuffle.

        Should be called between hands. Uses the injected rng so a seeded
        session is fully reproducible.
        """
        self._cards.extend(self._dealt)
        self._dealt = []
        self._rng.shuffle(self._cards)

    def deal(self):
        """
        Deal the top card from the deck.

        Returns:
            Card: The top card.

        Raises:
            RuntimeError: If the deck is empty.
        """
        if not self._cards:
            raise RuntimeError(
                "Deck is empty. Call shuffle() before dealing a new hand."
            )
        card = self._cards.pop()
        self._dealt.append(card)
        return card

    def deal_many(self, count):
        """
        Deal multiple cards at once.

        Args:
            count (int): Number of cards to deal.

        Returns:
            list[Card]: List of dealt cards.

        Raises:
            ValueError: If count exceeds remaining cards.
            RuntimeError: If deck is empty.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if count > len(self._cards):
            raise ValueError(
                f"Cannot deal {count} cards — only {len(self._cards)} remaining."
            )
        return [self.deal() for _ in range(count)]

    def remove(self, cards):
        """
        Remove specific cards from the deck (used in Monte Carlo simulation
        to exclude known hole cards and community cards).

        Args:
            cards (list[Card]): Cards to remove.

        Raises:
            ValueError: If a card is not found in the remaining deck.
        """
        for card in cards:
            try:
                self._cards.remove(card)
                self._dealt.append(card)
            except ValueError:
                raise ValueError(
                    f"Card {card} not found in remaining deck. "
                    "It may have already been dealt."
                )

    @property
    def remaining(self):
        """Return the number of cards left in the deck."""
        return len(self._cards)

    @property
    def cards(self):
        """Return a copy of the remaining cards (read-only view)."""
        return list(self._cards)

    def __len__(self):
        return len(self._cards)

    def __repr__(self):
        return f"Deck(remaining={self.remaining})"
