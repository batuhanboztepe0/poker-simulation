"""
events.py
---------
Structured event schema for the poker engine (Phase 0).

The GameEngine emits a `HandEvent` after every action, at the start of
each street, at showdown, and at hand end. This is the single source of
truth for the event schema — analytics (Phase 2) and any other consumer
import `HandEvent` from here rather than re-deriving the field set.

Design:
    - Observer pattern: consumers subscribe via GameEngine(observers=[...]);
      the engine never subclasses or hard-wires a consumer.
    - Events are plain dataclasses; `.to_dict()` yields a flat dict suitable
      for a pandas DataFrame row.
    - `community_cards` are stored as compact rank+suit codes (e.g. "As",
      "Td") so events serialize cleanly to Parquet/JSON.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

# event_type values -----------------------------------------------------
EVENT_ACTION   = "action"      # a player acted (fold/check/call/raise/all_in)
EVENT_STREET   = "street"      # a new betting round began
EVENT_SHOWDOWN = "showdown"    # hands revealed at showdown
EVENT_HAND_END = "hand_end"    # pot awarded, hand complete

STREET_SHOWDOWN = "Showdown"
STREET_HAND_END = "Hand-End"


@dataclass
class HandEvent:
    """
    A single structured event in a poker hand.

    Fields:
        hand_id (int): 1-based hand number within the session.
        street (str): Betting round name (or "Showdown" / "Hand-End").
        event_type (str): One of EVENT_ACTION/STREET/SHOWDOWN/HAND_END.
        player_id (int | None): Acting player (None for street/showdown/end).
        action (str | None): Action string for EVENT_ACTION.
        amount (int | None): Chips the player added this action (0 fold/check).
        pot (int): Pot total at the moment the event was emitted.
        community_cards (list[str]): Board as rank+suit codes.
        equity (float | None): Hero equity if known (engine leaves None).
        payload (dict | None): Event-specific extras (winnings, contenders).
    """
    hand_id: int
    street: str
    event_type: str
    player_id: Optional[int] = None
    action: Optional[str] = None
    amount: Optional[int] = None
    pot: int = 0
    community_cards: List[str] = field(default_factory=list)
    equity: Optional[float] = None
    payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a flat dict representation (one DataFrame row)."""
        return asdict(self)


def card_code(card) -> str:
    """Return a compact rank+suit code for a Card, e.g. Card('A','s') -> 'As'."""
    return f"{card.rank}{card.suit}"


def board_codes(cards) -> List[str]:
    """Return a list of compact card codes for a list of Cards."""
    return [card_code(c) for c in cards]
