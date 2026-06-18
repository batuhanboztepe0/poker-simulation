"""
analytics.py
------------
Event collection + analytical helpers over a session (Phase 2).

`HandResultCollector` is a GameEngine observer: register it and every emitted
HandEvent is captured into a flat, Parquet-safe pandas DataFrame. It can also
be built after the fact from a SessionResult / event list.

The pure helpers turn a SessionResult into the data behind each dashboard
panel:
    equity_curve            cumulative stack per player over hands
    win_rate_by_player      fraction of hands won
    hand_label_distribution made-hand breakdown at showdown
    ev_accuracy             predicted equity vs realised showdown outcome
"""

import json
from collections import Counter

import pandas as pd

from src.card import Card
from src.hand_evaluator import HandEvaluator
from src.events import (
    EVENT_ACTION, EVENT_SHOWDOWN, EVENT_HAND_END,
)

_EVALUATOR = HandEvaluator()


# ---------------------------------------------------------------------------
# Event collection
# ---------------------------------------------------------------------------

def _event_to_row(event):
    """Flatten a HandEvent into a Parquet-safe dict (lists/dicts serialized)."""
    d = event.to_dict()
    d["community_cards"] = " ".join(d.get("community_cards") or [])
    payload = d.get("payload")
    d["payload"] = json.dumps(payload) if payload is not None else None
    return d


def events_to_dataframe(events):
    """Build a flat DataFrame from an iterable of HandEvent."""
    rows = [_event_to_row(e) for e in events]
    columns = ["hand_id", "street", "event_type", "player_id", "action",
               "amount", "pot", "community_cards", "equity", "payload"]
    return pd.DataFrame(rows, columns=columns)


class HandResultCollector:
    """
    GameEngine observer that accumulates events into a DataFrame.

    Usage:
        collector = HandResultCollector()
        engine = GameEngine(players, observers=[collector])
        ... play hands ...
        df = collector.to_dataframe()
        collector.save_parquet("data/session.parquet")
    """

    def __init__(self):
        self.events = []

    def __call__(self, event):
        """Observer entry point: capture one event."""
        self.events.append(event)

    @classmethod
    def from_events(cls, events):
        """Build a collector from an existing list of HandEvent."""
        c = cls()
        c.events = list(events)
        return c

    @classmethod
    def from_session(cls, session):
        """Build a collector from a SessionResult's flattened events."""
        return cls.from_events(session.all_events())

    def to_dataframe(self):
        """Return a flat, Parquet-safe DataFrame of all captured events."""
        return events_to_dataframe(self.events)

    def save_parquet(self, path):
        """Persist the event DataFrame to Parquet. Returns the path."""
        self.to_dataframe().to_parquet(path, index=False)
        return path


def load_parquet(path):
    """Load a previously saved event DataFrame from Parquet."""
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Panel helpers (pure over SessionResult)
# ---------------------------------------------------------------------------

def equity_curve(session):
    """
    Cumulative stack per player over hands (the "equity curve" panel).

    Returns a tidy DataFrame with columns [hand_number, player_id, stack].
    Hand 0 is each player's starting stack.
    """
    rows = []
    for pid, stack in session.starting_stacks.items():
        rows.append({"hand_number": 0, "player_id": pid, "stack": stack})
    for hand in session.hands:
        stacks = hand.stacks or {}
        for pid, stack in stacks.items():
            rows.append({
                "hand_number": hand.hand_number,
                "player_id": pid,
                "stack": stack,
            })
    return pd.DataFrame(rows, columns=["hand_number", "player_id", "stack"])


def drawdown_curve(session, player_id):
    """
    Bankroll drawdown series for one player over hands.

    Returns a tidy DataFrame [hand_number, stack, peak, drawdown] where `peak`
    is the running high-water mark and `drawdown = peak - stack` (>= 0, the depth
    below the peak). Hand 0 is the starting stack. The poker analogue of a PnL
    max-drawdown curve.
    """
    eq = equity_curve(session)
    sub = eq[eq["player_id"] == player_id].sort_values("hand_number")
    rows = []
    peak = None
    for _, r in sub.iterrows():
        stack = r["stack"]
        peak = stack if peak is None else max(peak, stack)
        rows.append({
            "hand_number": int(r["hand_number"]),
            "stack": stack,
            "peak": peak,
            "drawdown": peak - stack,
        })
    return pd.DataFrame(rows, columns=["hand_number", "stack", "peak", "drawdown"])


def max_drawdown(session, player_id):
    """Largest peak-to-trough drop in a player's stack over the session."""
    dd = drawdown_curve(session, player_id)
    return float(dd["drawdown"].max()) if not dd.empty else 0.0


def win_rate_by_player(session):
    """
    Fraction of hands each player won (won > 0 chips).

    Returns {player_id: win_rate in [0, 1]}.
    """
    n = session.n_hands
    wins = Counter()
    for hand in session.hands:
        for pid, amount in hand.winnings.items():
            if amount > 0:
                wins[pid] += 1
    rates = {}
    for snap in session.players:
        rates[snap.player_id] = (wins[snap.player_id] / n) if n else 0.0
    return rates


def _cards_from_codes(codes):
    """Reconstruct Card objects from compact rank+suit codes ('As', 'Td')."""
    return [Card(code[0], code[1]) for code in codes]


def hand_label_distribution(session):
    """
    Distribution of made-hand labels shown at showdown across all contenders.

    Returns {label: count}. Requires a complete (5-card) board; showdown events
    that lack one (rare malformed edge) are skipped.
    """
    counts = Counter()
    for hand in session.hands:
        for e in hand.events:
            if e.event_type != EVENT_SHOWDOWN or not e.payload:
                continue
            board = e.community_cards or []
            if len(board) != 5:
                continue
            board_cards = _cards_from_codes(board)
            hole_map = e.payload.get("hole_cards", {})
            for _pid, hole_codes in hole_map.items():
                if len(hole_codes) != 2:
                    continue
                hole_cards = _cards_from_codes(hole_codes)
                score = _EVALUATOR.evaluate(hole_cards, board_cards)
                counts[_EVALUATOR.hand_label(score)] += 1
    return dict(counts)


def belief_trace_dataframe(session):
    """
    Tidy DataFrame of per-hand belief posteriors (Phase B).

    Columns [hand_number, player_id, posterior_mean, p_tilted]; empty when the
    session carried no belief-driven bots.
    """
    return pd.DataFrame(
        session.opponent_belief_traces,
        columns=["hand_number", "player_id", "posterior_mean", "p_tilted"],
    )


def ev_accuracy(session):
    """
    Predicted equity vs realised showdown outcome (the EV-accuracy panel).

    For every player that reached a showdown, take their last logged
    decision-time equity in that hand as the predicted win probability and
    realised = 1.0 if they won the hand else 0.0.

    Returns a DataFrame with columns
    [hand_number, player_id, predicted_equity, realised].
    """
    rows = []
    for hand in session.hands:
        showdown = next(
            (e for e in hand.events if e.event_type == EVENT_SHOWDOWN), None
        )
        if showdown is None or not showdown.payload:
            continue
        contenders = showdown.payload.get("contenders", [])
        # last decision-time equity per player this hand
        last_equity = {}
        for e in hand.events:
            if (e.event_type == EVENT_ACTION
                    and e.player_id is not None
                    and e.equity is not None):
                last_equity[e.player_id] = e.equity
        # Realised outcome as a pot-share, matching equity's semantics: a
        # k-way split credits 1/k to each tied winner (not 1.0 to all).
        winners = [pid for pid in contenders if hand.winnings.get(pid, 0) > 0]
        k = len(winners)
        for pid in contenders:
            if pid not in last_equity:
                continue
            realised = (1.0 / k) if (pid in winners and k > 0) else 0.0
            rows.append({
                "hand_number": hand.hand_number,
                "player_id": pid,
                "predicted_equity": last_equity[pid],
                "realised": realised,
            })
    return pd.DataFrame(
        rows,
        columns=["hand_number", "player_id", "predicted_equity", "realised"],
    )
