"""
simulation.py
-------------
Headless batch self-play harness (Phase 1).

`simulate_session` seeds one rng, runs N hands of bot-vs-bot poker, harvests
the engine's structured event log per hand, and returns a `SessionResult`
that downstream analytics / stats consume.

Design notes:
    - The harness OWNS reproducibility: it builds a single `random.Random(seed)`
      and injects it into the engine, every bot, and every bot's MC engine, so
      one top-level seed replays the whole session bit-for-bit.
    - `fast_mode=True` strips the MC engine off each bot (equity becomes the
      cheap rng proxy) for large throughput runs.
    - Eliminated players (stack == 0) stay seated; the engine skips them and the
      loop stops once fewer than two players have chips. Total chips are
      therefore conserved across the whole run.
"""

import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from src.game import GameEngine, DEFAULT_SMALL_BLIND, DEFAULT_BIG_BLIND
from src.events import HandEvent


@dataclass
class HandRecord:
    """One played hand: its number, pot distribution, events, and end stacks."""
    hand_number: int
    winnings: Dict[int, int]
    events: List[HandEvent] = field(default_factory=list)
    stacks: Dict[int, int] = field(default_factory=dict)


@dataclass
class PlayerSnapshot:
    """End-of-session snapshot of a single player (identity + result)."""
    player_id: int
    name: str
    starting_stack: int
    final_stack: int
    tight_threshold: Optional[float] = None
    aggression: Optional[float] = None


@dataclass
class SessionResult:
    """
    The full result of a simulated session.

    Attributes:
        hands: ordered list of HandRecord.
        players: per-player snapshots (identity, starting/final stack, knobs).
        starting_stacks / final_stacks: {player_id: chips}.
        seed, small_blind, big_blind: session configuration.
    """
    hands: List[HandRecord]
    players: List[PlayerSnapshot]
    starting_stacks: Dict[int, int]
    final_stacks: Dict[int, int]
    seed: Optional[int]
    small_blind: int
    big_blind: int
    # Phase B: per-hand belief snapshots for bots that carry a belief model.
    # Each entry: {hand_number, player_id, posterior_mean, p_tilted}.
    opponent_belief_traces: List[Dict] = field(default_factory=list)
    # All-in EV control variate (only populated when run with track_allin_ev):
    # {player_id: sum over all-in showdowns of (ev_winnings - realised_winnings)}.
    allin_ev_adjust: Dict[int, float] = field(default_factory=dict)

    @property
    def n_hands(self) -> int:
        return len(self.hands)

    def all_events(self) -> List[HandEvent]:
        """Flatten every event across every hand."""
        return [e for h in self.hands for e in h.events]

    def winnings_sequence(self, player_id: int) -> List[int]:
        """Per-hand chips_won for a player (0 when they won nothing)."""
        return [h.winnings.get(player_id, 0) for h in self.hands]

    def net_chips(self, player_id: int) -> int:
        """Final minus starting stack for a player."""
        return (self.final_stacks.get(player_id, 0)
                - self.starting_stacks.get(player_id, 0))

    def net_chips_luck_adjusted(self, player_id: int) -> float:
        """Net chips with all-in runout luck removed: all-in pots are scored by
        equity*pot (their EV) instead of the realised board. Requires the session
        to have been run with track_allin_ev=True; equals net_chips otherwise."""
        return (self.net_chips(player_id)
                + self.allin_ev_adjust.get(player_id, 0.0))


def simulate_session(players, n_hands,
                     small_blind=DEFAULT_SMALL_BLIND,
                     big_blind=DEFAULT_BIG_BLIND,
                     seed=None, fast_mode=False, verbose=False,
                     track_allin_ev=False, allin_ev_sims=200):
    """
    Run a seeded self-play session and return a SessionResult.

    Args:
        players (list[Player]): Seated players (mutated in place; pass fresh
            instances to reproduce a seed).
        n_hands (int): Maximum hands to play (stops early once < 2 have chips).
        small_blind, big_blind (int): Blind structure.
        seed (int | None): Top-level seed; controls deck, MC, and bot rng.
        fast_mode (bool): Strip MC engines for speed (random equity proxy).
        verbose (bool): Engine terminal logging.

    Returns:
        SessionResult
    """
    rng = random.Random(seed)
    starting_stacks = {p.player_id: p.stack for p in players}

    # Wire the single shared rng into every component for reproducibility.
    for p in players:
        if fast_mode and hasattr(p, "mc_engine"):
            p.mc_engine = None
        if hasattr(p, "_rng"):
            p._rng = rng
        mc = getattr(p, "mc_engine", None)
        if mc is not None and hasattr(mc, "_rng"):
            mc._rng = rng

    engine = GameEngine(players, small_blind=small_blind, big_blind=big_blind,
                        verbose=verbose, rng=rng,
                        track_allin_ev=track_allin_ev,
                        allin_ev_sims=allin_ev_sims)

    records = []
    belief_traces = []
    for _ in range(n_hands):
        if sum(1 for p in players if p.stack > 0) < 2:
            break
        start_idx = len(engine.event_log)
        winnings = engine.play_hand()
        events = list(engine.event_log[start_idx:])
        records.append(HandRecord(
            hand_number=engine.hand_number,
            winnings=dict(winnings),
            events=events,
            stacks={p.player_id: p.stack for p in players},
        ))
        # Snapshot belief posteriors for any bot carrying a belief model.
        for p in players:
            belief = getattr(p, "belief_state", None)
            if belief is not None:
                belief_traces.append({
                    "hand_number": engine.hand_number,
                    "player_id": p.player_id,
                    "posterior_mean": belief.posterior_mean(),
                    "p_tilted": belief.p_tilted(),
                })

    final_stacks = {p.player_id: p.stack for p in players}
    snapshots = [
        PlayerSnapshot(
            player_id=p.player_id,
            name=p.name,
            starting_stack=starting_stacks[p.player_id],
            final_stack=p.stack,
            tight_threshold=getattr(p, "tight_threshold", None),
            aggression=getattr(p, "aggression", None),
        )
        for p in players
    ]

    return SessionResult(
        hands=records,
        players=snapshots,
        starting_stacks=starting_stacks,
        final_stacks=final_stacks,
        seed=seed,
        small_blind=small_blind,
        big_blind=big_blind,
        opponent_belief_traces=belief_traces,
        allin_ev_adjust=dict(engine.allin_ev_adjust),
    )
