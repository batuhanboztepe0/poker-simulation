"""
stats.py
--------
Pure statistics over a `SessionResult` (Phase 1).

All functions are side-effect-free and read only the structured events and
stack snapshots on the SessionResult, so they are deterministic and unit
testable on synthetic sessions.

Definitions:
    VPIP  — fraction of dealt hands in which the player voluntarily put money
            in the pot pre-flop (call/raise/all-in; blind posts don't count).
    AF    — aggression frequency: aggressive / (aggressive + passive) actions,
            aggressive = raise/all-in, passive = call/check (folds excluded).
    SD%   — showdown win rate: showdowns won / showdowns reached.
    chip EV/hand — net stack change divided by hands played.

`bootstrap_ci` is the one general-purpose helper here (a percentile bootstrap CI
for the mean of any value list); it lives in this torch-free, dependency-light
module so analyses that must NOT pull the training path can import it without
dragging in `evaluation` -> `rl_agent` (-> torch). `evaluation` re-exports it for
backward compatibility.
"""

import random

from src.player import (
    ACTION_FOLD, ACTION_CHECK, ACTION_CALL, ACTION_RAISE, ACTION_ALL_IN,
)
from src.events import EVENT_ACTION, EVENT_SHOWDOWN

PRE_FLOP = "Pre-Flop"

VOLUNTARY  = {ACTION_CALL, ACTION_RAISE, ACTION_ALL_IN}
AGGRESSIVE = {ACTION_RAISE, ACTION_ALL_IN}
PASSIVE    = {ACTION_CALL, ACTION_CHECK}


def _preflop_actions(hand, player_id):
    """Return the player's pre-flop action events in a hand."""
    return [
        e for e in hand.events
        if e.event_type == EVENT_ACTION
        and e.player_id == player_id
        and e.street == PRE_FLOP
    ]


def compute_vpip(session, player_id):
    """
    Voluntarily-put-money-in-pot rate, in [0, 1].

    Denominator is the number of hands in which the player took any pre-flop
    action (i.e. was dealt in and reached a decision); numerator is the hands
    in which at least one of those actions was a call/raise/all-in.
    """
    dealt = 0
    voluntary = 0
    for hand in session.hands:
        pre = _preflop_actions(hand, player_id)
        if not pre:
            continue
        dealt += 1
        if any(e.action in VOLUNTARY for e in pre):
            voluntary += 1
    if dealt == 0:
        return 0.0
    return voluntary / dealt


def compute_aggression_frequency(session, player_id):
    """
    Aggression frequency across all streets, in [0, 1].

    aggressive / (aggressive + passive), where aggressive = raise/all-in and
    passive = call/check. Folds are excluded. Returns 0.0 if the player made
    no aggressive or passive actions.
    """
    aggressive = 0
    passive = 0
    for hand in session.hands:
        for e in hand.events:
            if e.event_type != EVENT_ACTION or e.player_id != player_id:
                continue
            if e.action in AGGRESSIVE:
                aggressive += 1
            elif e.action in PASSIVE:
                passive += 1
    total = aggressive + passive
    if total == 0:
        return 0.0
    return aggressive / total


def compute_showdown_win_rate(session, player_id):
    """
    Fraction of reached showdowns the player won, in [0, 1].

    A showdown is "reached" when the player is among the contenders in a
    showdown event; "won" when the hand's winnings credit them > 0 chips.
    """
    showdowns = 0
    wins = 0
    for hand in session.hands:
        sd = [e for e in hand.events if e.event_type == EVENT_SHOWDOWN]
        if not sd:
            continue
        payload = sd[0].payload or {}
        contenders = payload.get("contenders", [])
        if player_id not in contenders:
            continue
        showdowns += 1
        if hand.winnings.get(player_id, 0) > 0:
            # Credit a split pot as a fractional win (1/k), so a chopped
            # showdown is not double-counted as a full win for every player.
            tied = sum(1 for pid in contenders
                       if hand.winnings.get(pid, 0) > 0)
            wins += 1.0 / tied if tied > 0 else 1.0
    if showdowns == 0:
        return 0.0
    return wins / showdowns


def hands_played(session, player_id):
    """Number of hands the player took an action in or won chips from."""
    count = 0
    for hand in session.hands:
        took_action = any(
            e.event_type == EVENT_ACTION and e.player_id == player_id
            for e in hand.events
        )
        won = hand.winnings.get(player_id, 0) > 0
        if took_action or won:
            count += 1
    return count


def chip_ev_per_hand(session, player_id):
    """
    Average net chip result per hand played.

    Net = final - starting stack; denominator is hands_played (falls back to
    total session hands when the player has no recorded participation).
    """
    net = session.net_chips(player_id)
    played = hands_played(session, player_id)
    denom = played if played > 0 else session.n_hands
    if denom == 0:
        return 0.0
    return net / denom


def session_summary(session):
    """
    Per-player summary dict keyed by player_id.

    Each value: name, vpip, aggression_frequency, showdown_win_rate,
    chip_ev_per_hand, net_chips, hands_played.
    """
    summary = {}
    for snap in session.players:
        pid = snap.player_id
        summary[pid] = {
            "name": snap.name,
            "vpip": compute_vpip(session, pid),
            "aggression_frequency": compute_aggression_frequency(session, pid),
            "showdown_win_rate": compute_showdown_win_rate(session, pid),
            "chip_ev_per_hand": chip_ev_per_hand(session, pid),
            "net_chips": session.net_chips(pid),
            "hands_played": hands_played(session, pid),
        }
    return summary


def bootstrap_ci(values, n_resamples: int = 10000, ci: float = 0.95,
                 seed: int = 12345) -> dict:
    """
    Percentile bootstrap confidence interval for the MEAN of `values`.

    Resamples `values` with replacement `n_resamples` times (seeded, so the CI is
    reproducible) and returns the empirical CI of the resample means. A
    distribution-free complement to the paired t-test: if the CI excludes 0, the
    mean effect is significant at the (1-ci) level without assuming normality —
    the right uncertainty statement for heavy-tailed per-seed poker PnL.

    Returns {mean, lo, hi, ci, n}.
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "ci": ci, "n": 0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": float(mean), "lo": float(values[0]),
                "hi": float(values[0]), "ci": ci, "n": 1}
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = min(n_resamples - 1, int((1 + ci) / 2 * n_resamples))
    return {"mean": float(mean), "lo": float(means[lo_idx]),
            "hi": float(means[hi_idx]), "ci": ci, "n": n}
