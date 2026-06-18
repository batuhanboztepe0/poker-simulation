"""
tournament.py
-------------
Agent tournament harness (Track C).

Runs every unordered pair of agents over a set of seeds, then projects
results into the full directed win_matrix. This ensures:
    - win_matrix[A][B] + win_matrix[B][A] <= len(seeds) (ties possible).
    - Each session's chips are counted once per agent.
    - Determinism: same args -> same result on every call.

No streamlit imports; no side-effects on import.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.simulation import simulate_session
from src.game import DEFAULT_SMALL_BLIND, DEFAULT_BIG_BLIND

# Kuhn CFR note (the agent plays 3-card Kuhn, not hold'em; excluded from matrix)
_KUHN_CFR_NOTE = (
    "Kuhn CFR (different game - 3-card Kuhn poker, not hold'em): "
    "GTO game value = -1/18 ≈ -0.0556 for player 1; "
    "excluded from hold'em leaderboard."
)


def _pair_seed(name_a: str, name_b: str, seed_index: int) -> int:
    """
    Deterministic per-pair seed that is symmetric ((A,B) and (B,A) give
    the same seed) but differs across pairs and seed indices.

    Uses a polynomial hash over the lexicographically sorted pair names to
    avoid reliance on Python's PYTHONHASHSEED (which changes across runs).
    """
    lo, hi = (name_a, name_b) if name_a <= name_b else (name_b, name_a)
    s = lo + "\x00" + hi + "\x00" + str(seed_index)
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFF_FFFF
    return (h + seed_index * 6367) & 0x7FFF_FFFF


@dataclass
class TournamentResult:
    """
    Output of run_tournament.

    Attributes:
        win_matrix: win_matrix[A][B] = number of seeds where A ended with
            more chips than B (same underlying game, so
            win_matrix[A][B] + win_matrix[B][A] <= len(seeds)).
        leaderboard: list of dicts sorted by mean_net_chips descending;
            each entry has keys {name, mean_net_chips, n_matches}.
        pair_stats: {(nameA, nameB): {n_seeds, mean_chip_diff}} where
            chip_diff = net_chips(A) - net_chips(B), A < B lexicographically.
        notes: informational warnings (e.g., torch unavailable).
        kuhn_cfr_note: fixed string about KuhnCFR game value.
    """
    win_matrix: Dict[str, Dict[str, int]]
    leaderboard: List[dict]
    pair_stats: Dict[Tuple[str, str], dict]
    notes: List[str]
    kuhn_cfr_note: str


def run_tournament(
    roster: Dict,
    seeds: List[int],
    n_hands: int = 100,
    small_blind: int = DEFAULT_SMALL_BLIND,
    big_blind: int = DEFAULT_BIG_BLIND,
    fast_mode: bool = True,
    starting_stack: int = 1000,
) -> "TournamentResult":
    """
    Run a paired round-robin tournament between every unordered pair of agents.

    For each unordered pair {A, B} and each seed index, one game is played
    with A as player_id=1 and B as player_id=2. Outcomes are projected into
    both directions of the directed win_matrix.

    Args:
        roster: {name: factory(player_id, stack) -> Player}
            Factories must produce FRESH player instances on every call.
        seeds: list of integers; one session per (unordered pair, seed index).
        n_hands: max hands per session.
        small_blind: blind structure.
        big_blind: blind structure.
        fast_mode: strips MC engines for speed (passed to simulate_session).
        starting_stack: starting chip count for each player.

    Returns:
        TournamentResult
    """
    notes: List[str] = []

    # Resolve torch availability at call time (not import time).
    _torch_ok = _check_torch()
    if not _torch_ok:
        notes.append(
            "torch not available; RL agent (RLBotPlayer) excluded from tournament."
        )

    # Filter out agents that require torch when torch is absent.
    active_roster: Dict = {}
    for name, factory in roster.items():
        if _requires_torch(factory) and not _torch_ok:
            notes.append(f"Agent '{name}' skipped: requires torch.")
            continue
        active_roster[name] = factory

    names = list(active_roster.keys())

    # Initialise accumulators.
    win_matrix: Dict[str, Dict[str, int]] = {
        a: {b: 0 for b in names if b != a} for a in names
    }
    net_chips_total: Dict[str, float] = {name: 0.0 for name in names}
    n_matches: Dict[str, int] = {name: 0 for name in names}
    pair_stats: Dict[Tuple[str, str], dict] = {}

    # Iterate over every UNORDERED pair exactly once.
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a = names[i]
            name_b = names[j]

            chip_diffs: List[float] = []

            for seed_idx in range(len(seeds)):
                ps = _pair_seed(name_a, name_b, seed_idx)
                # Always seat A as player_id=1, B as player_id=2.
                player_a = active_roster[name_a](1, starting_stack)
                player_b = active_roster[name_b](2, starting_stack)

                total_before = player_a.stack + player_b.stack

                # Wire the deterministic rng into nested MC engines (e.g.
                # RolloutPolicy.mc_engine) that simulate_session does not reach.
                shared_rng = random.Random(ps)
                for p in (player_a, player_b):
                    _wire_rng(p, shared_rng)

                result = simulate_session(
                    [player_a, player_b],
                    n_hands=n_hands,
                    small_blind=small_blind,
                    big_blind=big_blind,
                    seed=ps,
                    fast_mode=fast_mode,
                )

                total_after = sum(result.final_stacks.values())
                assert total_after == total_before, (
                    f"Chip conservation violated for ({name_a} vs {name_b}), "
                    f"seed_idx={seed_idx}: before={total_before}, "
                    f"after={total_after}"
                )

                net_a = result.net_chips(1)
                net_b = result.net_chips(2)
                chip_diffs.append(net_a - net_b)

                # Update directed win counts.
                if net_a > net_b:
                    win_matrix[name_a][name_b] += 1
                elif net_b > net_a:
                    win_matrix[name_b][name_a] += 1

                # Each agent accumulates their own net chips (counted once).
                net_chips_total[name_a] += net_a
                net_chips_total[name_b] += net_b
                n_matches[name_a] += 1
                n_matches[name_b] += 1

            pair_stats[(name_a, name_b)] = {
                "n_seeds": len(seeds),
                "mean_chip_diff": (
                    sum(chip_diffs) / len(chip_diffs) if chip_diffs else 0.0
                ),
            }

    # Build leaderboard sorted by mean net chips descending.
    leaderboard = []
    for name in names:
        nm = n_matches[name]
        mean_net = (net_chips_total[name] / nm) if nm > 0 else 0.0
        leaderboard.append({
            "name": name,
            "mean_net_chips": mean_net,
            "n_matches": nm,
        })
    leaderboard.sort(key=lambda d: d["mean_net_chips"], reverse=True)

    return TournamentResult(
        win_matrix=win_matrix,
        leaderboard=leaderboard,
        pair_stats=pair_stats,
        notes=notes,
        kuhn_cfr_note=_KUHN_CFR_NOTE,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wire_rng(player, rng: random.Random) -> None:
    """
    Wire a shared rng into a player and any nested MC engine that
    simulate_session does not reach (e.g. RolloutPolicy.mc_engine).

    This is called before simulate_session so that simulate_session's own
    wiring takes over from there; the pre-call wiring ensures nested engines
    start from the same seed.
    """
    rollout_policy = getattr(player, "rollout_policy", None)
    if rollout_policy is not None:
        mc = getattr(rollout_policy, "mc_engine", None)
        if mc is not None and hasattr(mc, "_rng"):
            mc._rng = rng
        if hasattr(rollout_policy, "rng"):
            rollout_policy.rng = rng


def _check_torch() -> bool:
    """Return True if torch is importable at call time."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _requires_torch(factory) -> bool:
    """
    Return True if the factory carries a ``requires_torch = True`` attribute
    (used to mark RL agent factories that depend on PyTorch).
    """
    return getattr(factory, "requires_torch", False)
