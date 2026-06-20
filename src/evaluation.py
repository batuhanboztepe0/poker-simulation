"""
evaluation.py
-------------
Monte-Carlo PnL evaluation between agents (the "measure PnL with MC" layer).

Each seed is one self-contained heads-up bankroll match driven by a shared
`random.Random(seed)` (deck + every bot's MC sampling), so a seed is a paired
observation and the per-seed chip difference is a realised PnL draw. This is the
torch-free generalisation of `rl_agent.evaluate_vs_baseline`: it works for ANY
agent factories (myopic / Kelly / rollout, and RL when a qnet-backed factory is
supplied by the caller), and exposes the *per-seed* diffs so the dashboard can
chart the PnL distribution and its paired t-test, not just the mean.

Quant parallel: per-seed diff = a strategy's realised PnL per scenario; the
paired t-test t-stat is its Sharpe-like significance over the scenario set.

Pure of torch and Streamlit; reuses the deterministic seeding helpers from
`tournament` and the `paired_t_test` from `rl_agent` (both torch-free).
"""

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from src.simulation import simulate_session
from src.game import DEFAULT_SMALL_BLIND, DEFAULT_BIG_BLIND
from src.tournament import _wire_rng, _pair_seed
from src.rl_agent import paired_t_test
from src.player import BotPlayer
from src.monte_carlo import MonteCarloEngine

# A factory builds a FRESH player each call: factory(player_id, stack) -> Player.
AgentFactory = Callable[[int, int], object]


@dataclass
class MatchupResult:
    """
    One A-vs-B evaluation over a set of seeds (A seated as player_id=1).

    Attributes:
        name_a, name_b: agent display names.
        seeds: the seeds actually played.
        diffs: per-seed PnL diff = net_chips(A) - net_chips(B).
        net_a, net_b: per-seed net chips for each agent.
        wins_a, wins_b, ties: per-seed outcome counts.
        mean_diff: mean of `diffs`.
        t_test: paired_t_test(diffs) -> {mean, n, t, p_value}.
    """
    name_a: str
    name_b: str
    seeds: List[int]
    diffs: List[int]
    net_a: List[int]
    net_b: List[int]
    wins_a: int
    wins_b: int
    ties: int
    mean_diff: float
    t_test: dict


@dataclass
class RosterResult:
    """
    Round-robin PnL evaluation over a roster of named agent factories.

    Attributes:
        leaderboard: list of {name, mean_net_chips, n_matches} sorted desc.
        win_matrix: win_matrix[A][B] = seeds A finished ahead of B.
        per_agent_nets: {name: [net chips for every (pair, seed) it played]}.
        pair_results: {(A, B): MatchupResult} for A < B lexicographically.
        n_seeds: seeds per pair.
    """
    leaderboard: List[dict]
    win_matrix: Dict[str, Dict[str, int]]
    per_agent_nets: Dict[str, List[int]]
    pair_results: Dict[Tuple[str, str], MatchupResult]
    n_seeds: int = 0


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


def _play_match(factory_seat1, factory_seat2, name_a, name_b, seed, n_hands,
                small_blind, big_blind, starting_stack, fast_mode,
                luck_adjusted=False):
    """Play one seeded heads-up match; return (net_seat1, net_seat2). Card luck is
    fixed by the deck seed (independent of actions), so swapping the two factories
    on the same seed gives a true duplicate/mirror of the deal. With
    ``luck_adjusted`` the nets are all-in-EV adjusted (realised runouts replaced
    by their equity-weighted EV)."""
    p1 = factory_seat1(1, starting_stack)
    p2 = factory_seat2(2, starting_stack)
    total_before = p1.stack + p2.stack
    shared_rng = random.Random(seed)
    for p in (p1, p2):
        _wire_rng(p, shared_rng)
    result = simulate_session(
        [p1, p2], n_hands=n_hands, small_blind=small_blind,
        big_blind=big_blind, seed=seed, fast_mode=fast_mode,
        track_allin_ev=luck_adjusted)
    total_after = sum(result.final_stacks.values())
    assert total_after == total_before, (
        f"Chip conservation violated ({name_a} vs {name_b}, seed={seed}): "
        f"{total_before} -> {total_after}")
    if luck_adjusted:
        return (result.net_chips_luck_adjusted(1),
                result.net_chips_luck_adjusted(2))
    return result.net_chips(1), result.net_chips(2)


def evaluate_matchup(factory_a: AgentFactory, factory_b: AgentFactory,
                     name_a: str, name_b: str, seeds: List[int],
                     n_hands: int = 200, small_blind: int = DEFAULT_SMALL_BLIND,
                     big_blind: int = DEFAULT_BIG_BLIND,
                     starting_stack: int = 1000,
                     fast_mode: bool = False,
                     mirror: bool = False,
                     luck_adjusted: bool = False) -> MatchupResult:
    """
    Play one seeded heads-up bankroll match per seed and collect the PnL diffs.

    For each seed, A is seated as player_id=1 and B as player_id=2 over the same
    deck (paired). Chip conservation is asserted per match. Returns the per-seed
    diffs plus their paired t-test against 0.

    Variance-reduction options (both opt-in, default off → byte-identical;
    composable; references.md §2):
    - ``mirror=True`` — DUPLICATE/mirror match: replay the same deck with the
      agents in swapped seats and average A's result over both orientations,
      cancelling the deck-luck that favours one seat.
    - ``luck_adjusted=True`` — all-in EV control variate: score all-in pots by
      their equity-weighted EV instead of the realised runout, removing the
      board-runout chance variance (the AIVAT-family chance-node adjustment).
    """
    diffs, net_a, net_b = [], [], []
    wins_a = wins_b = ties = 0
    for seed in seeds:
        na1, nb1 = _play_match(factory_a, factory_b, name_a, name_b, seed,
                               n_hands, small_blind, big_blind, starting_stack,
                               fast_mode, luck_adjusted)
        if mirror:
            # Replay the same deck with seats swapped; A now sits in seat 2.
            nb2, na2 = _play_match(factory_b, factory_a, name_b, name_a, seed,
                                   n_hands, small_blind, big_blind,
                                   starting_stack, fast_mode, luck_adjusted)
            na, nb = (na1 + na2) / 2, (nb1 + nb2) / 2
        else:
            na, nb = na1, nb1
        net_a.append(na)
        net_b.append(nb)
        diffs.append(na - nb)
        if na > nb:
            wins_a += 1
        elif nb > na:
            wins_b += 1
        else:
            ties += 1

    return MatchupResult(
        name_a=name_a, name_b=name_b, seeds=list(seeds),
        diffs=diffs, net_a=net_a, net_b=net_b,
        wins_a=wins_a, wins_b=wins_b, ties=ties,
        mean_diff=(sum(diffs) / len(diffs) if diffs else 0.0),
        t_test=paired_t_test(diffs),
    )


def evaluate_roster(roster: Dict[str, AgentFactory], seeds: List[int],
                    n_hands: int = 200, small_blind: int = DEFAULT_SMALL_BLIND,
                    big_blind: int = DEFAULT_BIG_BLIND,
                    starting_stack: int = 1000,
                    fast_mode: bool = False) -> RosterResult:
    """
    Round-robin every unordered pair via `evaluate_matchup`, reusing the
    tournament's deterministic per-pair seeding, and aggregate into a leaderboard
    (mean net chips), a directed win matrix, and per-agent net-chip samples (for
    a PnL distribution box/violin).
    """
    names = list(roster.keys())
    win_matrix = {a: {b: 0 for b in names if b != a} for a in names}
    per_agent_nets = {a: [] for a in names}
    net_total = {a: 0.0 for a in names}
    n_matches = {a: 0 for a in names}
    pair_results: Dict[Tuple[str, str], MatchupResult] = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pair_seeds = [_pair_seed(a, b, k) for k in range(len(seeds))]
            mr = evaluate_matchup(
                roster[a], roster[b], a, b, pair_seeds,
                n_hands=n_hands, small_blind=small_blind, big_blind=big_blind,
                starting_stack=starting_stack, fast_mode=fast_mode,
            )
            pair_results[(a, b)] = mr
            win_matrix[a][b] += mr.wins_a
            win_matrix[b][a] += mr.wins_b
            per_agent_nets[a].extend(mr.net_a)
            per_agent_nets[b].extend(mr.net_b)
            net_total[a] += sum(mr.net_a)
            net_total[b] += sum(mr.net_b)
            n_matches[a] += len(mr.seeds)
            n_matches[b] += len(mr.seeds)

    leaderboard = sorted(
        ({"name": a,
          "mean_net_chips": (net_total[a] / n_matches[a]) if n_matches[a] else 0.0,
          "n_matches": n_matches[a]}
         for a in names),
        key=lambda d: d["mean_net_chips"], reverse=True,
    )
    return RosterResult(
        leaderboard=leaderboard, win_matrix=win_matrix,
        per_agent_nets=per_agent_nets, pair_results=pair_results,
        n_seeds=len(seeds),
    )


def parameter_sweep(tight_values, aggr_values, seeds, n_hands=120,
                    mc_sims=None, fast_mode=True, extra_agents=None,
                    small_blind=DEFAULT_SMALL_BLIND, big_blind=DEFAULT_BIG_BLIND,
                    starting_stack=1000):
    """
    Round-robin a GRID of static (tight_threshold × aggression) personalities —
    plus any `extra_agents` (e.g. Kelly, adaptive, RL) — so you don't have to
    hand-tune bots one at a time. Returns (RosterResult, grid) where `grid` is a
    list of {tight, aggr, mean_net_chips} cells: the personality "fitness
    landscape" behind a heatmap, and the static-skill bar an RL agent must beat.

    With `mc_sims` set and `fast_mode=False` the landscape reflects real Monte
    Carlo equity (slower); `fast_mode=True` (default) is the quick random-equity
    proxy — still a valid bot-vs-bot landscape, just betting-discipline-driven.

    Args:
        tight_values, aggr_values (list[float]): grid axes.
        seeds (list[int]): seeds per pair (paired, deterministic).
        extra_agents (dict | None): {name: factory(player_id, stack)} added to
            the grid roster (named distinctly from the t../a.. cells).
    """
    def _make(t, a):
        def factory(pid, stack):
            mc = MonteCarloEngine(mc_sims) if (mc_sims and not fast_mode) else None
            return BotPlayer(pid, f"t{t:.2f}/a{a:.2f}", stack,
                             tight_threshold=t, aggression=a, mc_engine=mc)
        return factory

    roster, cell_name = {}, {}
    for t in tight_values:
        for a in aggr_values:
            name = f"t{t:.2f}/a{a:.2f}"
            roster[name] = _make(t, a)
            cell_name[(t, a)] = name
    if extra_agents:
        roster.update(extra_agents)

    rr = evaluate_roster(roster, seeds, n_hands=n_hands,
                         small_blind=small_blind, big_blind=big_blind,
                         starting_stack=starting_stack, fast_mode=fast_mode)
    mean_by_name = {e["name"]: e["mean_net_chips"] for e in rr.leaderboard}
    grid = [{"tight": t, "aggr": a,
             "mean_net_chips": mean_by_name[cell_name[(t, a)]]}
            for t in tight_values for a in aggr_values]
    return rr, grid
