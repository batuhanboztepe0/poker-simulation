"""
icm_eval.py
-----------
Multi-player tournament evaluation (Phase A5).

The heads-up harnesses (`rl_agent.evaluate_vs_baseline`, `evaluation.evaluate_*`)
cannot measure the ICM/bankroll angle, because in a winner-take-all heads-up
bust-match the loser simply busts — there is no concave prize ladder for
risk-aversion to exploit. This module seats N agents at a MULTI-PRIZE table,
plays seeded bankroll tournaments to a finish, and scores each agent by its
realised tournament prize (finish-order payouts, with the still-live players'
top prizes split by ICM equity of their final stacks). That makes the
"does ICM concavity pay?" comparison measurable: an ICM/log-utility agent that
correctly values survival near the money should earn more PRIZE than a
risk-neutral (chip-EV) agent, even when they earn similar chips.

Agent factories have the signature `(player_id, name, stack, mc_engine, rng) ->
Player`, so any mix of myopic / Kelly / RL agents can be seated. Seeds drive the
deck shuffle and every bot's MC sampling, so each seed is a paired observation
(all agents play the same seeded tournament); seats are rotated across seeds so
no agent keeps a positional edge.
"""

from src.icm import icm_equity


def tournament_prizes(player_ids, final_stacks, bust_order, prize_structure):
    """
    Assign each player its realised tournament prize.

    Finish order: a player that busts EARLIER finishes LOWER. Busted players
    take the bottom places (the lowest prizes), the last-to-bust taking the best
    of those; the still-live players take the top places, splitting the top
    prizes by ICM equity of their final stacks (a snapshot ICM has no
    elimination order among survivors, so chips decide). The prize pool is
    conserved: the returned prizes sum to `sum(prize_structure)`.

    Args:
        player_ids (list): Player ids, one per seat.
        final_stacks (dict): {player_id: final_stack}.
        bust_order (list): Player ids in the order they busted (earliest first);
            excludes players still live at the stopping point.
        prize_structure (list[float]): Prize for 1st, 2nd, ... (padded with 0).

    Returns:
        dict: {player_id: prize}.
    """
    n = len(player_ids)
    prizes = list(prize_structure[:n]) + [0.0] * max(0, n - len(prize_structure))

    survivors = [pid for pid in player_ids if pid not in bust_order]
    n_surv = len(survivors)

    out = {}
    # Busters take the bottom places; reverse bust order so the LAST player to
    # bust takes the best of the bottom prizes.
    bottom_prizes = prizes[n_surv:]
    for prize, pid in zip(bottom_prizes, reversed(bust_order)):
        out[pid] = prize

    # Survivors split the top n_surv prizes by ICM equity of their final stacks.
    top_prizes = prizes[:n_surv]
    surv_stacks = [final_stacks[pid] for pid in survivors]
    for pid, eq in zip(survivors, icm_equity(surv_stacks, top_prizes)):
        out[pid] = eq
    return out


def play_tournament(players, prize_structure, max_hands=200,
                    small_blind=10, big_blind=20, rng=None):
    """
    Play one bankroll tournament to a finish and return each seat's prize.

    Stacks persist across hands; a player that reaches 0 chips is out (its bust
    is recorded in order). The tournament ends when fewer than two players have
    chips or `max_hands` is reached, after which `tournament_prizes` scores it.

    Returns:
        dict: {player_id: prize}.
    """
    from src.game import GameEngine

    engine = GameEngine(players, small_blind=small_blind, big_blind=big_blind,
                        verbose=False, rng=rng)
    bust_order = []
    for _ in range(max_hands):
        if sum(1 for p in players if p.stack > 0) < 2:
            break
        engine.play_hand()
        for p in players:
            if p.stack <= 0 and p.player_id not in bust_order:
                bust_order.append(p.player_id)

    ids = [p.player_id for p in players]
    final_stacks = {p.player_id: p.stack for p in players}
    return tournament_prizes(ids, final_stacks, bust_order, prize_structure)


def evaluate_icm_tournament(factories, prize_structure, seeds, names=None,
                            stack0=1000, max_hands=200, mc_sims=100,
                            small_blind=10, big_blind=20, rotate_seats=True):
    """
    Seat `len(factories)` agents at a multi-prize table and measure each agent's
    mean realised tournament prize over `seeds` seeded bankroll tournaments.

    Each agent factory has signature `(player_id, name, stack, mc_engine, rng) ->
    Player`. Seats are rotated each seed (agent i sits in seat (i + seed) % n) so
    positional advantage cancels over the seed set, while each seed remains a
    paired observation (all agents share the seeded deck / MC stream).

    Args:
        factories (list[callable]): One agent factory per seat.
        prize_structure (list[float]): Prize ladder (1st, 2nd, ...).
        seeds (iterable[int]): Tournament seeds.
        names (list[str] | None): Agent names (defaults to A0, A1, ...).
        stack0 (int): Starting stack for every agent.
        max_hands (int): Hand cap per tournament.
        mc_sims (int): Monte-Carlo simulations per equity estimate.
        small_blind, big_blind (int): Blind structure.
        rotate_seats (bool): Rotate seating across seeds to cancel position.

    Returns:
        dict: {agent_name: {"mean_prize": float, "per_seed": list[float]}}.
    """
    import random as _random
    from src.monte_carlo import MonteCarloEngine

    n = len(factories)
    if names is None:
        names = [f"A{i}" for i in range(n)]

    per_agent = {i: [] for i in range(n)}
    for seed in seeds:
        rng = _random.Random(seed)
        mc = MonteCarloEngine(n_simulations=mc_sims, rng=rng)
        shift = (seed % n) if rotate_seats else 0

        players = []
        seat_agent = {}            # player_id -> agent index
        for seat in range(n):
            agent_i = (seat - shift) % n
            pid = seat + 1
            players.append(
                factories[agent_i](pid, names[agent_i], stack0, mc, rng)
            )
            seat_agent[pid] = agent_i

        prizes = play_tournament(players, prize_structure, max_hands=max_hands,
                                 small_blind=small_blind, big_blind=big_blind,
                                 rng=rng)
        for pid, prize in prizes.items():
            per_agent[seat_agent[pid]].append(prize)

    return {
        names[i]: {
            "mean_prize": sum(v) / len(v) if v else 0.0,
            "per_seed": v,
        }
        for i, v in per_agent.items()
    }
