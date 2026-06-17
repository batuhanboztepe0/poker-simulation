"""
icm.py
------
Independent Chip Model (Malmuth-Harville) for tournament equity (Phase C).

Converts chip stacks into expected prize equity. Because the prize ladder is
concave, the marginal value of a chip is decreasing — the model is risk-averse
near the money and near elimination, which is the bankroll-aware correction a
Kelly / log-utility agent applies on top of single-hand EV.

Quant parallel: ICM = high-water-mark concave utility on NAV.
"""


def icm_equity(stacks, payouts):
    """
    Malmuth-Harville expected prize per player.

    P(player i finishes 1st) = stack_i / total_chips; conditioning on the
    first-place finisher and recursing assigns each subsequent place. Only as
    many places as there are payouts are enumerated (deeper places pay 0).

    Args:
        stacks (list[float]): Chip stacks (length n).
        payouts (list[float]): Prize for 1st, 2nd, ... (length <= n).

    Returns:
        list[float]: Expected prize for each player (same order as stacks).
    """
    n = len(stacks)
    if n == 0:
        return []

    # Pad prizes to one per seat (deeper places pay 0).
    prizes = list(payouts[:n])
    prizes += [0.0] * (n - len(prizes))

    equities = [0.0] * n

    live = [i for i in range(n) if stacks[i] > 0]
    busted = [i for i in range(n) if stacks[i] <= 0]
    n_live = len(live)

    # Busted players have already finished and occupy the BOTTOM places; they
    # split the remaining (lowest) prizes equally. This keeps the prize pool
    # conserved instead of stranding those places (a snapshot ICM has no
    # elimination order to break ties, so equal split is the neutral choice).
    bottom_prizes = prizes[n_live:]
    if busted and bottom_prizes:
        share = sum(bottom_prizes) / len(busted)
        for i in busted:
            equities[i] = share

    # Live players compete via Malmuth-Harville for the top n_live prizes.
    top_prizes = prizes[:n_live]
    paid = n_live

    def recurse(remaining, level, prob):
        if level >= paid or not remaining:
            return
        total = sum(stacks[i] for i in remaining)
        if total <= 0:
            return
        prize = top_prizes[level]
        for i in remaining:
            p_first = stacks[i] / total
            equities[i] += prob * p_first * prize
            if level + 1 < paid:
                rest = tuple(j for j in remaining if j != i)
                recurse(rest, level + 1, prob * p_first)

    recurse(tuple(live), 0, 1.0)
    return equities


def marginal_chip_value(stacks, payouts, player_index, delta=1.0):
    """
    Marginal ICM value of `delta` extra chips to one player.

        (ICM(stack_i + delta) - ICM(stack_i)) / delta

    Because ICM equity is concave in own chips, this is a decreasing function
    of the player's stack — the formal statement of "chips you have are worth
    more than chips you can win".

    Returns:
        float: Marginal prize value per chip.
    """
    base = icm_equity(stacks, payouts)[player_index]
    bumped = list(stacks)
    bumped[player_index] += delta
    new = icm_equity(bumped, payouts)[player_index]
    return (new - base) / delta
