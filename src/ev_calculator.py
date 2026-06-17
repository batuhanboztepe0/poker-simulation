"""
ev_calculator.py
----------------
Expected value and pot odds calculations for Texas Hold'em.

All functions are pure (no side effects, no state) to support
easy unit testing and future RL reward signal integration.

Core decision criterion:
    EV(call) = equity * (pot + call) - call
    EV(fold) = 0 (by definition)

    Call iff EV(call) > 0, equivalently: equity > pot_odds
    where pot_odds = call / (pot + call)

Raise sizing:
    EV(raise) = equity * (pot + raise) - raise
    Optimal raise size maximises EV subject to stack constraints.
    Fold equity is excluded here (Phase 5).
"""

# Minimum pot odds threshold below which calling is never rational
MIN_RATIONAL_EQUITY = 0.0

# Raise sizing multipliers relative to pot (Phase A adds 0.33-pot)
RAISE_SIZE_OPTIONS = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0]  # as fraction of pot

# EV rounding precision
EV_PRECISION = 4


def pot_odds(call_amount, pot):
    """
    Calculate the pot odds for a call decision.

    Pot odds represent the minimum equity required to make a call
    break-even. If equity >= pot_odds, the call has non-negative EV.

    Args:
        call_amount (int): Chips required to call.
        pot (int): Current pot size before the call.

    Returns:
        float: Pot odds in [0, 1). Returns 0.0 if call_amount is 0
               (free to act).

    Raises:
        ValueError: If call_amount or pot are negative, or pot is zero
                    when call_amount > 0.
    """
    if call_amount < 0:
        raise ValueError(
            f"call_amount must be non-negative, got {call_amount}."
        )
    if pot < 0:
        raise ValueError(
            f"pot must be non-negative, got {pot}."
        )
    if call_amount == 0:
        return 0.0
    if pot == 0:
        raise ValueError(
            "pot cannot be 0 when call_amount > 0. "
            "This indicates a game state error."
        )

    return round(call_amount / (pot + call_amount), EV_PRECISION)


def ev_call(equity, pot, call_amount):
    """
    Calculate the expected value of calling.

    EV(call) = equity * (pot + call) - call
             = equity * pot - (1 - equity) * call

    Args:
        equity (float): Hero's estimated win probability in [0, 1].
        pot (int): Current pot before the call.
        call_amount (int): Chips required to call.

    Returns:
        float: Expected value. Positive = profitable call.

    Raises:
        ValueError: If equity is outside [0, 1].
    """
    _validate_equity(equity)
    if call_amount < 0:
        raise ValueError(
            f"call_amount must be non-negative, got {call_amount}."
        )
    if call_amount == 0:
        return 0.0

    result = equity * (pot + call_amount) - call_amount
    return round(result, EV_PRECISION)


def ev_raise(equity, pot, raise_amount):
    """
    Calculate the expected value of raising (no fold equity).

    EV(raise) = equity * (pot + raise) - raise

    Note: This assumes opponents always call. Fold equity
    (probability opponent folds * pot won) is added in Phase 5.

    Args:
        equity (float): Hero's estimated win probability in [0, 1].
        pot (int): Current pot before the raise.
        raise_amount (int): Total chips put in by hero for this raise.

    Returns:
        float: Expected value of the raise.
    """
    _validate_equity(equity)
    if raise_amount <= 0:
        raise ValueError(
            f"raise_amount must be positive, got {raise_amount}."
        )

    result = equity * (pot + raise_amount) - raise_amount
    return round(result, EV_PRECISION)


def should_call(equity, pot, call_amount):
    """
    Determine whether calling is rational based on EV.

    Equivalent to: equity >= pot_odds(call_amount, pot)

    Args:
        equity (float): Estimated win probability.
        pot (int): Current pot size.
        call_amount (int): Chips to call.

    Returns:
        bool: True if calling has non-negative EV.
    """
    if call_amount == 0:
        return True  # Free to act: always at least check

    odds = pot_odds(call_amount, pot)
    return equity >= odds


def optimal_raise_size(equity, pot, min_raise, max_raise):
    """
    Find the raise size that maximises EV from the available options.

    Tests a set of standard raise sizes (fractions of the pot) plus
    the min and max raise boundaries. Returns the size with highest EV.

    Args:
        equity (float): Estimated win probability.
        pot (int): Current pot before raise.
        min_raise (int): Minimum legal raise (chips).
        max_raise (int): Maximum raise (typically player's stack).

    Returns:
        int: Optimal raise amount in chips.

    Raises:
        ValueError: If min_raise > max_raise or either is non-positive.
    """
    if min_raise <= 0:
        raise ValueError(
            f"min_raise must be positive, got {min_raise}."
        )
    if max_raise < min_raise:
        raise ValueError(
            f"max_raise ({max_raise}) must be >= min_raise ({min_raise})."
        )

    # Generate candidate sizes: fractions of pot + boundaries
    candidates = set()
    candidates.add(min_raise)
    candidates.add(max_raise)

    for fraction in RAISE_SIZE_OPTIONS:
        size = int(pot * fraction)
        if min_raise <= size <= max_raise:
            candidates.add(size)

    best_size = min_raise
    best_ev = ev_raise(equity, pot, min_raise)

    for size in candidates:
        candidate_ev = ev_raise(equity, pot, size)
        if candidate_ev > best_ev:
            best_ev = candidate_ev
            best_size = size

    return best_size


def ev_summary(equity, pot, call_amount, min_raise, max_raise):
    """
    Compute EV for all three actions: fold, call, raise.

    Useful for logging and analytics (Phase 7).

    Args:
        equity (float): Estimated win probability.
        pot (int): Current pot size.
        call_amount (int): Chips to call (0 if checking is available).
        min_raise (int): Minimum legal raise.
        max_raise (int): Maximum raise (player stack).

    Returns:
        dict: {
            'fold': 0.0,
            'call': float,
            'raise': float,
            'pot_odds': float,
            'optimal_raise_size': int,
            'best_action': str
        }
    """
    fold_ev = 0.0
    call_ev = ev_call(equity, pot, call_amount)
    raise_size = optimal_raise_size(equity, pot, min_raise, max_raise)
    raise_ev = ev_raise(equity, pot, raise_size)
    odds = pot_odds(call_amount, pot)

    evs = {
        "fold": fold_ev,
        "call": call_ev if call_amount > 0 else fold_ev,
        "raise": raise_ev,
    }
    best_action = max(evs, key=evs.get)

    return {
        "fold": fold_ev,
        "call": call_ev,
        "raise": raise_ev,
        "pot_odds": odds,
        "equity": round(equity, EV_PRECISION),
        "optimal_raise_size": raise_size,
        "best_action": best_action,
    }


# ------------------------------------------------------------------
# Phase A — fold equity & strategic bluffing
# ------------------------------------------------------------------

def gto_bluff_ratio(raise_amount, pot):
    """
    GTO bluff-to-value ratio b / (pot + b).

    This is also the risk-neutral ("indifference") fold probability: at exactly
    this p_fold a pure bluff of size `raise_amount` breaks even. The opponent's
    minimum defense frequency is its complement, pot / (pot + b).

    Args:
        raise_amount (int): The bet/raise size b (chips).
        pot (int): Pot before the raise.

    Returns:
        float: b / (pot + b) in [0, 1).
    """
    if raise_amount <= 0:
        raise ValueError(f"raise_amount must be positive, got {raise_amount}.")
    if pot < 0:
        raise ValueError(f"pot must be non-negative, got {pot}.")
    return round(raise_amount / (pot + raise_amount), EV_PRECISION)


def minimum_defense_frequency(raise_amount, pot):
    """
    Minimum defense frequency MDF = pot / (pot + b).

    The fraction of its range the bettor's opponent must continue with to
    avoid being exploited by a pure bluff.
    """
    if raise_amount <= 0:
        raise ValueError(f"raise_amount must be positive, got {raise_amount}.")
    if pot < 0:
        raise ValueError(f"pot must be non-negative, got {pot}.")
    return round(pot / (pot + raise_amount), EV_PRECISION)


def breakeven_pfold(equity, pot, raise_amount):
    """
    The fold probability at which a raise of `raise_amount` breaks even.

    Derived from EV(b) = 0:
        p_fold* = max(0, (b(1 - equity) - equity*pot) / (pot + b(1 - equity)))

    For a pure bluff (equity = 0) this reduces to b / (pot + b), the GTO ratio.

    Returns:
        float: Break-even p_fold in [0, 1].
    """
    _validate_equity(equity)
    if raise_amount <= 0:
        raise ValueError(f"raise_amount must be positive, got {raise_amount}.")
    # Solving EV(b)=0 for p_fold gives (b(1-eq) - eq*pot) / ((pot+b)(1-eq)).
    if equity >= 1.0:
        return 0.0  # a nut hand never needs folds to break even
    b = raise_amount
    numerator = b * (1.0 - equity) - equity * pot
    denominator = (pot + b) * (1.0 - equity)
    if denominator <= 0:
        return 0.0
    return round(max(0.0, numerator / denominator), EV_PRECISION)


def ev_raise_with_fold_equity(equity, pot, raise_amount, p_fold):
    """
    Expected value of a raise including fold equity.

        EV(b) = p_fold*pot + (1 - p_fold)*[equity*(pot + b) - b]

    When p_fold == 0 this is exactly `ev_raise` (the baseline, no-fold-equity
    case), so the fold-equity layer is a strict superset of the MVP behavior.

    Args:
        equity (float): Hero win probability when called, in [0, 1].
        pot (int): Pot before the raise.
        raise_amount (int): Raise size b (chips).
        p_fold (float): Probability all opponents fold, in [0, 1].

    Returns:
        float: Expected value of the raise.
    """
    _validate_equity(equity)
    _validate_p_fold(p_fold)
    if raise_amount <= 0:
        raise ValueError(f"raise_amount must be positive, got {raise_amount}.")
    called_ev = equity * (pot + raise_amount) - raise_amount
    result = p_fold * pot + (1.0 - p_fold) * called_ev
    return round(result, EV_PRECISION)


def optimal_raise_size_with_fold_equity(equity, pot, min_raise, max_raise,
                                        p_fold_fn):
    """
    Raise size maximising fold-equity EV over the candidate sizes.

    `p_fold_fn(raise_amount) -> p_fold` supplies the (size-dependent) fold
    probability. When p_fold_fn returns 0 for every size, the result is
    identical to `optimal_raise_size` (which always picks the minimum, since
    without fold equity EV is decreasing in b).

    Args:
        equity (float): Hero win probability if called.
        pot (int): Pot before the raise.
        min_raise, max_raise (int): Legal raise bounds.
        p_fold_fn (callable): size -> p_fold.

    Returns:
        int: Optimal raise amount.
    """
    if min_raise <= 0:
        raise ValueError(f"min_raise must be positive, got {min_raise}.")
    if max_raise < min_raise:
        raise ValueError(
            f"max_raise ({max_raise}) must be >= min_raise ({min_raise})."
        )

    candidates = {min_raise, max_raise}
    for fraction in RAISE_SIZE_OPTIONS:
        size = int(pot * fraction)
        if min_raise <= size <= max_raise:
            candidates.add(size)

    best_size = min_raise
    best_ev = ev_raise_with_fold_equity(
        equity, pot, min_raise, p_fold_fn(min_raise)
    )
    for size in candidates:
        candidate_ev = ev_raise_with_fold_equity(
            equity, pot, size, p_fold_fn(size)
        )
        if candidate_ev > best_ev:
            best_ev = candidate_ev
            best_size = size
    return best_size


# ------------------------------------------------------------------
# Phase C — Kelly / growth-optimal bet sizing
# ------------------------------------------------------------------

def kelly_fraction(win_prob, payoff_odds, loss_odds=1.0):
    """
    Classic Kelly fraction f* = p/l - (1 - p)/b.

    Args:
        win_prob (float): Probability of winning the bet, p in [0, 1].
        payoff_odds (float): Net odds received on a win, b > 0 (win b per unit).
        loss_odds (float): Units lost on a loss, l > 0 (default 1).

    Returns:
        float: The growth-optimal fraction of bankroll to wager. May be
            negative (no bet) when the edge is unfavorable.
    """
    _validate_equity(win_prob)
    if payoff_odds <= 0:
        raise ValueError(f"payoff_odds must be positive, got {payoff_odds}.")
    if loss_odds <= 0:
        raise ValueError(f"loss_odds must be positive, got {loss_odds}.")
    return win_prob / loss_odds - (1.0 - win_prob) / payoff_odds


def kelly_raise_size(win_prob, payoff_odds, stack, min_raise, max_raise,
                     kelly_scalar=0.5, loss_odds=1.0):
    """
    Translate a Kelly fraction into a clamped raise size (chips).

        size = round(kelly_scalar * max(0, f*) * stack), clamped to [min, max]

    `kelly_scalar` in (0, 1] applies fractional Kelly to hedge equity-model
    error (default 0.5 = half-Kelly). Returns 0 when the edge is non-positive
    (no raise).

    Returns:
        int: Raise size in chips, or 0 for "do not raise".
    """
    f = max(0.0, kelly_fraction(win_prob, payoff_odds, loss_odds)) * kelly_scalar
    target = int(round(f * stack))
    if target <= 0:
        return 0
    return max(min_raise, min(target, max_raise))


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _validate_p_fold(p_fold):
    """Validate that p_fold is a probability in [0, 1]."""
    if not (0.0 <= p_fold <= 1.0):
        raise ValueError(f"p_fold must be in [0, 1], got {p_fold}.")


def _validate_equity(equity):
    """
    Validate that equity is in [0, 1].

    Raises:
        ValueError: If equity is outside valid range.
    """
    if not (0.0 <= equity <= 1.0):
        raise ValueError(
            f"equity must be in [0, 1], got {equity}."
        )
