"""
kelly_agent.py
--------------
Bankroll-aware bot using Kelly sizing + an ICM/log-utility risk correction
(Phase C).

`KellyBotPlayer` extends the baseline EV bot in two ways:
    1. Raises are sized by a fractional Kelly fraction of the stack (growth
       optimal, variance-hedged) rather than the always-minimum no-fold-equity
       optimum.
    2. Committing a large fraction of the stack (calling an all-in) demands an
       extra edge over pot odds that grows with the fraction at risk and with
       risk aversion (1 - kelly_scalar). This is the log-utility / ICM
       "chips you have are worth more than chips you can win" effect: the bot
       declines thin gambles for its tournament life that a risk-neutral myopic
       bot would take.

OFF by default for the rest of the system — it is a drop-in BotPlayer subclass,
so existing bots and tests are unaffected.
"""

from src.player import BotPlayer
from src.ev_calculator import (
    pot_odds, should_call, ev_raise, optimal_raise_size,
    kelly_fraction, kelly_raise_size,
)
from src.player import (
    ACTION_FOLD, ACTION_CHECK, ACTION_CALL, ACTION_RAISE, ACTION_ALL_IN,
)

# Extra edge (in equity) demanded to risk the whole stack, before scaling by
# risk aversion and the fraction of stack at risk.
RUIN_AVERSION = 0.15


class KellyBotPlayer(BotPlayer):
    """
    Kelly / ICM-aware bot.

    Args:
        kelly_scalar (float): Fractional-Kelly multiplier in (0, 1]; smaller =
            more conservative sizing and more ruin-averse (default 0.5).
        prize_structure (list | None): Optional tournament prize ladder; when
            present the bot reads it for ICM context (the heuristic ruin gate
            already captures the directional effect).
    """

    def __init__(self, *args, kelly_scalar=0.5, prize_structure=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not (0.0 < kelly_scalar <= 1.0):
            raise ValueError(
                f"kelly_scalar must be in (0, 1], got {kelly_scalar}."
            )
        self.kelly_scalar = kelly_scalar
        self.prize_structure = prize_structure

    def decide(self, game_state):
        call_amount = game_state.get("call_amount", 0)
        pot = game_state.get("pot", 1)
        min_raise = game_state.get("min_raise", self.big_blind_guess(call_amount))
        max_raise = self.stack

        equity = self._estimate_equity(game_state)
        self.last_equity = equity

        # Style filter: tight players still fold marginal hands to a bet.
        if equity < self.tight_threshold and call_amount > 0:
            return ACTION_FOLD, 0

        if call_amount == 0:
            size = self._kelly_raise(equity, pot, min_raise, max_raise)
            if size:
                return ACTION_RAISE, size
            return ACTION_CHECK, 0

        # Pot-odds gate.
        if not should_call(equity, pot, call_amount):
            return ACTION_FOLD, 0

        # Committing (near-)all-in: demand the ruin-aversion edge.
        if call_amount >= self.stack:
            if self._commit_ok(equity, pot, call_amount):
                return ACTION_ALL_IN, self.stack
            return ACTION_FOLD, 0

        size = self._kelly_raise(equity, pot, min_raise, max_raise)
        if size and size > call_amount:
            return ACTION_RAISE, size
        return ACTION_CALL, call_amount

    # ------------------------------------------------------------------

    def _kelly_raise(self, equity, pot, min_raise, max_raise):
        """Fractional-Kelly raise size (chips), or 0 if no +EV raise."""
        if max_raise < min_raise:
            return 0
        # Payoff odds of betting one min-raise unit to win the pot.
        payoff_odds = pot / max(1, min_raise)
        size = kelly_raise_size(
            equity, payoff_odds, self.stack, min_raise, max_raise,
            kelly_scalar=self.kelly_scalar,
        )
        if size <= 0:
            return 0
        # Without fold equity, EV(raise) decreases with size; if the Kelly
        # size overshoots break-even, fall back to the +EV-optimal (smallest)
        # raise so a genuine edge is still bet rather than silently checked.
        # The EV gate uses the net additional (pot already holds hero's bet).
        if ev_raise(equity, pot, max(1, size - self.current_bet)) <= 0:
            size = optimal_raise_size(equity, pot, min_raise, max_raise)
        if ev_raise(equity, pot, max(1, size - self.current_bet)) <= 0:
            return 0
        return size

    def _commit_ok(self, equity, pot, call_amount):
        """
        Whether to stack off: equity must clear pot odds by a ruin-aversion
        margin that scales with the fraction of stack at risk and (1 -
        kelly_scalar).
        """
        odds = pot_odds(call_amount, pot)
        risk_fraction = min(1.0, call_amount / max(1, self.stack))
        margin = (1.0 - self.kelly_scalar) * RUIN_AVERSION * risk_fraction
        return equity >= odds + margin
