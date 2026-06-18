"""
adaptive_agent.py
-----------------
Bots whose aggression is NOT static — the playing-style counterpart to Phase B's
opponent-tilt *models*. Two styles:

    random : aggression is resampled every hand, uniformly in [aggr_lo, aggr_hi].
    tilt   : a two-regime {normal, tilted} switch driven by realised PnL. After a
             losing hand the bot is more likely to tilt (raise more, fold less);
             it recovers with a fixed probability. Reuses Phase B's `TiltTrigger`
             (P(normal->tilted) rises with the loss fraction), so the tilt belief
             models finally have a genuine tilting opponent to detect, and the RL
             agent gets a non-stationary, exploitable opponent to learn against.

It is a `BotPlayer` subclass that only mutates its own `aggression` /
`tight_threshold` between hands — existing bots and tests are unaffected.

Quant parallel: a counterparty whose risk appetite shifts with its drawdown
(drawdown-triggered regime change) rather than staying constant.
"""

from src.player import BotPlayer, DEFAULT_TIGHT_THRESHOLD
from src.opponent_model import TiltTrigger

ADAPTIVE_MODES = ("random", "tilt")


class AdaptiveBotPlayer(BotPlayer):
    """
    EV bot with a non-stationary aggression / tightness.

    Args (beyond BotPlayer's mc_engine / rng / belief_state / fold_equity_model):
        mode (str): "random" or "tilt".
        aggr_lo, aggr_hi (float): range for the "random" mode resample.
        base_aggression, base_tight (float): the "normal"-regime style.
        tilt_aggression, tilt_tight (float): the "tilted"-regime style (more
            aggressive, looser). Used only in "tilt" mode.
        recover (float): per-hand P(tilted -> normal).
        tilt_trigger (TiltTrigger | None): PnL->tilt transition (Phase B).

    The style is recomputed in `reset_for_hand`, from the realised chip delta of
    the hand just finished, so `decide()` (inherited unchanged) reads the freshly
    updated `aggression` / `tight_threshold`.
    """

    def __init__(self, player_id, name, stack, mode="tilt",
                 aggr_lo=0.15, aggr_hi=0.9,
                 base_aggression=0.35, base_tight=DEFAULT_TIGHT_THRESHOLD,
                 tilt_aggression=0.9, tilt_tight=0.1,
                 recover=0.25, tilt_trigger=None, **kwargs):
        if mode not in ADAPTIVE_MODES:
            raise ValueError(f"mode must be one of {ADAPTIVE_MODES}, got {mode!r}")
        super().__init__(player_id, name, stack,
                         tight_threshold=base_tight, aggression=base_aggression,
                         **kwargs)
        self.mode = mode
        self.aggr_lo = aggr_lo
        self.aggr_hi = aggr_hi
        self.base_aggression = base_aggression
        self.base_tight = base_tight
        self.tilt_aggression = tilt_aggression
        self.tilt_tight = tilt_tight
        self.recover = recover
        self.tilt_trigger = tilt_trigger or TiltTrigger(stack0=stack)
        self.is_tilted = False
        self._prev_stack = stack

    def _update_style(self):
        """Recompute aggression / tightness from the last hand's realised PnL."""
        delta = self.stack - self._prev_stack
        self._prev_stack = self.stack

        if self.mode == "random":
            self.aggression = (self.aggr_lo
                               + (self.aggr_hi - self.aggr_lo) * self._rng.random())
            return

        # tilt mode: a PnL-driven {normal, tilted} regime switch.
        if self.is_tilted:
            if self._rng.random() < self.recover:
                self.is_tilted = False
        elif self._rng.random() < self.tilt_trigger.transition_prob(delta):
            self.is_tilted = True

        if self.is_tilted:
            self.aggression = self.tilt_aggression
            self.tight_threshold = self.tilt_tight
        else:
            self.aggression = self.base_aggression
            self.tight_threshold = self.base_tight

    def reset_for_hand(self):
        super().reset_for_hand()
        self._update_style()
