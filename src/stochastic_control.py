"""
stochastic_control.py
---------------------
Model-based one-step rollout policy (Phase C).

`RolloutPolicy` scores each legal action by its immediate EV using the Monte
Carlo engine as the leaf equity evaluator and (optionally) a belief model as
the opponent model and a fold-equity model for the raise branch. It needs no
neural network — it is approximate dynamic programming with a depth-1 lookahead:

    argmax_a [ r_imm(a) + E_{opp ~ belief}[ value of resulting state ] ]

`RolloutBotPlayer` is a thin BotPlayer wrapper so the policy can be seated at a
table and is chip-conservation testable.
"""

from src.player import (
    BotPlayer,
    ACTION_FOLD, ACTION_CHECK, ACTION_CALL, ACTION_RAISE, ACTION_ALL_IN,
)
from src.ev_calculator import (
    ev_call, ev_raise_with_fold_equity, optimal_raise_size_with_fold_equity,
)
from src.belief_conditioned_equity import conditioned_equity
from src.player import MIN_HANDS_FOR_BELIEF, RANGE_SAMPLE_SIZE


class RolloutPolicy:
    """
    Depth-1 EV-maximizing policy over {fold, check/call, raise}.

    Args:
        mc_engine (MonteCarloEngine): leaf equity evaluator.
        beliefs (dict | None): {opponent_id: BeliefState} opponent model.
        rng (random.Random | None): random source for range sampling.
        fold_equity_model (FoldEquityModel | None): supplies p_fold for raises.
    """

    def __init__(self, mc_engine, beliefs=None, rng=None,
                 fold_equity_model=None):
        self.mc_engine = mc_engine
        self.beliefs = beliefs if beliefs is not None else {}
        self.rng = rng
        self.fold_equity_model = fold_equity_model

    def _equity(self, hero_hole, community, opponent_ids, n_opp):
        warm = [oid for oid in opponent_ids
                if oid in self.beliefs
                and self.beliefs[oid].n_observations >= MIN_HANDS_FOR_BELIEF]
        if warm and self.rng is not None:
            return conditioned_equity(
                self.mc_engine, self.beliefs, hero_hole, community,
                warm, self.rng, n_combos=RANGE_SAMPLE_SIZE,
            )
        return self.mc_engine.estimate_equity_unknown_opponents(
            hero_hole, n_opp, community
        )

    def action_values(self, hero_hole, game_state, stack):
        """
        Return (equity, {action: value}) for the legal actions. The raise value
        is stored as (ev, size); other values are scalars.
        """
        pot = max(1, game_state.get("pot", 1))
        call_amount = game_state.get("call_amount", 0)
        min_raise = game_state.get("min_raise", max(call_amount * 2, 1))
        max_raise = stack
        community = game_state.get("community_cards", [])
        opponent_ids = game_state.get("opponent_ids") or []
        n_opp = (len(opponent_ids) if opponent_ids
                 else max(1, game_state.get("active_player_count", 2) - 1))

        equity = self._equity(hero_hole, community, opponent_ids, n_opp)

        values = {ACTION_FOLD: 0.0}
        if call_amount == 0:
            values[ACTION_CHECK] = 0.0
        else:
            values[ACTION_CALL] = ev_call(equity, pot, call_amount)

        if max_raise >= min_raise:
            def p_fold_fn(b):
                if self.fold_equity_model is None:
                    return 0.0
                return self.fold_equity_model.estimate_p_fold(
                    opponent_ids, b, pot, n_opp
                )
            size = optimal_raise_size_with_fold_equity(
                equity, pot, min_raise, max_raise, p_fold_fn
            )
            ev_r = ev_raise_with_fold_equity(equity, pot, size, p_fold_fn(size))
            values[ACTION_RAISE] = (ev_r, size)

        return equity, values

    def decide(self, hero_hole, game_state, stack):
        """Return (action, amount) maximizing the depth-1 EV."""
        _equity, values = self.action_values(hero_hole, game_state, stack)
        scalar = {a: (v[0] if isinstance(v, tuple) else v)
                  for a, v in values.items()}
        best = max(scalar, key=scalar.get)
        call_amount = game_state.get("call_amount", 0)

        if best == ACTION_RAISE:
            size = values[ACTION_RAISE][1]
            if call_amount >= stack:
                return ACTION_ALL_IN, stack
            return ACTION_RAISE, size
        if best == ACTION_CALL:
            if call_amount >= stack:
                return ACTION_ALL_IN, stack
            return ACTION_CALL, call_amount
        if best == ACTION_CHECK:
            return ACTION_CHECK, 0
        return ACTION_FOLD, 0


class RolloutBotPlayer(BotPlayer):
    """BotPlayer driven by a RolloutPolicy (one-step lookahead)."""

    def __init__(self, *args, rollout_policy=None, **kwargs):
        super().__init__(*args, **kwargs)
        if rollout_policy is None:
            raise ValueError("RolloutBotPlayer requires a rollout_policy.")
        self.rollout_policy = rollout_policy

    def decide(self, game_state):
        # One equity computation: derive both the action and the analytics
        # equity from a single action_values call.
        equity, values = self.rollout_policy.action_values(
            self.hole_cards, game_state, self.stack
        )
        self.last_equity = equity
        scalar = {a: (v[0] if isinstance(v, tuple) else v)
                  for a, v in values.items()}
        best = max(scalar, key=scalar.get)
        call_amount = game_state.get("call_amount", 0)

        if best == ACTION_RAISE:
            if call_amount >= self.stack:
                return ACTION_ALL_IN, self.stack
            return ACTION_RAISE, values[ACTION_RAISE][1]
        if best == ACTION_CALL:
            if call_amount >= self.stack:
                return ACTION_ALL_IN, self.stack
            return ACTION_CALL, call_amount
        if best == ACTION_CHECK:
            return ACTION_CHECK, 0
        return ACTION_FOLD, 0
