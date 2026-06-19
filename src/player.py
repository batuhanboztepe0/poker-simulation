"""
player.py
---------
Player abstractions for Texas Hold'em simulation.

Defines the base Player class with chip/action management,
HumanPlayer (terminal input), and BotPlayer.

Phase 1: Random equity proxy.
Phase 2: MonteCarloEngine injection for real equity.
Phase 3: EV-based call/fold decisions. Raise sizing via optimal_raise_size().
Phase 5: Fold equity added to raise EV.
Phase 6: RL agent replaces BotPlayer entirely.
"""

import random

from src.ev_calculator import (
    should_call,
    optimal_raise_size,
    ev_call,
    ev_raise,
    pot_odds,
    ev_summary,
    ev_raise_with_fold_equity,
    optimal_raise_size_with_fold_equity,
)

ACTION_FOLD   = "fold"
ACTION_CHECK  = "check"
ACTION_CALL   = "call"
ACTION_RAISE  = "raise"
ACTION_ALL_IN = "all_in"

VALID_ACTIONS = {ACTION_FOLD, ACTION_CHECK, ACTION_CALL, ACTION_RAISE, ACTION_ALL_IN}

DEFAULT_TIGHT_THRESHOLD = 0.5
DEFAULT_AGGRESSION      = 0.5
BOT_RAISE_MULTIPLIER_MIN = 2
BOT_RAISE_MULTIPLIER_MAX = 4

# Phase 3: minimum opponent observations before trusting the belief model;
# below this we fall back to the uniform unknown-opponents equity estimate.
MIN_HANDS_FOR_BELIEF = 10
# Number of candidate hands drawn from a belief range per equity estimate.
RANGE_SAMPLE_SIZE = 200

# Phase A: with a fold-equity model, `aggression` is repurposed from a
# raise *probability* into an EV-*margin* / risk-aversion threshold. The
# required raise margin over the next-best action is (1 - aggression) scaled
# by the pot: aggression=1 raises on any +EV edge, aggression=0 demands a
# clear edge. This is a deliberate semantic change for fold-equity-aware raising.
FOLD_EQUITY_MARGIN_SCALE = 0.05

# Phase A2 (opt-in): per-street offsets applied to the base tight_threshold /
# aggression when `street_aware=True`. Rationale: tighten pre-flop (most streets
# still to navigate, equity least realized), loosen on later streets (pot odds
# improve and equity is more certain), and bet more aggressively post-flop
# (polarized value/bluff betting). The shifted knobs are clamped to [0, 1].
# Default off -> the base knobs are used unchanged and the bot is byte-identical.
STREET_TIGHT_DELTA = {
    "Pre-Flop": +0.05,
    "Flop":      0.00,
    "Turn":     -0.03,
    "River":    -0.05,
}
STREET_AGGR_DELTA = {
    "Pre-Flop":  0.00,
    "Flop":     +0.05,
    "Turn":     +0.05,
    "River":    +0.10,
}


def _clamp01(x):
    """Clamp a float to the closed interval [0, 1]."""
    return max(0.0, min(1.0, x))


class Player:
    """
    Base class for all players (human and bot).

    Manages chip stack, hole cards, and current-hand state.
    Subclasses implement the decide() method.
    """

    def __init__(self, player_id, name, stack):
        """
        Initialize a player.

        Args:
            player_id (int): Unique numeric identifier.
            name (str): Display name.
            stack (int): Starting chip count.

        Raises:
            ValueError: If stack is not a positive integer.
        """
        if not isinstance(stack, int) or stack <= 0:
            raise ValueError(
                f"Stack must be a positive integer, got {stack!r}."
            )
        self.player_id = player_id
        self.name      = name
        self.stack     = stack

        self.hole_cards    = []
        self.current_bet   = 0
        self.total_invested = 0
        self.is_folded     = False
        self.is_all_in     = False
        self.is_active     = True

    def receive_cards(self, cards):
        """
        Deal hole cards to the player.

        Args:
            cards (list[Card]): Exactly 2 hole cards.

        Raises:
            ValueError: If not exactly 2 cards are provided.
        """
        if len(cards) != 2:
            raise ValueError(
                f"A player must receive exactly 2 hole cards, got {len(cards)}."
            )
        self.hole_cards = cards

    def post_blind(self, amount):
        """
        Force-post a blind. Handles partial all-in.

        Args:
            amount (int): Blind amount.

        Returns:
            int: Actual amount posted.
        """
        actual = min(amount, self.stack)
        self.stack       -= actual
        self.current_bet += actual
        self.total_invested += actual
        if self.stack == 0:
            self.is_all_in = True
        return actual

    def place_bet(self, amount):
        """
        Place a bet or call, deducting from the stack.

        Args:
            amount (int): Intended bet amount.

        Returns:
            int: Actual amount placed (capped at stack).
        """
        actual = min(amount, self.stack)
        self.stack       -= actual
        self.current_bet += actual
        self.total_invested += actual
        if self.stack == 0:
            self.is_all_in = True
        return actual

    def reset_for_hand(self):
        """Reset all per-hand state."""
        self.hole_cards     = []
        self.current_bet    = 0
        self.total_invested = 0
        self.is_folded      = False
        self.is_all_in      = False
        self.is_active      = self.stack > 0

    def reset_for_round(self):
        """Reset bet tracking for a new betting round."""
        self.current_bet = 0

    def decide(self, game_state):
        """
        Choose an action given the current game state.

        Args:
            game_state (dict): Table snapshot.

        Returns:
            tuple: (action, amount)

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement decide()."
        )

    def observe_action(self, observation):
        """
        Post-action callback: observe an opponent's action (Phase 0 hook).

        Called by the engine after every action for every player except the
        actor, so an opponent-modeling bot can update beliefs without
        touching engine internals. The base implementation is a no-op.

        Args:
            observation (dict): Keys include hand_id, street, actor_id,
                action, amount, pot, current_bet, is_aggressive.
        """
        pass

    def observe_hand_result(self, observation):
        """
        Post-hand callback: observe another player's realised per-hand chip
        delta (Phase B tilt hook). The engine calls this once per other player
        at the end of every hand, so a PnL-driven belief can transition its
        tilt regime at the hand boundary. Base implementation is a no-op.

        Args:
            observation (dict): Keys include hand_id, player_id (the player whose
                delta this is), delta_stack.
        """
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"id={self.player_id}, name='{self.name}', stack={self.stack})"
        )


class HumanPlayer(Player):
    """
    Human-controlled player that reads decisions from terminal input.
    """

    def __init__(self, player_id, name, stack):
        super().__init__(player_id, name, stack)

    def decide(self, game_state):
        """
        Prompt the human player for an action via terminal.

        Args:
            game_state (dict): Keys include pot, call_amount, min_raise,
                community_cards, round_name.

        Returns:
            tuple: (action, amount)
        """
        self._display_state(game_state)
        available = self._available_actions(game_state)
        print(f"  Available actions: {', '.join(available)}")

        while True:
            raw = input("  Your action: ").strip().lower()
            if raw not in available:
                print(f"  Invalid action. Choose from: {', '.join(available)}")
                continue

            if raw == ACTION_RAISE:
                amount = self._prompt_raise_amount(game_state)
                return ACTION_RAISE, amount

            if raw == ACTION_CALL:
                return ACTION_CALL, game_state["call_amount"]

            return raw, 0

    def _display_state(self, game_state):
        """Print the current game state for the human player."""
        community = game_state.get("community_cards", [])
        board_str = "  ".join(str(c) for c in community) if community else "(none)"
        print("\n" + "=" * 50)
        print(f"  Round      : {game_state.get('round_name', '?')}")
        print(f"  Your cards : {str(self.hole_cards[0])}  {str(self.hole_cards[1])}")
        print(f"  Board      : {board_str}")
        print(f"  Pot        : {game_state.get('pot', 0)} chips")
        print(f"  Your stack : {self.stack} chips")
        print(f"  To call    : {game_state.get('call_amount', 0)} chips")
        print("=" * 50)

    def _available_actions(self, game_state):
        """Return list of legal action strings."""
        actions = [ACTION_FOLD]
        call_amount = game_state.get("call_amount", 0)

        if call_amount == 0:
            actions.append(ACTION_CHECK)
        else:
            actions.append(ACTION_CALL)

        if self.stack > call_amount:
            actions.append(ACTION_RAISE)

        actions.append(ACTION_ALL_IN)
        return actions

    def _prompt_raise_amount(self, game_state):
        """Prompt and validate a raise amount."""
        min_raise = game_state.get("min_raise", 1)
        print(f"  Minimum raise: {min_raise} chips | Your stack: {self.stack}")
        while True:
            raw = input(f"  Raise amount (>= {min_raise}): ").strip()
            try:
                amount = int(raw)
            except ValueError:
                print("  Enter a valid integer.")
                continue
            if amount < min_raise:
                print(f"  Must be at least {min_raise}.")
                continue
            if amount > self.stack:
                print(f"  Exceeds your stack ({self.stack}). Going all-in instead.")
                return self.stack
            return amount


class BotPlayer(Player):
    """
    EV-driven bot player with optional Monte Carlo equity estimation.

    Decision logic by phase:
        Phase 1: Random equity proxy, threshold-based fold/call/raise.
        Phase 2: MonteCarloEngine for real equity, same threshold logic.
        Phase 3: EV-based call/fold (should_call), EV-optimal raise sizing.
        Phase 5: Fold equity added to raise EV (personality-driven bluffing).
        Phase 6: RL agent replaces this class entirely.

    Personality parameters:
        tight_threshold (float 0-1): Minimum equity to consider continuing.
            Acts as a pre-filter before EV calculation to model
            risk-averse / tighter play styles.
        aggression (float 0-1): Probability of raising when EV allows it.
    """

    def __init__(self, player_id, name, stack,
                 tight_threshold=DEFAULT_TIGHT_THRESHOLD,
                 aggression=DEFAULT_AGGRESSION,
                 mc_engine=None, rng=None, belief_state=None,
                 fold_equity_model=None, respect_pot_odds=False,
                 street_aware=False):
        """
        Initialize a bot player.

        Args:
            player_id (int): Unique ID.
            name (str): Display name.
            stack (int): Starting chips.
            tight_threshold (float): Equity floor. Hands below this
                threshold are folded regardless of pot odds.
            aggression (float): Raise probability when raising is +EV.
            mc_engine (MonteCarloEngine | None): Equity estimator.
                If None, falls back to random proxy.
            rng (random.Random | None): Random source for the raise
                coin-flip and the no-MC equity proxy. When None, the
                module-level `random` is used (unseeded). Inject a seeded
                `random.Random(seed)` for reproducible sessions.
            respect_pot_odds (bool): When False (default) the tight_threshold
                style filter hard-folds every sub-threshold hand facing a bet,
                even +EV ones (the historical behavior). When True, a
                sub-threshold hand is only style-folded if calling is ALSO -EV
                (equity < pot_odds), closing the +EV-fold leak. Opt-in so the
                baseline bot stays byte-identical.
            street_aware (bool): When True, shift tight_threshold/aggression by
                a per-street offset (tighter pre-flop, looser/more aggressive on
                later streets; see STREET_TIGHT_DELTA / STREET_AGGR_DELTA).
                Opt-in (default False) so the baseline bot stays byte-identical.
        """
        super().__init__(player_id, name, stack)
        self.tight_threshold = tight_threshold
        self.aggression      = aggression
        self.respect_pot_odds = respect_pot_odds
        self.street_aware    = street_aware
        # Per-decision effective knobs (street-modulated when street_aware).
        # Set each decide(); initialized to the base values so the decision
        # helpers are correct even if called before a decide() (e.g. in tests).
        self._eff_tight      = tight_threshold
        self._eff_aggr       = aggression
        self.mc_engine       = mc_engine
        self._rng            = rng if rng is not None else random
        # Most recent decision-time equity estimate, surfaced on the action
        # event so analytics can score predicted-vs-realised (Phase 2).
        self.last_equity     = None
        # Optional static Bayesian opponent model (Phase 3). When present and
        # warmed up, equity is conditioned on the modeled opponent range.
        self.belief_state    = belief_state
        # Dedupe the per-hand PnL->tilt belief transition: the engine notifies
        # observe_hand_result once per OTHER player, but a single belief models
        # one (blended) opponent, so observe_pnl must fire at most once per hand.
        self._last_pnl_hand_id = None
        # Optional fold-equity model (Phase A). When present, raising becomes a
        # strict EV gate including fold equity (enables EV-driven bluffing).
        # OFF by default -> the bot is byte-identical to the baseline.
        self.fold_equity_model = fold_equity_model

    def decide(self, game_state):
        """
        Make an EV-driven decision.

        Decision flow:
            1. Estimate equity via MC (or random fallback).
            2. If equity < tight_threshold: fold (style filter).
            3. Compute EV(call). If call_amount == 0: check or raise.
            4. If EV(call) < 0: fold.
            5. Raise with probability = aggression using optimal sizing.
            6. Otherwise call/check.

        Args:
            game_state (dict): Keys:
                pot (int), call_amount (int), min_raise (int),
                community_cards (list[Card]), round_name (str),
                active_player_count (int).

        Returns:
            tuple: (action, amount)
        """
        call_amount = game_state.get("call_amount", 0)
        pot         = game_state.get("pot", 1)
        min_raise   = game_state.get("min_raise", self.big_blind_guess(call_amount))
        max_raise   = self.stack

        equity = self._estimate_equity(game_state)
        self.last_equity = equity

        # Phase A2: resolve the per-decision effective knobs (street-modulated
        # when street_aware, else the base values). The decision helpers below
        # read self._eff_tight / self._eff_aggr.
        self._eff_tight, self._eff_aggr = self._street_style(game_state)

        # Phase A: fold-equity path is a strict EV gate that can value a bluff.
        # OFF by default (fold_equity_model is None) -> baseline flow below.
        if self.fold_equity_model is not None:
            return self._decide_with_fold_equity(
                equity, pot, call_amount, min_raise, max_raise, game_state
            )

        # Style filter: tight players fold marginal hands
        if self._tight_fold(equity, pot, call_amount):
            return ACTION_FOLD, 0

        # Free action (no bet to call): check or consider raising
        if call_amount == 0:
            if self._should_raise(equity, pot, min_raise, max_raise,
                                  hero_bet=self.current_bet):
                raise_size = optimal_raise_size(equity, pot, min_raise, max_raise)
                return ACTION_RAISE, raise_size
            return ACTION_CHECK, 0

        # EV gate: fold if calling is not profitable
        if not should_call(equity, pot, call_amount):
            return ACTION_FOLD, 0

        # All-in if call exceeds or matches stack
        if call_amount >= self.stack:
            return ACTION_ALL_IN, self.stack

        # Raise with probability = aggression
        if self._should_raise(equity, pot, min_raise, max_raise):
            raise_size = optimal_raise_size(equity, pot, min_raise, max_raise)
            return ACTION_RAISE, raise_size

        return ACTION_CALL, call_amount

    def _decide_with_fold_equity(self, equity, pot, call_amount,
                                 min_raise, max_raise, game_state):
        """
        EV-gated decision including fold equity (Phase A).

        Considers the raise option first — a raise can be +EV via fold equity
        even when calling is -EV — so bluffs are valued, not random. The raise
        is taken only when its fold-equity EV beats max(EV_call, 0) by the
        aggression-derived margin. Otherwise the bot calls (subject to the
        tight-threshold edge filter and pot odds) or folds.
        """
        raise_size = self._fold_equity_raise(
            equity, pot, call_amount, min_raise, max_raise, game_state
        )

        if call_amount == 0:
            if raise_size is not None:
                return ACTION_RAISE, raise_size
            return ACTION_CHECK, 0

        # Facing a bet: a profitable raise dominates call/fold.
        if raise_size is not None:
            if call_amount >= self.stack:
                return ACTION_ALL_IN, self.stack
            return ACTION_RAISE, raise_size

        # No profitable raise -> call vs fold on edge filter + pot odds.
        if self._tight_fold(equity, pot, call_amount):
            return ACTION_FOLD, 0
        if not should_call(equity, pot, call_amount):
            return ACTION_FOLD, 0
        if call_amount >= self.stack:
            return ACTION_ALL_IN, self.stack
        return ACTION_CALL, call_amount

    def _fold_equity_raise(self, equity, pot, call_amount,
                           min_raise, max_raise, game_state):
        """
        Return the best fold-equity raise size, or None if no raise clears the
        EV gate (ev_raise_with_fold_equity > max(ev_call, 0) + margin).
        """
        if max_raise < min_raise:
            return None

        model = self.fold_equity_model
        opponent_ids = game_state.get("opponent_ids") or []
        if opponent_ids:
            n_opp = len(opponent_ids)
        else:
            n_opp = max(1, game_state.get("active_player_count", 2) - 1)

        def p_fold_fn(raise_amount):
            return model.estimate_p_fold(opponent_ids, raise_amount, pot, n_opp)

        best_size = optimal_raise_size_with_fold_equity(
            equity, pot, min_raise, max_raise, p_fold_fn
        )
        ev_raise_fe = ev_raise_with_fold_equity(
            equity, pot, best_size, p_fold_fn(best_size)
        )

        ev_call_val = ev_call(equity, pot, call_amount) if call_amount > 0 else 0.0
        baseline = max(ev_call_val, 0.0)
        margin = (1.0 - self._eff_aggr) * pot * FOLD_EQUITY_MARGIN_SCALE

        if ev_raise_fe > baseline + margin:
            return best_size
        return None

    def _street_style(self, game_state):
        """
        Return the (tight_threshold, aggression) to use for this decision.

        With `street_aware=True` the base knobs are shifted by a per-street
        offset (STREET_TIGHT_DELTA / STREET_AGGR_DELTA) and clamped to [0, 1]:
        tighter pre-flop, looser and more aggressive on later streets. With the
        flag off (default) the base knobs are returned unchanged, so the
        baseline bot is byte-identical.
        """
        if not self.street_aware:
            return self.tight_threshold, self.aggression
        rn = game_state.get("round_name")
        tight = _clamp01(self.tight_threshold + STREET_TIGHT_DELTA.get(rn, 0.0))
        aggr  = _clamp01(self.aggression + STREET_AGGR_DELTA.get(rn, 0.0))
        return tight, aggr

    def _tight_fold(self, equity, pot, call_amount):
        """
        Whether the tight_threshold style filter folds this hand.

        Baseline (`respect_pot_odds=False`): fold any sub-threshold hand facing
        a bet, even one the pot prices in as a +EV call -- the historical
        behavior (an exploitable +EV-fold leak). `respect_pot_odds=True` closes
        that leak: a sub-threshold hand is only style-folded when calling is
        ALSO -EV (`equity < pot_odds`), so a marginal +EV call is no longer
        auto-folded. Returns False when there is nothing to call (free action).
        """
        if call_amount <= 0 or equity >= self._eff_tight:
            return False
        if self.respect_pot_odds and equity >= pot_odds(call_amount, pot):
            return False
        return True

    def _should_raise(self, equity, pot, min_raise, max_raise, hero_bet=0):
        """
        Decide whether to raise. Equity-aware (Phase 0 fix).

        `hero_bet` is the hero's chips already committed this round (e.g. a
        blind pre-flop). Since `pot` already contains those chips, the EV gate
        uses the NET additional investment (raise_size - hero_bet), consistent
        with `ev_call` — otherwise the raise EV is understated pre-flop.

        .. deprecated:: Phase A
            Superseded by the fold-equity EV gate
            (`_decide_with_fold_equity` / `_fold_equity_raise`) when a
            `fold_equity_model` is attached. Retained for the baseline bot
            (no fold-equity model), whose behavior is unchanged.

        A raise is only considered when the best available raise size is
        strictly +EV (no fold equity in the MVP, so this requires
        equity > raise / (pot + raise)). Among +EV spots the bot raises
        with probability `aggression`, preserving the personality knob
        while no longer raising purely at random on losing hands.

        Args:
            equity (float): Current equity estimate.
            pot (int): Current pot.
            min_raise (int): Minimum legal raise.
            max_raise (int): Maximum raise (stack).

        Returns:
            bool: True if the bot should raise.
        """
        if max_raise < min_raise:
            return False
        raise_size = optimal_raise_size(equity, pot, min_raise, max_raise)
        additional = max(1, raise_size - hero_bet)
        if ev_raise(equity, pot, additional) <= 0:
            return False
        return self._rng.random() < self._eff_aggr

    def _estimate_equity(self, game_state):
        """
        Estimate hand equity.

        Decision path:
            - No MC engine or no hole cards -> random proxy.
            - A warmed-up belief model (>= MIN_HANDS_FOR_BELIEF observations)
              -> condition equity on the modeled opponent range.
            - Otherwise -> uniform unknown-opponents MC estimate.

        Args:
            game_state (dict): Current game state.

        Returns:
            float: Equity estimate in [0, 1].
        """
        if self.mc_engine is None or not self.hole_cards:
            return self._rng.random()

        community = game_state.get("community_cards", [])
        opponent_ids = game_state.get("opponent_ids")
        if opponent_ids:
            n_opponents = len(opponent_ids)
        else:
            n_opponents = max(1, game_state.get("active_player_count", 2) - 1)

        # Belief-conditioned path (Phase 3). `use_belief_equity` (default True)
        # lets a subclass keep equity vanilla while still carrying a belief model
        # for other purposes (the RL agent uses belief as a feature only).
        if (self.belief_state is not None
                and getattr(self, "use_belief_equity", True)
                and self.belief_state.n_observations >= MIN_HANDS_FOR_BELIEF
                and n_opponents >= 1):
            try:
                exclude = list(self.hole_cards) + list(community)
                candidate_range = self.belief_state.range_sample(
                    RANGE_SAMPLE_SIZE, exclude, self._rng
                )
                opponent_ranges = [candidate_range for _ in range(n_opponents)]
                return self.mc_engine.estimate_equity_vs_range(
                    self.hole_cards, opponent_ranges, community
                )
            except Exception as exc:
                print(
                    f"  [Belief Warning] Range equity failed for "
                    f"{self.name}: {exc}. Falling back to unknown opponents."
                )

        try:
            return self.mc_engine.estimate_equity_unknown_opponents(
                hero_hole=self.hole_cards,
                n_opponents=n_opponents,
                community_cards=community,
            )
        except Exception as exc:
            print(
                f"  [MC Warning] Equity estimation failed for "
                f"{self.name}: {exc}. Using random fallback."
            )
            return self._rng.random()

    def observe_action(self, observation):
        """
        Update the belief model from an observed opponent action (Phase 3).

        No-op when no belief model is attached, preserving baseline behavior.
        """
        if self.belief_state is None:
            return
        n_aggressive = 1 if observation.get("is_aggressive") else 0
        self.belief_state.update(
            action=observation.get("action"),
            n_aggressive=n_aggressive,
            n_actions=1,
            delta_stack=0,
        )

    def observe_hand_result(self, observation):
        """
        Feed an opponent's realised per-hand PnL into the belief's tilt regime
        (Phase B). No-op when no belief is attached, preserving baseline
        behaviour. In heads-up the observed delta is the sole opponent's; the
        HMM's observe_pnl then fires its PnL->tilt trigger so p_tilted leads the
        aggression-only signal by a hand. (Per-action `observe_action` carries no
        realised PnL — it stays emission-only with delta_stack=0; the realised
        chip delta is a hand-boundary quantity delivered here.)

        In a 3+ player game the engine calls this once per other player; a single
        belief models one blended opponent, so the transition is applied AT MOST
        ONCE per hand (deduped by hand_id) to avoid compounding it N-1 times.
        (True multi-opponent tracking would need a per-opponent belief dict.)
        """
        if self.belief_state is None:
            return
        hid = observation.get("hand_id")
        if hid is not None and hid == self._last_pnl_hand_id:
            return
        self._last_pnl_hand_id = hid
        self.belief_state.observe_pnl(observation.get("delta_stack", 0))

    def ev_breakdown(self, game_state):
        """
        Return a full EV breakdown for the current game state.

        Useful for logging, debugging, and Phase 7 analytics.

        Args:
            game_state (dict): Current game state.

        Returns:
            dict: Output of ev_calculator.ev_summary().
        """
        call_amount = game_state.get("call_amount", 0)
        pot         = game_state.get("pot", 1)
        min_raise   = game_state.get("min_raise", self.big_blind_guess(call_amount))
        max_raise   = self.stack
        equity      = self._estimate_equity(game_state)

        # Guard: a busted / very short player cannot make a legal raise
        # (max_raise < min_raise). ev_summary -> optimal_raise_size would
        # raise ValueError, so return a raise-free breakdown instead.
        if max_raise < min_raise:
            call_ev = ev_call(equity, pot, call_amount) if call_amount > 0 else 0.0
            best = "call" if (call_amount > 0 and call_ev > 0) else "fold"
            return {
                "fold": 0.0,
                "call": call_ev,
                "raise": None,
                "pot_odds": pot_odds(call_amount, pot),
                "equity": round(equity, 4),
                "optimal_raise_size": None,
                "best_action": best,
            }

        return ev_summary(equity, pot, call_amount, min_raise, max_raise)

    @staticmethod
    def big_blind_guess(call_amount):
        """
        Fallback minimum raise estimate when min_raise is not in game_state.

        Args:
            call_amount (int): Current call amount.

        Returns:
            int: Estimated minimum raise.
        """
        return max(call_amount * 2, 1)
