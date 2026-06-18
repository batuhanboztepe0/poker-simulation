"""
game.py
-------
Core GameEngine for Texas Hold'em simulation.

Supports 2-9 players. Handles full blind rotation, correct multi-player
action ordering (pre-flop UTG, post-flop left of dealer), raise
re-opening, all-in side pots, and showdown with kicker resolution.

Phase 0 additions:
    - Seeded RNG: inject a `seed` or `rng` so a full session is reproducible
      (the deck draws from this rng; bots / MC share it for one-seed replay).
    - Structured event log + observer pattern: `event_log` accumulates a
      HandEvent per action / street / showdown / hand-end; `observers` are
      callables notified on each event (analytics consume this).
    - Opponent-observable state: `_build_game_state` threads the acting player
      and exposes `opponent_ids`; a post-action callback (`observe_action`)
      lets bots model opponents without touching engine internals.
    - Correctness fixes: pre-flop action now starts UTG (left of BB, not the
      BB itself); raise re-opening keyed on a full-raise increment so all-in
      under-raises do not illegally re-open; `remove_player` preserves the
      dealer's seat identity.
"""

import random

from src.card import Deck
from src.hand_evaluator import HandEvaluator
from src.pot import PotManager
from src.player import (
    ACTION_FOLD, ACTION_CHECK, ACTION_CALL,
    ACTION_RAISE, ACTION_ALL_IN
)
from src.events import (
    HandEvent, board_codes,
    EVENT_ACTION, EVENT_STREET, EVENT_SHOWDOWN, EVENT_HAND_END,
    STREET_SHOWDOWN, STREET_HAND_END,
)

ROUND_PRE_FLOP = "Pre-Flop"
ROUND_FLOP     = "Flop"
ROUND_TURN     = "Turn"
ROUND_RIVER    = "River"

BETTING_ROUNDS = [ROUND_PRE_FLOP, ROUND_FLOP, ROUND_TURN, ROUND_RIVER]

FLOP_CARD_COUNT  = 3
TURN_CARD_COUNT  = 1
RIVER_CARD_COUNT = 1

DEFAULT_SMALL_BLIND  = 10
DEFAULT_BIG_BLIND    = 20
MAX_RAISES_PER_ROUND = 4
MIN_PLAYERS          = 2
MAX_PLAYERS          = 9

# Safety bound on actions within a single betting round (guards against any
# pathological non-termination; a real round never approaches this).
MAX_ACTIONS_PER_ROUND = 1000


class GameEngine:
    """
    Orchestrates Texas Hold'em hands for 2-9 players.

    Responsibilities:
        - Deck management and card dealing
        - Blind posting with correct seat rotation
        - Betting round management: action order, raise re-opening,
          all-in handling, raise cap
        - Side pot calculation via PotManager
        - Showdown evaluation and chip distribution
        - Structured event emission (event_log + observers)

    Call play_hand() repeatedly for multi-hand sessions.
    The dealer button rotates automatically after each hand.
    """

    def __init__(self, players, small_blind=DEFAULT_SMALL_BLIND,
                 big_blind=DEFAULT_BIG_BLIND, verbose=True,
                 seed=None, rng=None, observers=None, prize_structure=None):
        """
        Initialize the game engine.

        Args:
            players (list[Player]): 2-9 players in seat order.
                Seat order is fixed; dealer button rotates through seats.
            small_blind (int): Small blind chip amount.
            big_blind (int): Big blind chip amount.
            verbose (bool): Print hand progress to terminal.
            seed (int | None): Seed for the deck rng (ignored if `rng` given).
            rng (random.Random | None): Shared random source. Pass the SAME
                instance to the bots and MC engine for one-seed reproducibility.
                When both seed and rng are None, the module-level `random` is
                used (unseeded, matching pre-Phase-0 behavior).
            observers (list[callable] | None): Callables invoked with each
                HandEvent as it is emitted.

        Raises:
            ValueError: If player count or blind structure is invalid.
        """
        if not (MIN_PLAYERS <= len(players) <= MAX_PLAYERS):
            raise ValueError(
                f"Player count must be between {MIN_PLAYERS} and "
                f"{MAX_PLAYERS}, got {len(players)}."
            )
        if small_blind <= 0 or big_blind <= small_blind:
            raise ValueError(
                f"Invalid blinds: small={small_blind}, big={big_blind}. "
                "Big blind must be strictly greater than small blind."
            )

        self.players      = players
        self.small_blind  = small_blind
        self.big_blind    = big_blind
        self.verbose      = verbose

        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random

        # dealer_index points into self.players (seat index, not active index)
        self.dealer_index    = 0
        self.community_cards = []
        self.deck            = Deck(rng=self.rng)
        self.pot_manager     = PotManager()
        self.evaluator       = HandEvaluator()
        self.hand_number     = 0
        self.current_bet     = 0
        self.last_raise_size = 0

        # Observer / event infrastructure (Phase 0)
        self.event_log = []
        self.observers = list(observers) if observers else []

        # Optional tournament prize ladder (Phase C, ICM). None = cash game.
        self.prize_structure = prize_structure

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def play_hand(self):
        """
        Execute a complete hand from deal to chip distribution.

        Returns:
            dict: {player_id: chips_won}. Players who won nothing
                  are not included.
        """
        self.hand_number += 1
        self._log(f"\n{'='*60}")
        self._log(f"  HAND #{self.hand_number}  ({len(self._active_players())} players)")
        self._log(f"{'='*60}")

        # Stacks entering the hand (= end of the previous hand), for the
        # post-hand PnL callback.
        start_stacks = {p.player_id: p.stack for p in self.players}

        self._setup_hand()
        self._post_blinds()
        self._deal_hole_cards()

        for round_name in BETTING_ROUNDS:
            if self._count_active() < 2:
                break
            self._play_betting_round(round_name)

        winnings = self._showdown()
        self._distribute_winnings(winnings)
        self._emit_hand_end(winnings)
        self._notify_hand_result(start_stacks)
        self._log_stacks()
        self.rotate_dealer()
        return winnings

    def add_observer(self, observer):
        """
        Register a callable to be notified with each emitted HandEvent.

        Args:
            observer (callable): Invoked as observer(event: HandEvent).
        """
        self.observers.append(observer)

    def rotate_dealer(self):
        """
        Advance the dealer button to the next player with chips.

        Skips eliminated players (stack == 0).
        Called automatically at the end of play_hand().
        """
        n = len(self.players)
        for _ in range(n):
            self.dealer_index = (self.dealer_index + 1) % n
            if self.players[self.dealer_index].stack > 0:
                break

    def add_player(self, player):
        """
        Add a player to the table (call between hands only).

        Args:
            player (Player): Player to seat.

        Raises:
            ValueError: If the table is already at maximum capacity.
        """
        if len(self.players) >= MAX_PLAYERS:
            raise ValueError(
                f"Table is full ({MAX_PLAYERS} players maximum)."
            )
        self.players.append(player)
        self._log(f"  {player.name} joined with {player.stack} chips.")

    def remove_player(self, player_id):
        """
        Remove an eliminated or leaving player from the table.

        Preserves the dealer's seat identity: removing a player seated before
        the dealer shifts the button index down by one so it still points at
        the same dealer; removing the dealer itself lets the button advance to
        the next seat (which now occupies the dealer's index).

        Args:
            player_id (int): ID of the player to remove.

        Raises:
            ValueError: If the player is not found.
        """
        for i, p in enumerate(self.players):
            if p.player_id == player_id:
                self.players.pop(i)
                if i < self.dealer_index:
                    self.dealer_index -= 1
                if self.dealer_index >= len(self.players):
                    self.dealer_index = 0
                return
        raise ValueError(f"Player with id={player_id} not found.")

    # ------------------------------------------------------------------
    # Hand setup
    # ------------------------------------------------------------------

    def _setup_hand(self):
        """Reset all per-hand state. Called at the start of each hand."""
        for player in self.players:
            player.reset_for_hand()

        self.community_cards = []
        self.deck.shuffle()
        self.pot_manager.reset()
        self.current_bet     = 0
        self.last_raise_size = self.big_blind

    def _post_blinds(self):
        """
        Post small and big blinds in correct seat order.

        Seat positions (all indices into self.players):
            Heads-up (2 players): dealer = SB, other = BB.
            3+ players: (dealer+1) % n = SB, (dealer+2) % n = BB.

        Handles partial blinds when a player's stack is less than
        the required blind amount.
        """
        active = self._active_players()
        n_active = len(active)
        if n_active < MIN_PLAYERS:
            raise RuntimeError(
                f"Cannot start hand with fewer than {MIN_PLAYERS} "
                f"active players, got {n_active}."
            )

        if n_active == 2:
            # Heads-up: dealer is SB
            sb_player = self.players[self.dealer_index]
            bb_player = self._next_active_from(self.dealer_index)
        else:
            sb_player = self._next_active_from(self.dealer_index)
            bb_player = self._next_active_from(
                self.players.index(sb_player)
            )

        sb_posted = sb_player.post_blind(self.small_blind)
        self.pot_manager.add_contribution(sb_player.player_id, sb_posted)

        bb_posted = bb_player.post_blind(self.big_blind)
        self.pot_manager.add_contribution(bb_player.player_id, bb_posted)

        self.current_bet     = bb_posted
        self.last_raise_size = self.big_blind

        self._log(f"\n  Dealer  : {self.players[self.dealer_index].name}")
        self._log(f"  SB      : {sb_player.name} posts {sb_posted}")
        self._log(f"  BB      : {bb_player.name} posts {bb_posted}")

        # Store BB seat index for pre-flop action order
        self._bb_index = self.players.index(bb_player)

    def _deal_hole_cards(self):
        """Deal 2 hole cards to each active player."""
        active = self._active_players()
        for player in active:
            player.receive_cards(self.deck.deal_many(2))
        self._log(f"\n  Hole cards dealt to {len(active)} players.")

    # ------------------------------------------------------------------
    # Betting rounds
    # ------------------------------------------------------------------

    def _play_betting_round(self, round_name):
        """
        Execute one complete betting round.

        Action order:
            Pre-flop: starts UTG (left of BB), wraps around to BB last.
            Post-flop: starts left of dealer, wraps around to dealer last.

        Termination & re-opening:
            Players act in seat order, cycling, until no remaining player
            "needs action" (i.e. everyone non-folded/non-all-in has matched
            the current bet AND acted since the last bet that re-opened the
            action). A bet that increases the level by at least a full
            min-raise increment re-opens the action (clears the acted set);
            an all-in for less than a full raise does NOT re-open, matching
            standard incomplete-raise rules.

        Args:
            round_name (str): One of BETTING_ROUNDS.
        """
        self._deal_community_cards(round_name)

        # Pre-flop must KEEP the posted blinds in each player's current_bet so
        # the SB only owes (big_blind - small_blind) and the BB faces 0 (its
        # option to check). Resetting here would zero the blinds and force the
        # SB to overpay and the BB to "call" its own blind. Post-flop, clear
        # per-round bets and the table bet level.
        if round_name != ROUND_PRE_FLOP:
            for player in self.players:
                player.reset_for_round()
            self.current_bet     = 0
            self.last_raise_size = self.big_blind

        self._emit_street(round_name)

        self._log(f"\n  --- {round_name} ---")
        self._log(f"  Board: {self._board_str()}")
        self._log(f"  Pot  : {self.pot_manager.total_pot()}")

        order = self._build_action_order(round_name)
        if not order:
            return

        # acted: player_ids who have voluntarily acted since the last bet
        # that re-opened the action.
        acted = set()
        raises_this_round = 0
        n = len(order)
        pos = 0
        guard = 0

        while any(self._needs_action(p, acted) for p in order):
            guard += 1
            if guard > MAX_ACTIONS_PER_ROUND:
                break

            player = order[pos % n]
            pos += 1
            if not self._needs_action(player, acted):
                continue

            call_amount = max(0, self.current_bet - player.current_bet)

            game_state = self._build_game_state(
                round_name, call_amount, actor=player
            )
            action, amount = player.decide(game_state)
            self._validate_action(player, action, amount, call_amount)

            prev_bet = self.current_bet
            prev_increment = self.last_raise_size
            actual = self._apply_action(player, action, amount, call_amount)

            acted.add(player.player_id)
            self._emit_action(round_name, player, action, actual)
            self._notify_action_observers(round_name, player, action, actual)

            # A bet that raised the level by a FULL increment re-opens action
            # and counts toward the raise cap. An incomplete all-in (increase
            # below a full raise) neither re-opens nor consumes a cap slot,
            # matching standard incomplete-raise rules.
            if self.current_bet > prev_bet:
                increase = self.current_bet - prev_bet
                if increase >= prev_increment:
                    raises_this_round += 1
                    if raises_this_round < MAX_RAISES_PER_ROUND:
                        acted = {player.player_id}

            if self._count_active() < 2:
                break

    def _needs_action(self, player, acted):
        """
        Whether a player still owes an action in the current betting round.

        True if the player is live (not folded / all-in / eliminated) AND
        either faces an unmatched bet or has not yet acted since the last
        re-opening bet.
        """
        if player.is_folded or player.is_all_in or not player.is_active:
            return False
        if player.current_bet < self.current_bet:
            return True
        return player.player_id not in acted

    def _build_action_order(self, round_name):
        """
        Build the ordered list of players who may act this round.

        Pre-flop: UTG (left of BB) through to BB.
        Post-flop: left of dealer through to dealer.

        The start seat is found by pure seat arithmetic (NOT stack > 0): an
        all-in player is still in the hand and anchors the order, so a street
        where everyone is all-in produces a valid (all-skippable) order rather
        than crashing. `_needs_action` then skips the all-in players, ending
        the round immediately and proceeding to the next street / showdown.

        Args:
            round_name (str): Betting round name.

        Returns:
            list[Player]: In-hand players in action order (may all be all-in).
        """
        anchor = self._bb_index if round_name == ROUND_PRE_FLOP else self.dealer_index

        n = len(self.players)
        start_index = None
        for offset in range(1, n + 1):
            idx = (anchor + offset) % n
            p = self.players[idx]
            if p.is_active and not p.is_folded:
                start_index = idx
                break
        if start_index is None:
            return []

        return self._build_action_order_from(start_index, round_name)

    def _build_action_order_from(self, start_index, round_name):
        """
        Build action order starting from start_index, cycling once.

        Args:
            start_index (int): Index into self.players to start from.
            round_name (str): Used only for logging context.

        Returns:
            list[Player]: All non-folded, non-eliminated players in order.
        """
        n = len(self.players)
        order = []
        for offset in range(n):
            idx = (start_index + offset) % n
            p = self.players[idx]
            if p.is_active and not p.is_folded:
                order.append(p)
        return order

    def _deal_community_cards(self, round_name):
        """
        Deal community cards for the given round.

        Args:
            round_name (str): Betting round name.
        """
        if round_name == ROUND_PRE_FLOP:
            return
        elif round_name == ROUND_FLOP:
            self.community_cards.extend(self.deck.deal_many(FLOP_CARD_COUNT))
        elif round_name == ROUND_TURN:
            self.community_cards.extend(self.deck.deal_many(TURN_CARD_COUNT))
        elif round_name == ROUND_RIVER:
            self.community_cards.extend(self.deck.deal_many(RIVER_CARD_COUNT))

    def _build_game_state(self, round_name, call_amount, actor=None):
        """
        Build the game state dict passed to player.decide().

        Adds opponent-observable state (Phase 0): `opponent_ids` lists the
        live opponents of the actor (still in the hand, excluding the actor),
        and `acting_player_id` names the actor. Belief-driven bots use these
        to know whom to model and which cards to exclude.

        Args:
            round_name (str): Current betting round.
            call_amount (int): Chips needed to call.
            actor (Player | None): The player about to act.

        Returns:
            dict: Table state snapshot.
        """
        min_raise = max(self.last_raise_size, self.big_blind) + self.current_bet

        if actor is not None:
            opponent_ids = [
                p.player_id for p in self.players
                if p is not actor and not p.is_folded and p.is_active
            ]
        else:
            opponent_ids = [
                p.player_id for p in self.players
                if not p.is_folded and p.is_active
            ]

        return {
            "round_name":         round_name,
            "pot":                self.pot_manager.total_pot(),
            "call_amount":        call_amount,
            "min_raise":          min_raise,
            "current_bet":        self.current_bet,
            "community_cards":    list(self.community_cards),
            "active_player_count": self._count_active(),
            "n_players":          len(self.players),
            "opponent_ids":       opponent_ids,
            "acting_player_id":   actor.player_id if actor is not None else None,
            "all_stacks":         {p.player_id: p.stack for p in self.players},
            "prize_structure":    self.prize_structure,
        }

    def _validate_action(self, player, action, amount, call_amount):
        """
        Validate that a player's chosen action is legal.

        Args:
            player (Player): The acting player.
            action (str): Proposed action string.
            amount (int): Proposed chip amount.
            call_amount (int): Current call amount.

        Raises:
            ValueError: If the action violates game rules.
        """
        if action not in (ACTION_FOLD, ACTION_CHECK, ACTION_CALL,
                          ACTION_RAISE, ACTION_ALL_IN):
            raise ValueError(
                f"Unknown action '{action}' from {player.name}."
            )
        if action == ACTION_CHECK and call_amount > 0:
            raise ValueError(
                f"{player.name} cannot check: "
                f"must call {call_amount} chips or fold."
            )
        if action == ACTION_RAISE:
            min_raise = self.last_raise_size + self.current_bet
            if amount < min_raise and amount < player.stack:
                raise ValueError(
                    f"{player.name} raise of {amount} is below "
                    f"minimum raise of {min_raise}."
                )

    def _apply_action(self, player, action, amount, call_amount):
        """
        Apply a player's action: update player state and pot.

        Args:
            player (Player): The acting player.
            action (str): Chosen action.
            amount (int): Bet/raise amount (ignored for fold/check).
            call_amount (int): Chips needed to call (for call action).

        Returns:
            int: Chips the player added to the pot this action (0 fold/check).
        """
        if action == ACTION_FOLD:
            player.is_folded = True
            self.pot_manager.mark_folded(player.player_id)
            self._log(f"  {player.name:14s} FOLDS")
            return 0

        elif action == ACTION_CHECK:
            self._log(f"  {player.name:14s} CHECKS")
            return 0

        elif action == ACTION_CALL:
            actual = player.place_bet(call_amount)
            self.pot_manager.add_contribution(player.player_id, actual)
            self._log(f"  {player.name:14s} CALLS    {actual}")
            return actual

        elif action == ACTION_RAISE:
            additional = amount - player.current_bet
            actual = player.place_bet(additional)
            self.pot_manager.add_contribution(player.player_id, actual)
            raise_size = amount - self.current_bet
            self.last_raise_size = max(raise_size, self.big_blind)
            self.current_bet = player.current_bet
            self._log(f"  {player.name:14s} RAISES   to {player.current_bet}")
            return actual

        elif action == ACTION_ALL_IN:
            actual = player.place_bet(player.stack)
            self.pot_manager.add_contribution(player.player_id, actual)
            if player.current_bet > self.current_bet:
                raise_size = player.current_bet - self.current_bet
                self.last_raise_size = max(raise_size, self.big_blind)
                self.current_bet = player.current_bet
            self._log(f"  {player.name:14s} ALL-IN   ({player.current_bet} total)")
            return actual

        return 0

    # ------------------------------------------------------------------
    # Showdown and distribution
    # ------------------------------------------------------------------

    def _showdown(self):
        """
        Evaluate hands and determine pot winners at showdown.

        Handles:
            - Uncontested pot (one non-folded player)
            - Multi-way showdown with side pots
            - Split pots (ties)

        Returns:
            dict: {player_id: chips_won}
        """
        contenders = [
            p for p in self.players
            if not p.is_folded and p.is_active and len(p.hole_cards) == 2
        ]

        if not contenders:
            # Fallback: all remaining active players (edge case guard)
            contenders = [p for p in self.players
                          if not p.is_folded and p.is_active]

        if len(contenders) == 1:
            winner = contenders[0]
            total = self.pot_manager.total_pot()
            self._log(f"\n  {winner.name} wins {total} chips uncontested.")
            return {winner.player_id: total}

        # Need 5 community cards for full showdown evaluation.
        # If we reach showdown before river (all-in scenarios), complete board.
        while len(self.community_cards) < 5:
            self.community_cards.extend(
                self.deck.deal_many(1)
            )

        self._log(f"\n  --- Showdown ---")
        self._log(f"  Board: {self._board_str()}")
        for p in contenders:
            label = self.evaluator.hand_label(
                self.evaluator.evaluate(p.hole_cards, self.community_cards)
            )
            self._log(
                f"  {p.name:14s} "
                f"[{str(p.hole_cards[0])} {str(p.hole_cards[1])}]  {label}"
            )

        self._emit_showdown(contenders)

        def resolver(eligible_ids):
            eligible = {
                p.player_id: p.hole_cards
                for p in contenders
                if p.player_id in eligible_ids and len(p.hole_cards) == 2
            }
            if not eligible:
                # All eligible players for this pot have no hole cards
                # (edge case: return any contender so pot is not orphaned)
                fallback = [p for p in contenders if len(p.hole_cards) == 2]
                if not fallback:
                    raise RuntimeError(
                        "No valid contenders with hole cards at showdown. "
                        "This indicates a critical game state error."
                    )
                eligible = {fallback[0].player_id: fallback[0].hole_cards}
            winners, _ = self.evaluator.best_hand_among(
                eligible, self.community_cards
            )
            # Order winners first-left-of-dealer so any indivisible odd chip
            # (awarded to winners[0] by PotManager) goes to the correct seat
            # per standard rules, not simply the lowest seat index.
            n = len(self.players)
            seat_of = {p.player_id: i for i, p in enumerate(self.players)}
            winners.sort(
                key=lambda pid: (seat_of[pid] - self.dealer_index - 1) % n
            )
            return winners

        return self.pot_manager.distribute(resolver)

    def _distribute_winnings(self, winnings):
        """
        Add won chips to player stacks and log results.

        Args:
            winnings (dict): {player_id: chips_won}
        """
        id_to_player = {p.player_id: p for p in self.players}
        for player_id, amount in winnings.items():
            if amount > 0:
                player = id_to_player[player_id]
                player.stack += amount
                self._log(f"  {player.name} wins {amount} chips.")

    # ------------------------------------------------------------------
    # Event emission (Phase 0)
    # ------------------------------------------------------------------

    def _emit(self, event):
        """Append an event to the log and notify every observer."""
        self.event_log.append(event)
        for observer in self.observers:
            observer(event)

    def _emit_street(self, round_name):
        """Emit a 'street' event at the start of a betting round."""
        self._emit(HandEvent(
            hand_id=self.hand_number,
            street=round_name,
            event_type=EVENT_STREET,
            pot=self.pot_manager.total_pot(),
            community_cards=board_codes(self.community_cards),
        ))

    def _emit_action(self, round_name, player, action, amount):
        """Emit an 'action' event after a player acts."""
        self._emit(HandEvent(
            hand_id=self.hand_number,
            street=round_name,
            event_type=EVENT_ACTION,
            player_id=player.player_id,
            action=action,
            amount=amount,
            pot=self.pot_manager.total_pot(),
            community_cards=board_codes(self.community_cards),
            equity=getattr(player, "last_equity", None),
        ))

    def _emit_showdown(self, contenders):
        """Emit a 'showdown' event listing the contenders and board."""
        self._emit(HandEvent(
            hand_id=self.hand_number,
            street=STREET_SHOWDOWN,
            event_type=EVENT_SHOWDOWN,
            pot=self.pot_manager.total_pot(),
            community_cards=board_codes(self.community_cards),
            payload={
                "contenders": [p.player_id for p in contenders],
                "hole_cards": {
                    p.player_id: board_codes(p.hole_cards)
                    for p in contenders
                },
            },
        ))

    def _emit_hand_end(self, winnings):
        """Emit a 'hand_end' event with the final pot distribution."""
        self._emit(HandEvent(
            hand_id=self.hand_number,
            street=STREET_HAND_END,
            event_type=EVENT_HAND_END,
            pot=sum(winnings.values()),
            community_cards=board_codes(self.community_cards),
            payload={"winnings": dict(winnings)},
        ))

    def _notify_action_observers(self, round_name, actor, action, amount):
        """
        Post-action callback: let every other player observe the action.

        Belief-modeling bots override Player.observe_action to update their
        posterior about the actor. The base implementation is a no-op, so
        this is backward compatible.
        """
        observation = {
            "hand_id":      self.hand_number,
            "street":       round_name,
            "actor_id":     actor.player_id,
            "action":       action,
            "amount":       amount,
            "pot":          self.pot_manager.total_pot(),
            "current_bet":  self.current_bet,
            "is_aggressive": action in (ACTION_RAISE, ACTION_ALL_IN),
        }
        for p in self.players:
            if p is not actor:
                p.observe_action(observation)

    def _notify_hand_result(self, start_stacks):
        """
        Post-hand callback: tell each player the realised per-hand chip delta of
        every OTHER player (this hand's PnL).

        A PnL-driven belief (the HMM tilt regime) transitions on this, mirroring
        an AdaptiveBotPlayer's own per-hand regime switch, so tilt is detected
        from a realised loss before the opponent's aggression changes. Players
        without such a belief see a no-op (Player.observe_hand_result), so this
        is backward compatible. It only reads stacks, so it is chip-neutral.
        """
        deltas = {p.player_id: p.stack - start_stacks.get(p.player_id, p.stack)
                  for p in self.players}
        for observer in self.players:
            for p in self.players:
                if p is not observer:
                    observer.observe_hand_result({
                        "hand_id":     self.hand_number,
                        "player_id":   p.player_id,
                        "delta_stack": deltas[p.player_id],
                    })

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _active_players(self):
        """Return all players with a non-zero stack (not eliminated)."""
        return [p for p in self.players if p.stack > 0]

    def _eligible_for_action(self):
        """Return players who can voluntarily act (not folded, not all-in)."""
        return [
            p for p in self.players
            if not p.is_folded and not p.is_all_in and p.is_active
        ]

    def _count_active(self):
        """Return number of players still contesting the pot (not folded)."""
        return sum(
            1 for p in self.players
            if not p.is_folded and p.is_active
        )

    def _next_active_from(self, seat_index, skip=True):
        """
        Return the next player with chips, starting after seat_index.

        Args:
            seat_index (int): Index into self.players to start searching from.
            skip (bool): If True, skip seat_index itself (start from +1).

        Returns:
            Player: The next active player.

        Raises:
            RuntimeError: If no active player is found.
        """
        n = len(self.players)
        start = (seat_index + 1) if skip else seat_index
        for offset in range(n):
            idx = (start + offset) % n
            if self.players[idx].stack > 0:
                return self.players[idx]
        raise RuntimeError(
            "No active player found. This indicates a game state error."
        )

    def _board_str(self):
        """Return a formatted string of the community cards."""
        if not self.community_cards:
            return "(none)"
        return "  ".join(str(c) for c in self.community_cards)

    def _log(self, message):
        """Print message to terminal if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _log_stacks(self):
        """Print chip counts for all players at end of hand."""
        self._log("\n  --- Chip Counts ---")
        for p in self.players:
            status = "" if p.stack > 0 else "  [eliminated]"
            self._log(f"  {p.name:14s}: {p.stack}{status}")
