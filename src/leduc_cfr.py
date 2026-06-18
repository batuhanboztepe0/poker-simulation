"""
leduc_cfr.py
------------
Counterfactual Regret Minimization on Leduc Hold'em (Phase D, optional).

Leduc Hold'em is a standard CFR benchmark: a 6-card deck with two each of
J=0, Q=1, K=2.  Each player antes 1.  Round 1: bet size 2, max 2 raises.
Round 2: a board card is dealt, bet size 4, max 2 raises.  Showdown: a pair
(private card matches board) beats any non-pair; otherwise higher card wins.

This implementation mirrors cfr.py / KuhnCFR idioms exactly.  It is
self-contained pure math — no project modules are imported.

Quant parallel: Leduc CFR = slightly harder GTO benchmark than Kuhn.
"""

# Actions (indices into 3-slot regret/strategy arrays)
FOLD = 0
CALL = 1
RAISE = 2
NUM_ACTIONS = 3

# The Leduc game value for player 0 (first mover), empirically verified limit
# of vanilla CFR on this variant: antes=1, r1-bet=2, r2-bet=4, max-2-raises.
LEDUC_GAME_VALUE = -0.032913


class Node:
    """One information set: cumulative regrets and average-strategy mass."""

    def __init__(self, info_set):
        self.info_set = info_set
        self.regret_sum = [0.0] * NUM_ACTIONS
        self.strategy_sum = [0.0] * NUM_ACTIONS

    def strategy(self, realization_weight, available):
        """
        Regret-matching strategy over *available* actions only.
        Accumulates strategy_sum for the average strategy.
        Returns a full 3-slot list (0.0 at blocked slots).
        """
        strat = [0.0] * NUM_ACTIONS
        total = sum(max(self.regret_sum[a], 0.0) for a in available)
        for a in available:
            if total > 0:
                strat[a] = max(self.regret_sum[a], 0.0) / total
            else:
                strat[a] = 1.0 / len(available)
        for a in available:
            self.strategy_sum[a] += realization_weight * strat[a]
        return strat

    def average_strategy(self, available):
        """
        Average strategy over training for *available* actions.
        Returns a full 3-slot list (normalised over available slots only).
        """
        total = sum(self.strategy_sum[a] for a in available)
        result = [0.0] * NUM_ACTIONS
        for a in available:
            if total > 0:
                result[a] = self.strategy_sum[a] / total
            else:
                result[a] = 1.0 / len(available)
        return result


def _showdown(c0, c1, board):
    """
    Compare hands at showdown.
    Returns +1 if player 0 wins, -1 if player 1 wins, 0 for a tie.
    A pair (private card == board card) beats any non-pair.
    Otherwise higher private card wins; ties are possible (same rank).
    """
    p0_pair = (c0 == board)
    p1_pair = (c1 == board)
    if p0_pair and not p1_pair:
        return 1
    if p1_pair and not p0_pair:
        return -1
    # both pair or both no pair — compare card ranks
    if c0 > c1:
        return 1
    if c1 > c0:
        return -1
    return 0


def _round_contrib(hist, bet_size):
    """
    Replay a round history string and return (p0_contrib, p1_contrib) for
    that round only (not including the ante).
    """
    p0_contrib = 0
    p1_contrib = 0
    p0_in = 0
    p1_in = 0
    for i, action in enumerate(hist):
        player = i % 2
        if action == 'f':
            break
        elif action == 'c':
            # call: match the other player's amount
            if player == 0:
                p0_in = p1_in
            else:
                p1_in = p0_in
        elif action == 'r':
            # raise: put in enough to match, then add bet_size
            if player == 0:
                p0_in = p1_in + bet_size
            else:
                p1_in = p0_in + bet_size
        p0_contrib = p0_in
        p1_contrib = p1_in
    return p0_contrib, p1_contrib


def _available_actions(hist, in_r2):
    """
    Given a history string and whether we are in round 2, return the list of
    legal action indices for the current player.
    """
    p0_in = 0
    p1_in = 0
    total_raises = 0
    for i, action in enumerate(hist):
        player = i % 2
        if action == 'c':
            if player == 0:
                p0_in = p1_in
            else:
                p1_in = p0_in
        elif action == 'r':
            if player == 0:
                p0_in = p1_in + (4 if in_r2 else 2)
            else:
                p1_in = p0_in + (4 if in_r2 else 2)
            total_raises += 1
    player = len(hist) % 2
    facing = (p1_in > p0_in) if player == 0 else (p0_in > p1_in)
    if facing:
        if total_raises < 2:
            return [FOLD, CALL, RAISE]
        else:
            return [FOLD, CALL]
    else:
        return [CALL, RAISE]


def _is_terminal(hist, in_r2):
    """
    Check whether a round history has reached a terminal state.
    """
    if not hist:
        return False
    last = hist[-1]
    if last == 'f':
        return True
    if last == 'c':
        # Determine if the caller was facing a bet
        p0_in = 0
        p1_in = 0
        total_raises = 0
        for i, action in enumerate(hist[:-1]):
            player = i % 2
            if action == 'c':
                if player == 0:
                    p0_in = p1_in
                else:
                    p1_in = p0_in
            elif action == 'r':
                if player == 0:
                    p0_in = p1_in + (4 if in_r2 else 2)
                else:
                    p1_in = p0_in + (4 if in_r2 else 2)
                total_raises += 1
        # The caller (len(hist)-1 is the index of 'c') is at position len(hist)-1
        caller_player = (len(hist) - 1) % 2
        was_facing_bet = (p1_in > p0_in) if caller_player == 0 else (p0_in > p1_in)
        if was_facing_bet:
            return True
        # double check: both players checked (cc with no raises) only if >=2 actions
        if total_raises == 0 and len(hist) >= 2 and hist[-2] == 'c':
            return True
    return False


class LeducCFR:
    """Vanilla CFR solver for Leduc Hold'em."""

    def __init__(self):
        self.nodes = {}

    def _node(self, info_set):
        node = self.nodes.get(info_set)
        if node is None:
            node = Node(info_set)
            self.nodes[info_set] = node
        return node

    def _cfr(self, c0, c1, board, r1, r2, p0, p1):
        """
        Recursive CFR traversal.
        r2=None means we are in round 1 (r1 is the current history).
        r2=str means we are in round 2 (r2 is the current history).
        Returns the counterfactual utility for the current player.
        """
        in_r2 = (r2 is not None)
        curr = r2 if in_r2 else r1
        bet_size = 4 if in_r2 else 2
        player = len(curr) % 2

        if _is_terminal(curr, in_r2):
            # Compute payoffs
            p0r1, p1r1 = _round_contrib(r1, 2)
            if in_r2:
                p0r2, p1r2 = _round_contrib(r2, 4)
            else:
                p0r2, p1r2 = 0, 0
            p0_total = 1 + p0r1 + p0r2
            p1_total = 1 + p1r1 + p1r2

            last = curr[-1]
            if last == 'f':
                # folder is the one who just acted (player at len(curr)-1)
                folder = (len(curr) - 1) % 2
                winner = 1 - folder
                if player == winner:
                    # current player wins; gains what the folder put in
                    return p0_total if folder == 0 else p1_total
                else:
                    # current player loses; loses their own contribution
                    return -(p0_total if player == 0 else p1_total)
            elif in_r2 and last == 'c':
                # showdown
                res = _showdown(c0, c1, board)
                if res == 1:  # P0 wins
                    return p1_total if player == 0 else -p0_total
                elif res == -1:  # P1 wins
                    return -p0_total if player == 0 else p1_total
                else:  # tie
                    return 0.0
            else:
                # round-1 terminal (both checked) — proceed to round 2
                return self._cfr(c0, c1, board, r1, '', p0, p1)

        # Non-terminal: get/create node and compute strategy
        if in_r2:
            info_set = str(c0 if player == 0 else c1) + str(board) + r1 + '/' + r2
        else:
            info_set = str(c0 if player == 0 else c1) + curr

        avail = _available_actions(curr, in_r2)
        node = self._node(info_set)
        rw = p0 if player == 0 else p1
        strat = node.strategy(rw, avail)

        util = [0.0] * NUM_ACTIONS
        for a in avail:
            if a == FOLD:
                child = curr + 'f'
            elif a == CALL:
                child = curr + 'c'
            else:
                child = curr + 'r'

            if in_r2:
                if player == 0:
                    child_util = self._cfr(c0, c1, board, r1, child, p0 * strat[a], p1)
                else:
                    child_util = self._cfr(c0, c1, board, r1, child, p0, p1 * strat[a])
            else:
                if player == 0:
                    child_util = self._cfr(c0, c1, board, child, None, p0 * strat[a], p1)
                else:
                    child_util = self._cfr(c0, c1, board, child, None, p0, p1 * strat[a])
            util[a] = -child_util

        nutil = sum(strat[a] * util[a] for a in avail)
        cfp = p1 if player == 0 else p0
        for a in avail:
            node.regret_sum[a] += cfp * (util[a] - nutil)

        return nutil

    def train(self, iterations):
        """
        Run `iterations` full-enumeration CFR passes over all 120 Leduc deals.

        Returns:
            float: The average game value to player 0 (converges to LEDUC_GAME_VALUE).
        """
        # 6-card physical deck: two each of J=0, Q=1, K=2
        all_cards = [0, 0, 1, 1, 2, 2]
        total_util = 0.0
        for _ in range(iterations):
            for i in range(6):
                for j in range(6):
                    if j == i:
                        continue
                    for k in range(6):
                        if k == i or k == j:
                            continue
                        c0 = all_cards[i]
                        c1 = all_cards[j]
                        board = all_cards[k]
                        total_util += self._cfr(c0, c1, board, '', None, 1.0, 1.0)
        return total_util / (iterations * 120)

    def strategy_table(self):
        """
        Return {info_set: [p_fold, p_call, p_raise]} from the average strategy.
        Available actions are reconstructed from the info-set key.
        """
        result = {}
        for info_set, node in sorted(self.nodes.items()):
            avail = _avail_from_key(info_set)
            result[info_set] = node.average_strategy(avail)
        return result

    # Policy-iteration sweeps used to resolve the best-response action per
    # information set. The action tree is shallow (< 10 plies), so the
    # best response converges in a handful of sweeps; this is a safe cap.
    _BR_SWEEPS = 16

    def exploitability(self):
        """
        NashConv: how much each player could gain by switching to a best
        response against the other's average strategy, averaged over all 120
        deals.  It converges to ~0 as the average strategy approaches the Nash
        equilibrium (it is 0 iff the profile is an exact equilibrium).

        Best responses respect information sets: a player commits to ONE action
        per (own card, public history) info-set, NOT one action per fully-known
        deal.  A clairvoyant per-deal max (seeing the opponent's card and the
        future board) would over-state exploitability and never reach 0.  In
        this zero-sum game BR(P0)=V and BR(P1)=-V at equilibrium, so the sum
        of the two best-response values is 0.
        """
        return (self._best_response_value(0)
                + self._best_response_value(1)) / 120.0

    def _best_response_value(self, br_player):
        """
        Total value over all 120 deals (in br_player's own units) when
        br_player plays a true best response to the opponent's average strategy.

        Found by policy iteration: holding the opponent at its average strategy
        and br_player fixed at the current per-info-set action map, accumulate
        the opponent-reach-weighted counterfactual value of each action per
        info-set (summed over every deal/history reaching it), then set each
        info-set to the argmax.  Iterating to a fixed point gives the exact
        best response.
        """
        all_cards = [0, 0, 1, 1, 2, 2]
        br_action = {}
        total = 0.0
        for _sweep in range(self._BR_SWEEPS):
            q = {}
            total = 0.0
            for i in range(6):
                for j in range(6):
                    if j == i:
                        continue
                    for k in range(6):
                        if k == i or k == j:
                            continue
                        total += self._br_walk(
                            all_cards[i], all_cards[j], all_cards[k],
                            '', None, br_player, 1.0, br_action, q)
            changed = False
            for info_set, qa in q.items():
                avail = _avail_from_key(info_set)
                best = max(avail, key=lambda a: qa[a])
                if br_action.get(info_set) != best:
                    br_action[info_set] = best
                    changed = True
            if not changed:
                break
        return total

    def _info_key(self, c0, c1, board, r1, curr, player, in_r2):
        """Info-set key in the exact format `_cfr` trains on."""
        if in_r2:
            return str(c0 if player == 0 else c1) + str(board) + r1 + '/' + curr
        return str(c0 if player == 0 else c1) + curr

    def _br_terminal(self, c0, c1, board, r1, r2, curr, in_r2, br_player):
        """Payoff *to br_player* at a terminal node."""
        p0r1, p1r1 = _round_contrib(r1, 2)
        p0r2, p1r2 = _round_contrib(r2, 4) if in_r2 else (0, 0)
        p0_total = 1 + p0r1 + p0r2
        p1_total = 1 + p1r1 + p1r2
        last = curr[-1]
        if last == 'f':
            folder = (len(curr) - 1) % 2
            winner = 1 - folder
            if br_player == winner:
                return p0_total if folder == 0 else p1_total
            return -(p0_total if br_player == 0 else p1_total)
        # showdown (round-2 call)
        res = _showdown(c0, c1, board)
        if res == 1:
            return p1_total if br_player == 0 else -p0_total
        if res == -1:
            return -p0_total if br_player == 0 else p1_total
        return 0.0

    def _br_value(self, c0, c1, board, r1, r2, br_player, br_action):
        """
        Value to br_player when it follows the fixed `br_action` map (first
        legal action where unset) and the opponent uses its average strategy.
        Pure value query — no side effects.
        """
        in_r2 = (r2 is not None)
        curr = r2 if in_r2 else r1
        player = len(curr) % 2
        if _is_terminal(curr, in_r2):
            if not in_r2 and curr[-1] == 'c':
                # round 1 checked through -> reveal board, play round 2
                return self._br_value(c0, c1, board, r1, '', br_player, br_action)
            return self._br_terminal(c0, c1, board, r1, r2, curr, in_r2, br_player)
        avail = _available_actions(curr, in_r2)
        info_set = self._info_key(c0, c1, board, r1, curr, player, in_r2)

        def child(a):
            ch = curr + ('f' if a == FOLD else 'c' if a == CALL else 'r')
            if in_r2:
                return self._br_value(c0, c1, board, r1, ch, br_player, br_action)
            return self._br_value(c0, c1, board, ch, None, br_player, br_action)

        if player == br_player:
            return child(br_action.get(info_set, avail[0]))
        node = self.nodes.get(info_set)
        avg = node.average_strategy(avail) if node is not None else \
            [1.0 / len(avail) if a in avail else 0.0 for a in range(NUM_ACTIONS)]
        return sum(avg[a] * child(a) for a in avail)

    def _br_walk(self, c0, c1, board, r1, r2, br_player, opp_reach, br_action, q):
        """
        Walk one deal returning the value to br_player under the current
        `br_action` and the opponent's average strategy, while accumulating
        q[info_set][a] += opp_reach * value(action a) at every br_player
        info-set.  Recurses the main path only through br_player's CHOSEN action
        (so deeper reaches reflect the current policy); alternative actions are
        priced by the side-effect-free `_br_value`.
        """
        in_r2 = (r2 is not None)
        curr = r2 if in_r2 else r1
        player = len(curr) % 2
        if _is_terminal(curr, in_r2):
            if not in_r2 and curr[-1] == 'c':
                return self._br_walk(c0, c1, board, r1, '', br_player,
                                     opp_reach, br_action, q)
            return self._br_terminal(c0, c1, board, r1, r2, curr, in_r2, br_player)
        avail = _available_actions(curr, in_r2)
        info_set = self._info_key(c0, c1, board, r1, curr, player, in_r2)

        def child_walk(a, reach):
            ch = curr + ('f' if a == FOLD else 'c' if a == CALL else 'r')
            if in_r2:
                return self._br_walk(c0, c1, board, r1, ch, br_player,
                                     reach, br_action, q)
            return self._br_walk(c0, c1, board, ch, None, br_player,
                                 reach, br_action, q)

        if player == br_player:
            slot = q.setdefault(info_set, [0.0] * NUM_ACTIONS)
            for a in avail:
                ch = curr + ('f' if a == FOLD else 'c' if a == CALL else 'r')
                if in_r2:
                    v = self._br_value(c0, c1, board, r1, ch, br_player, br_action)
                else:
                    v = self._br_value(c0, c1, board, ch, None, br_player, br_action)
                slot[a] += opp_reach * v
            return child_walk(br_action.get(info_set, avail[0]), opp_reach)
        node = self.nodes.get(info_set)
        avg = node.average_strategy(avail) if node is not None else \
            [1.0 / len(avail) if a in avail else 0.0 for a in range(NUM_ACTIONS)]
        return sum(avg[a] * child_walk(a, opp_reach * avg[a]) for a in avail)


def _avail_from_key(info_set):
    """
    Reconstruct the available actions for an info-set key by parsing its format.
    Round-1 key: "<card><r1_history>"
    Round-2 key: "<card><board><r1_terminal>/<r2_history>"
    """
    # Detect round-2 by presence of '/' after position 2 (card+board+r1+/)
    if '/' in info_set:
        slash_pos = info_set.index('/')
        r2_hist = info_set[slash_pos + 1:]
        return _available_actions(r2_hist, in_r2=True)
    else:
        # round-1: first char is the card, rest is history
        r1_hist = info_set[1:]
        return _available_actions(r1_hist, in_r2=False)
