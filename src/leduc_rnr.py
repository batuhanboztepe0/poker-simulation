"""
leduc_rnr.py
------------
Restricted Nash Response (RNR) on Leduc Hold'em, measured EXACTLY (v2 Phase C3).

RNR (Johanson, Zinkevich & Bowling, NIPS 2007; Johanson & Bowling, AISTATS 2009)
is the validated way to exploit a known suboptimal opponent without the brittleness
of a raw best response. The opponent is RESTRICTED to play a fixed strategy
`sigma_fix` with probability `p` and a free (regret-minimised) strategy with
probability `1 - p`; our player regret-minimises against that p-restricted opponent.

    p = 0  ->  our counter is the Nash strategy (unexploitable, does not exploit)
    p = 1  ->  our counter is the exact best response to sigma_fix (max exploitation,
               maximally exploitable)
    0<p<1  ->  the exploitation-vs-exploitability frontier in between.

Everything here is EXACT (full 120-deal enumeration, exact best response), so the
EV gain over Nash against an exploitable opponent and the counter's own
exploitability are computed with ZERO sampling variance. This is the exact-Leduc
counterpart to the high-variance heads-up exploitation attempt (PREREGISTRATION §13),
and reuses the verified CFR / best-response machinery in `leduc_cfr`.

Quant parallel: RNR's p is the exploit-vs-robustness knob, the same tradeoff as
sizing a position to a counterparty model you are only partly confident in.
"""

from src.leduc_cfr import (
    LeducCFR, Node, NUM_ACTIONS, FOLD, CALL, RAISE,
    _is_terminal, _available_actions, _round_contrib, _showdown, _avail_from_key,
    LEDUC_GAME_VALUE,
)

_DECK = [0, 0, 1, 1, 2, 2]   # two each of J=0, Q=1, K=2


def _deals():
    """Yield every distinct (c0, c1, board) physical deal (120 of them)."""
    for i in range(6):
        for j in range(6):
            if j == i:
                continue
            for k in range(6):
                if k == i or k == j:
                    continue
                yield _DECK[i], _DECK[j], _DECK[k]


def _info_key(c0, c1, board, r1, curr, player, in_r2):
    """Info-set key in the exact format `leduc_cfr` trains on."""
    if in_r2:
        return str(c0 if player == 0 else c1) + str(board) + r1 + '/' + curr
    return str(c0 if player == 0 else c1) + curr


def _terminal_value0(c0, c1, board, r1, r2, curr, in_r2):
    """Payoff to PLAYER 0 at a terminal node (mirrors leduc_cfr._br_terminal with
    br_player=0)."""
    p0r1, p1r1 = _round_contrib(r1, 2)
    p0r2, p1r2 = _round_contrib(r2, 4) if in_r2 else (0, 0)
    p0_total = 1 + p0r1 + p0r2
    p1_total = 1 + p1r1 + p1r2
    last = curr[-1]
    if last == 'f':
        folder = (len(curr) - 1) % 2
        # player 0 wins iff player 1 folded
        return p1_total if folder == 1 else -p0_total
    # showdown (round-2 call)
    res = _showdown(c0, c1, board)
    if res == 1:
        return p1_total       # P0 wins what P1 put in
    if res == -1:
        return -p0_total      # P0 loses its own contribution
    return 0.0


# ---------------------------------------------------------------------------
# Exact value of two fixed strategies played against each other.
# A strategy is a callable (info_set, avail) -> length-NUM_ACTIONS distribution.
# ---------------------------------------------------------------------------

def _value0(c0, c1, board, r1, r2, s0, s1):
    """Exact EV to player 0 on one deal when player 0 plays s0 and player 1 plays
    s1 (both distributions over available actions)."""
    in_r2 = (r2 is not None)
    curr = r2 if in_r2 else r1
    player = len(curr) % 2
    if _is_terminal(curr, in_r2):
        if not in_r2 and curr[-1] == 'c':
            # round-1 betting closed by a (check/call) -> reveal board, play round 2
            return _value0(c0, c1, board, r1, '', s0, s1)
        return _terminal_value0(c0, c1, board, r1, r2, curr, in_r2)
    avail = _available_actions(curr, in_r2)
    info_set = _info_key(c0, c1, board, r1, curr, player, in_r2)
    dist = (s0 if player == 0 else s1)(info_set, avail)
    total = 0.0
    for a in avail:
        ch = curr + ('f' if a == FOLD else 'c' if a == CALL else 'r')
        nxt = (c0, c1, board, r1, ch) if in_r2 else (c0, c1, board, ch, None)
        if dist[a] != 0.0:
            total += dist[a] * _value0(*nxt, s0, s1)
    return total


def ev_player0(s0, s1):
    """Exact mean EV to player 0 over all 120 deals (s0 vs s1)."""
    return sum(_value0(c0, c1, board, '', None, s0, s1)
               for c0, c1, board in _deals()) / 120.0


# ---------------------------------------------------------------------------
# Fixed exploitable opponents (suboptimal strategies; each is exploitable).
# ---------------------------------------------------------------------------

def station(info_set, avail):
    """Calling station: always check/call, never fold to a bet, never raise."""
    d = [0.0] * NUM_ACTIONS
    d[CALL] = 1.0                      # CALL is legal at every non-terminal node
    return d


def maniac(info_set, avail):
    """Maniac: raise whenever legal, otherwise call (loose-aggressive)."""
    d = [0.0] * NUM_ACTIONS
    d[RAISE if RAISE in avail else CALL] = 1.0
    return d


def uniform(info_set, avail):
    """Uniform random over the legal actions."""
    d = [0.0] * NUM_ACTIONS
    for a in avail:
        d[a] = 1.0 / len(avail)
    return d


OPPONENTS = {"station": station, "maniac": maniac, "uniform": uniform}


# ---------------------------------------------------------------------------
# RNR solver: CFR with a p-restricted opponent.
# ---------------------------------------------------------------------------

class LeducRNR(LeducCFR):
    """CFR where the `restricted` player plays `p*sigma_fix + (1-p)*free` and our
    player regret-minimises against it. Our average strategy is RNR(p)."""

    def __init__(self, sigma_fix, p, restricted=1):
        super().__init__()
        self.sigma_fix = sigma_fix
        self.p = float(p)
        self.restricted = restricted

    def _cfr(self, c0, c1, board, r1, r2, p0, p1):
        in_r2 = (r2 is not None)
        curr = r2 if in_r2 else r1
        player = len(curr) % 2

        if _is_terminal(curr, in_r2):
            # Identical terminal handling to the base solver (alternating sign,
            # value to the CURRENT player); p0/p1 are threaded into the round-2
            # subgame so the reach weighting matches base CFR (required for p=0 to
            # reproduce Nash).
            return self._terminal_cfr(c0, c1, board, r1, r2, curr, in_r2,
                                      player, p0, p1)

        info_set = _info_key(c0, c1, board, r1, curr, player, in_r2)
        avail = _available_actions(curr, in_r2)
        node = self._node(info_set)
        rw = p0 if player == 0 else p1
        free = node.strategy(rw, avail)               # regret-matching (free) part
        if player == self.restricted and self.p > 0.0:
            fix = self.sigma_fix(info_set, avail)
            eff = [self.p * fix[a] + (1.0 - self.p) * free[a]
                   for a in range(NUM_ACTIONS)]
        else:
            eff = free

        util = [0.0] * NUM_ACTIONS
        for a in avail:
            child = curr + ('f' if a == FOLD else 'c' if a == CALL else 'r')
            if in_r2:
                if player == 0:
                    cu = self._cfr(c0, c1, board, r1, child, p0 * eff[a], p1)
                else:
                    cu = self._cfr(c0, c1, board, r1, child, p0, p1 * eff[a])
            else:
                if player == 0:
                    cu = self._cfr(c0, c1, board, child, None, p0 * eff[a], p1)
                else:
                    cu = self._cfr(c0, c1, board, child, None, p0, p1 * eff[a])
            util[a] = -cu

        nutil = sum(eff[a] * util[a] for a in avail)
        cfp = p1 if player == 0 else p0
        for a in avail:
            node.regret_sum[a] += cfp * (util[a] - nutil)
        return nutil

    def _terminal_cfr(self, c0, c1, board, r1, r2, curr, in_r2, player, p0, p1):
        """Terminal value to the CURRENT player, copied from leduc_cfr._cfr so the
        round-1->round-2 transition and the alternating sign are identical."""
        p0r1, p1r1 = _round_contrib(r1, 2)
        p0r2, p1r2 = _round_contrib(r2, 4) if in_r2 else (0, 0)
        p0_total = 1 + p0r1 + p0r2
        p1_total = 1 + p1r1 + p1r2
        last = curr[-1]
        if last == 'f':
            folder = (len(curr) - 1) % 2
            winner = 1 - folder
            if player == winner:
                return p0_total if folder == 0 else p1_total
            return -(p0_total if player == 0 else p1_total)
        if in_r2 and last == 'c':
            res = _showdown(c0, c1, board)
            if res == 1:
                return p1_total if player == 0 else -p0_total
            if res == -1:
                return -p0_total if player == 0 else p1_total
            return 0.0
        # round-1 betting closed without a fold -> play round 2 (value in P0 units),
        # threading the current reach p0/p1 exactly as base CFR does.
        v2 = self._cfr(c0, c1, board, r1, '', p0, p1)
        return v2 if player == 0 else -v2

    def counter_fn(self):
        """Our (player 0) average strategy as a callable (info_set, avail) -> dist.
        Falls back to uniform on an unseen info-set."""
        def f(info_set, avail):
            node = self.nodes.get(info_set)
            if node is None:
                return [1.0 / len(avail) if a in avail else 0.0
                        for a in range(NUM_ACTIONS)]
            return node.average_strategy(avail)
        return f

    def counter_exploitability(self):
        """Exploitability of OUR counter (player 0): how much an adversary best
        responding to it gains over the game value. _best_response_value(1) prices
        player 1's best response to player 0's average (= our counter); at p=0 this
        is ~0, and it rises with p (the exploitation-vs-exploitability tradeoff)."""
        return self._best_response_value(1) / 120.0 + LEDUC_GAME_VALUE


def nash_strategy(iters=4000):
    """A Nash strategy (the converged CFR average) as a callable counter, plus the
    solver (so its EV/exploitability can be queried)."""
    cfr = LeducCFR()
    cfr.train(iters)

    def f(info_set, avail):
        node = cfr.nodes.get(info_set)
        if node is None:
            return [1.0 / len(avail) if a in avail else 0.0
                    for a in range(NUM_ACTIONS)]
        return node.average_strategy(avail)
    return f, cfr
