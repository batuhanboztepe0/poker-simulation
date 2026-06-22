"""
leduc_q.py
----------
Tabular Q-learning self-play on Leduc Hold'em — the value-based, greedy,
NO-averaging learner that operates in the *same regime as DQN self-play*
(off-policy TD bootstrapping, an ε-greedy behaviour policy, a single shared
value table for both seats). It reuses the exact Leduc game logic in
`leduc_cfr` and produces a `{info_set: [p_fold, p_call, p_raise]}` greedy
(last-iterate) policy that `leduc_eval.exploitability_of` scores exactly.

The point (made directly, not by analogy): CFR's TIME-AVERAGE converges to the
Nash equilibrium, but the LAST-ITERATE of an independent greedy value learner —
this Q-learner, just like a DQN — does NOT. Each player greedily best-responds to
the other's *current* (non-stationary) policy, so the iterate stays exploitable.
NFSP (references.md §1) is exactly the fix: average the policy.

Pure-Python, seeded, and self-contained beyond `leduc_cfr`; an offline analysis
module that touches no engine, training, or baseline behaviour.
"""

import random

from src.leduc_cfr import (
    LeducCFR, FOLD, CALL, RAISE, NUM_ACTIONS,
    _available_actions, _is_terminal, _round_contrib, _showdown, _avail_from_key,
)

_DECK = [0, 0, 1, 1, 2, 2]            # two each of J=0, Q=1, K=2
_ACT_CHAR = {FOLD: "f", CALL: "c", RAISE: "r"}

_ALL_INFO_SETS = None


def _all_info_sets():
    """Every reachable Leduc info-set key (one cheap CFR enumeration, cached) so
    the greedy table can be filled at info-sets the greedy walk never reaches."""
    global _ALL_INFO_SETS
    if _ALL_INFO_SETS is None:
        cfr = LeducCFR()
        cfr.train(1)
        _ALL_INFO_SETS = sorted(cfr.nodes.keys())
    return _ALL_INFO_SETS


def _payoff0(c0, c1, board, r1, r2):
    """Net result for player 0 (zero-sum, so player 1 gets the negative) at a
    terminal Leduc history. Mirrors the payoff logic in `LeducCFR._cfr`."""
    in_r2 = r2 is not None
    curr = r2 if in_r2 else r1
    p0r1, p1r1 = _round_contrib(r1, 2)
    p0r2, p1r2 = _round_contrib(r2, 4) if in_r2 else (0, 0)
    p0_total = 1 + p0r1 + p0r2          # ante + round contributions
    p1_total = 1 + p1r1 + p1r2
    if curr[-1] == "f":
        folder = (len(curr) - 1) % 2
        return -p0_total if folder == 0 else p1_total
    res = _showdown(c0, c1, board)      # round-2 call -> showdown
    return p1_total if res == 1 else -p0_total if res == -1 else 0.0


class LeducQLearner:
    """Independent greedy value-based self-play (the DQN regime, tabular)."""

    def __init__(self, alpha=0.1, eps=0.1, gamma=1.0, seed=0):
        self.q = {}                     # info_set -> [q_fold, q_call, q_raise]
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.rng = random.Random(seed)

    def _row(self, info_set):
        row = self.q.get(info_set)
        if row is None:
            row = [0.0, 0.0, 0.0]
            self.q[info_set] = row
        return row

    def _eps_greedy(self, info_set, avail):
        if self.rng.random() < self.eps:
            return self.rng.choice(avail)
        row = self._row(info_set)
        best = max(row[a] for a in avail)
        return self.rng.choice([a for a in avail if row[a] == best])

    def _play(self, c0, c1, board):
        """Play one hand ε-greedily; return (decisions, payoff0) where decisions
        is an ordered list of (player, info_set, action)."""
        cards, r1, r2, decisions = (c0, c1), "", None, []
        while True:
            in_r2 = r2 is not None
            curr = r2 if in_r2 else r1
            if _is_terminal(curr, in_r2):
                if curr[-1] == "f" or in_r2:
                    break                       # fold or round-2 showdown ends it
                r2 = ""                          # round-1 betting closed -> round 2
                continue
            player = len(curr) % 2
            info_set = (str(cards[player]) + str(board) + r1 + "/" + r2
                        if in_r2 else str(cards[player]) + curr)
            a = self._eps_greedy(info_set, _available_actions(curr, in_r2))
            decisions.append((player, info_set, a))
            if in_r2:
                r2 = curr + _ACT_CHAR[a]
            else:
                r1 = curr + _ACT_CHAR[a]
        return decisions, _payoff0(c0, c1, board, r1, r2)

    def _update(self, decisions, payoff0):
        """Episodic TD(0) per player: reward 0 until terminal, bootstrap on the
        player's OWN next decision (max-Q over its available actions)."""
        for p in (0, 1):
            seq = [(s, a) for (pl, s, a) in decisions if pl == p]
            reward = payoff0 if p == 0 else -payoff0
            for i, (s, a) in enumerate(seq):
                if i + 1 < len(seq):
                    s2 = seq[i + 1][0]
                    row2, av2 = self._row(s2), _avail_from_key(seq[i + 1][0])
                    target = self.gamma * max(row2[x] for x in av2)
                else:
                    target = reward
                row = self._row(s)
                row[a] += self.alpha * (target - row[a])

    def train(self, episodes):
        for _ in range(episodes):
            i, j, k = self.rng.sample(range(6), 3)
            self._play_and_update(_DECK[i], _DECK[j], _DECK[k])

    def _play_and_update(self, c0, c1, board):
        decisions, payoff0 = self._play(c0, c1, board)
        self._update(decisions, payoff0)

    def greedy_strategy_table(self):
        """The LAST-ITERATE greedy policy over EVERY Leduc info-set (uniform at
        info-sets that are unvisited or still flat), as a
        {info_set: [p_fold, p_call, p_raise]} table for `exploitability_of`."""
        table = {}
        for s in _all_info_sets():
            av = _avail_from_key(s)
            row = self.q.get(s)
            probs = [0.0] * NUM_ACTIONS
            if row is None or max(row[a] for a in av) == min(row[a] for a in av):
                for a in av:
                    probs[a] = 1.0 / len(av)
            else:
                best = max(row[a] for a in av)
                winners = [a for a in av if row[a] == best]
                for a in winners:
                    probs[a] = 1.0 / len(winners)
            table[s] = probs
        return table
