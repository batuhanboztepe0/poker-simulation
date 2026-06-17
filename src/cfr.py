"""
cfr.py
------
Counterfactual Regret Minimization on Kuhn poker (Phase D, optional).

Kuhn poker is the canonical CFR teaching game: a 3-card deck {J=1, Q=2, K=3},
each player antes 1 and is dealt one card, then a single check/bet round. Its
Nash equilibrium is known in closed form — player 1's game value is exactly
-1/18 — so it is the standard unit-test target for a correct CFR solver.

This is the project's "market-neutral / unexploitable benchmark" counterpart to
the exploitative agents of Phases A-C. Vanilla CFR with full chance enumeration
(all 6 card deals per iteration) converges deterministically.

Quant parallel: CFR / GTO = Nash equilibrium / risk-neutral unexploitable price.
"""

import itertools

PASS = 0
BET = 1
NUM_ACTIONS = 2

# The Kuhn-poker Nash game value for the first player.
KUHN_GAME_VALUE = -1.0 / 18.0


class Node:
    """One information set: cumulative regrets and average-strategy mass."""

    def __init__(self, info_set):
        self.info_set = info_set
        self.regret_sum = [0.0] * NUM_ACTIONS
        self.strategy_sum = [0.0] * NUM_ACTIONS

    def strategy(self, realization_weight):
        """Current regret-matching strategy; accumulate the average."""
        strat = [max(r, 0.0) for r in self.regret_sum]
        total = sum(strat)
        if total > 0:
            strat = [s / total for s in strat]
        else:
            strat = [1.0 / NUM_ACTIONS] * NUM_ACTIONS
        for a in range(NUM_ACTIONS):
            self.strategy_sum[a] += realization_weight * strat[a]
        return strat

    def average_strategy(self):
        """Average strategy over training — this converges to equilibrium."""
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        return [1.0 / NUM_ACTIONS] * NUM_ACTIONS


class KuhnCFR:
    """Vanilla CFR solver for Kuhn poker."""

    def __init__(self):
        self.nodes = {}

    def _node(self, info_set):
        node = self.nodes.get(info_set)
        if node is None:
            node = Node(info_set)
            self.nodes[info_set] = node
        return node

    def _cfr(self, cards, history, p0, p1):
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        if plays >= 2:
            terminal_pass = history[-1] == "p"
            double_bet = history[-2:] == "bb"
            player_wins = cards[player] > cards[opponent]
            if terminal_pass:
                if history == "pp":
                    return 1 if player_wins else -1
                # opponent bet, current player passed (folded)
                return 1
            elif double_bet:
                return 2 if player_wins else -2

        info_set = str(cards[player]) + history
        node = self._node(info_set)
        strategy = node.strategy(p0 if player == 0 else p1)

        util = [0.0] * NUM_ACTIONS
        node_util = 0.0
        for a in range(NUM_ACTIONS):
            next_history = history + ("p" if a == PASS else "b")
            if player == 0:
                util[a] = -self._cfr(cards, next_history, p0 * strategy[a], p1)
            else:
                util[a] = -self._cfr(cards, next_history, p0, p1 * strategy[a])
            node_util += strategy[a] * util[a]

        counterfactual_weight = p1 if player == 0 else p0
        for a in range(NUM_ACTIONS):
            regret = util[a] - node_util
            node.regret_sum[a] += counterfactual_weight * regret

        return node_util

    def train(self, iterations):
        """
        Run `iterations` full-enumeration CFR passes.

        Returns:
            float: The average game value to player 1 (converges to -1/18).
        """
        deals = list(itertools.permutations([1, 2, 3]))
        total_util = 0.0
        for _ in range(iterations):
            for cards in deals:
                total_util += self._cfr(list(cards), "", 1.0, 1.0)
        return total_util / (iterations * len(deals))

    def strategy_table(self):
        """{info_set: [P(pass), P(bet)]} from the average strategy."""
        return {
            info_set: node.average_strategy()
            for info_set, node in sorted(self.nodes.items())
        }

    def bet_probability(self, info_set):
        """Average P(bet) at an information set (0.0 if unseen)."""
        node = self.nodes.get(info_set)
        return node.average_strategy()[BET] if node else 0.0
