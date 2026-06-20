"""
leduc_eval.py
-------------
Exact exploitability of an ARBITRARY Leduc Hold'em strategy, and the
average-vs-last-iterate comparison that explains why equilibrium self-play
converges to Nash while greedy best-response self-play does not (references.md
§1–§2).

`exploitability_of(strategy_table)` reuses the verified best-response machinery in
`leduc_cfr.LeducCFR` (which already computes NashConv against its own internal
average strategy) by swapping in fixed-strategy nodes — so it scores any
`{info_set: [p_fold, p_call, p_raise]}` policy, not just CFR's.

The rigorous point for the write-up: CFR's TIME-AVERAGE strategy drives
exploitability toward 0 (it approaches the Nash equilibrium), but the
LAST-ITERATE regret-matching strategy — the regime a DQN self-play agent operates
in — stays exploitable and oscillates. Neural Fictitious Self-Play (NFSP) is the
method that carries this averaging to large games via a neural average-policy
network; this metric is what one would evaluate it with.
"""

from src.leduc_cfr import LeducCFR, _avail_from_key, NUM_ACTIONS


class _FixedNode:
    """A node whose `average_strategy` returns a fixed distribution (renormalised
    over the available actions), mimicking `leduc_cfr.Node`'s interface so the
    best-response walk can price a best response against this fixed policy."""

    def __init__(self, dist):
        self._dist = list(dist)

    def average_strategy(self, available):
        total = sum(self._dist[a] for a in available)
        if total <= 0:
            return [1.0 / len(available) if a in available else 0.0
                    for a in range(NUM_ACTIONS)]
        return [self._dist[a] / total if a in available else 0.0
                for a in range(NUM_ACTIONS)]


def exploitability_of(strategy_table):
    """NashConv of an arbitrary Leduc strategy: how much both players together
    could gain by best-responding to it (0 iff it is an exact equilibrium).
    `strategy_table` maps each info-set key to a full 3-slot [fold, call, raise]
    distribution (as produced by `LeducCFR.strategy_table()` /
    `current_strategy_table`)."""
    cfr = LeducCFR()
    cfr.nodes = {iset: _FixedNode(dist) for iset, dist in strategy_table.items()}
    return cfr.exploitability()


def current_strategy_table(cfr):
    """The LAST-ITERATE (current regret-matching) strategy per info-set — no
    averaging. This is the regime a greedy/DQN-style self-play agent plays in;
    its exploitability does NOT converge to 0 even as CFR's average does."""
    result = {}
    for iset, node in cfr.nodes.items():
        avail = _avail_from_key(iset)
        total = sum(max(node.regret_sum[a], 0.0) for a in avail)
        dist = [0.0] * NUM_ACTIONS
        for a in avail:
            dist[a] = (max(node.regret_sum[a], 0.0) / total if total > 0
                       else 1.0 / len(avail))
        result[iset] = dist
    return result


def uniform_strategy_table():
    """A uniform-random strategy over every Leduc info-set (a maximally
    unsophisticated baseline; its exploitability is large)."""
    cfr = LeducCFR()
    cfr.train(1)  # populates cfr.nodes with every reachable info-set
    return {iset: [1.0 / len(_avail_from_key(iset))
                   if a in _avail_from_key(iset) else 0.0
                   for a in range(NUM_ACTIONS)]
            for iset in cfr.nodes}
