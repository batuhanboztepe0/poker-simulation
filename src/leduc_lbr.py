"""
leduc_lbr.py
------------
v2 Phase 1 tool: Local Best Response (LBR) exploitability as a LOWER BOUND.

Exact NashConv (`leduc_eval.exploitability_of`) requires solving a true best
response over every info-set — tractable on Leduc, but NOT on games too large to
enumerate. LBR (Lisý & Bowling 2017; REFERENCES.md) estimates exploitability from
BELOW by having the best-responder play a cheap, well-defined heuristic strategy:
at each of its decisions it picks the action with the highest value assuming it then
plays a fixed PASSIVE rollout (always check/call to showdown) for the rest of the
hand, valued against a Bayesian BELIEF over the opponent's private card (updated from
the opponent's fixed strategy along the public history). Because the assembled LBR
strategy is a *valid* strategy, its value is ≤ the exact best-response value, so:

    LBR exploitability  ≤  exact NashConv     (a lower bound, never an upper bound)

Implementation reuses the verified exact value-to-go evaluator `LeducCFR._br_value`
(opponent plays its fixed strategy; the LBR player follows a passive action map):
the LBR action at info-set s is the argmax over available actions of the
BELIEF-WEIGHTED value-to-go from s (take a, then passive), and the reported LBR
exploitability is the exact value of the assembled LBR map. On Leduc, where exact
NashConv is available, this lets us VALIDATE the lower-bound property (LBR ≤ exact)
before trusting LBR on a larger game where exact is infeasible.

Pure analysis, side-effect free; imported nowhere in the engine.
"""

from src.leduc_cfr import LeducCFR, CALL, NUM_ACTIONS, _avail_from_key
from src.leduc_eval import _FixedNode
from src.leduc_q import _all_info_sets

_ALL_CARDS = [0, 0, 1, 1, 2, 2]
_DEALS = [(_ALL_CARDS[i], _ALL_CARDS[j], _ALL_CARDS[k])
          for i in range(6) for j in range(6) for k in range(6)
          if i != j and i != k and j != k]
_ACT_IDX = {"f": 0, "c": 1, "r": 2}


def _owner(info_set):
    """Which player acts at this info-set (0/1), by round-history parity."""
    if "/" in info_set:
        return len(info_set[info_set.index("/") + 1:]) % 2
    return len(info_set[1:]) % 2


def _parse(info_set):
    """(c_br, board_or_None, r1, r2_or_None) for an info-set key."""
    if "/" in info_set:
        s = info_set.index("/")
        return int(info_set[0]), int(info_set[1]), info_set[2:s], info_set[s + 1:]
    return int(info_set[0]), None, info_set[1:], None


def _opp_reach(strategy_table, c_opp, board, r1, r2, opp):
    """P(opponent, holding c_opp, would have played the public history under its
    fixed strategy) — the unnormalised Bayesian belief weight for this c_opp."""
    reach = 1.0

    def play(hist, prefix_key_fn):
        nonlocal reach
        for i, ch in enumerate(hist):
            if i % 2 != opp:
                continue
            iset = prefix_key_fn(hist[:i])
            dist = strategy_table.get(iset)
            p = (dist[_ACT_IDX[ch]] if dist is not None
                 else 1.0 / len(_avail_from_key(iset)))
            reach *= p
            if reach == 0.0:
                return

    play(r1, lambda h: str(c_opp) + h)
    if r2 is not None:
        play(r2, lambda h: str(c_opp) + str(board) + r1 + "/" + h)
    return reach


def _lbr_action_map(cfr, strategy_table, br):
    """Assemble the LBR strategy for `br`: at each of its info-sets, the action
    maximising the belief-weighted value-to-go under a passive (check/call) rollout
    against the opponent's fixed strategy."""
    opp = 1 - br
    br_isets = [s for s in _all_info_sets() if _owner(s) == br]
    passive = {s: CALL for s in br_isets}   # CALL is legal at every Leduc info-set
    lbr_map = {}
    for s in br_isets:
        c_br, board, r1, r2 = _parse(s)
        avail = _avail_from_key(s)
        # Enumerate physical deals consistent with this info-set (br's rank fixed;
        # board fixed in round 2). Weight by the opponent's belief reach.
        consistent = []
        for ib in range(6):
            if _ALL_CARDS[ib] != c_br:
                continue
            for io in range(6):
                if io == ib:
                    continue
                for ik in range(6):
                    if ik == ib or ik == io:
                        continue
                    if board is not None and _ALL_CARDS[ik] != board:
                        continue
                    c_opp = _ALL_CARDS[io]
                    w = _opp_reach(strategy_table, c_opp, _ALL_CARDS[ik], r1, r2, opp)
                    if w == 0.0:
                        continue
                    c0 = c_br if br == 0 else c_opp
                    c1 = c_opp if br == 0 else c_br
                    consistent.append((c0, c1, _ALL_CARDS[ik], w))
        best_a, best_q = avail[0], float("-inf")
        for a in avail:
            trial = dict(passive)
            trial[s] = a
            # value-to-go from s: br takes a then passive, opp plays its strategy.
            q = sum(w * cfr._br_value(c0, c1, bd, r1, r2, br, trial)
                    for (c0, c1, bd, w) in consistent)
            if q > best_q:
                best_q, best_a = q, a
        lbr_map[s] = best_a
    return lbr_map


def lbr_exploitability(strategy_table):
    """LBR lower bound on the exploitability (NashConv) of a Leduc strategy.

    `strategy_table` maps each info-set key to a [p_fold, p_call, p_raise]
    distribution (same format as `leduc_eval.exploitability_of`). Returns a single
    float GUARANTEED ≤ the exact NashConv. It can be loose; for a near-equilibrium
    strategy it sits near zero. Never reported as an upper bound.
    """
    cfr = LeducCFR()
    cfr.nodes = {iset: _FixedNode(dist) for iset, dist in strategy_table.items()}
    total = 0.0
    for br in (0, 1):
        lbr_map = _lbr_action_map(cfr, strategy_table, br)
        total += sum(cfr._br_value(c0, c1, board, "", None, br, lbr_map)
                     for (c0, c1, board) in _DEALS)
    return total / len(_DEALS)
