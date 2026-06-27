"""
big_leduc_lbr.py
----------------
v2 Phase 2 scale-up: Local Best Response (LBR) lower-bound exploitability for the
parameterised R-rank Leduc (`big_leduc`). The R-rank analogue of `leduc_lbr`,
validated the same way: at small R, LBR ≤ exact NashConv. At the scaled R, where
exact NashConv is impractical, LBR is the lower-bound measurement, optionally
estimated from a sampled subset of deals (with a `samples` cap) so its cost does not
grow with the ~R^3 deal count.

Identical method to `leduc_lbr`: the best-responder plays the action maximising the
belief-weighted value-to-go under a passive check/call rollout, valued by the
verified `BigLeducCFR._br_value`. The assembled LBR strategy is valid, so its value
is ≤ the exact best-response value — a strict lower bound, never an upper bound.

Pure analysis, imported nowhere in the engine.
"""

import random

from src.leduc_cfr import CALL, _avail_from_key
from src.leduc_eval import _FixedNode
from src.big_leduc import BigLeducCFR, deck, char_rank, all_info_sets

_ACT_IDX = {"f": 0, "c": 1, "r": 2}


def _owner(info_set):
    if "/" in info_set:
        return len(info_set[info_set.index("/") + 1:]) % 2
    return len(info_set[1:]) % 2


def _parse(info_set):
    """(c_br_char, board_char_or_None, r1, r2_or_None)."""
    if "/" in info_set:
        s = info_set.index("/")
        return info_set[0], info_set[1], info_set[2:s], info_set[s + 1:]
    return info_set[0], None, info_set[1:], None


def _opp_reach(strategy_table, c_opp, board, r1, r2, opp):
    """Unnormalised Bayesian belief weight: P(opponent holding c_opp played the
    public history under its fixed strategy)."""
    reach = 1.0

    def play(hist, key_fn):
        nonlocal reach
        for i, ch in enumerate(hist):
            if i % 2 != opp:
                continue
            iset = key_fn(hist[:i])
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


def _lbr_value(cfr, ranks, strategy_table, br, all_deals, rng, samples):
    opp = 1 - br
    full_deck = deck(ranks)
    nd = len(full_deck)
    br_isets = [s for s in all_info_sets(ranks) if _owner(s) == br]
    passive = {s: CALL for s in br_isets}
    lbr_map = {}
    for s in br_isets:
        c_br, board, r1, r2 = _parse(s)
        avail = _avail_from_key(s)
        # Physical deals consistent with this info-set (br's rank fixed, board
        # fixed in round 2), weighted by the opponent's belief reach.
        consistent = []
        for ib in range(nd):
            if full_deck[ib] != c_br:
                continue
            for io in range(nd):
                if io == ib:
                    continue
                for ik in range(nd):
                    if ik == ib or ik == io:
                        continue
                    if board is not None and full_deck[ik] != board:
                        continue
                    c_opp = full_deck[io]
                    w = _opp_reach(strategy_table, c_opp, full_deck[ik], r1, r2, opp)
                    if w == 0.0:
                        continue
                    c0 = c_br if br == 0 else c_opp
                    c1 = c_opp if br == 0 else c_br
                    consistent.append((c0, c1, full_deck[ik], w))
        best_a, best_q = avail[0], float("-inf")
        for a in avail:
            trial = dict(passive)
            trial[s] = a
            q = sum(w * cfr._br_value(c0, c1, bd, r1, r2, br, trial)
                    for (c0, c1, bd, w) in consistent)
            if q > best_q:
                best_q, best_a = q, a
        lbr_map[s] = best_a
    # Value of the assembled LBR strategy, over all deals or a sampled subset.
    deals = all_deals
    if samples is not None and samples < len(all_deals):
        deals = rng.sample(all_deals, samples)
    return (sum(cfr._br_value(c0, c1, bd, "", None, br, lbr_map)
                for (c0, c1, bd) in deals) / len(deals)) * len(all_deals)


def lbr_exploitability(strategy_table, ranks, samples=None, seed=0):
    """LBR lower bound on the exploitability of an R-rank Leduc strategy.

    `samples` optionally caps the number of deals used for the final value estimate
    (deterministic given `seed`), so the cost does not scale with the ~R^3 deal
    count. With full enumeration (`samples=None`, `samples <= 0`, or
    `samples >= num_deals`) the result is GUARANTEED ≤ the exact NashConv. With
    sampling it is an UNBIASED ESTIMATE of that lower bound, not a guaranteed
    bound: a single sampled draw can land above the exact NashConv — only the
    expectation is bounded.
    """
    if samples is not None and samples <= 0:
        samples = None        # 0 / negative means "no cap" -> full enumeration
    cfr = BigLeducCFR(ranks)
    cfr.nodes = {iset: _FixedNode(dist) for iset, dist in strategy_table.items()}
    full_deck = deck(ranks)
    nd = len(full_deck)
    all_deals = [(full_deck[i], full_deck[j], full_deck[k])
                 for i in range(nd) for j in range(nd) for k in range(nd)
                 if i != j and i != k and j != k]
    # Independent rng per player so the two value estimates are not correlated
    # through one shared, advancing stream (matters only under sampling; with
    # full enumeration the rng is unused, so this is a no-op there).
    total = (_lbr_value(cfr, ranks, strategy_table, 0, all_deals,
                        random.Random(seed), samples)
             + _lbr_value(cfr, ranks, strategy_table, 1, all_deals,
                          random.Random(seed + 1), samples))
    return total / len(all_deals)
