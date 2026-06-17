"""
belief_conditioned_equity.py
----------------------------
Adapter wiring beliefs (Phase 3/B) into equity and fold-equity (Phase A/B).

Two pieces:
    conditioned_equity   — hero equity vs per-opponent belief ranges (each
                           opponent's range widens automatically in a tilted
                           regime, since the belief's range_fraction does).
    tilt_adjusted_p_fold — deflates a base fold probability by the most-tilted
                           opponent's P(tilted): a tilting opponent folds less.
"""


def conditioned_equity(mc_engine, beliefs, hero_hole, community,
                       opponent_ids, rng, n_combos=200):
    """
    Equity for hero conditioned on each opponent's modeled range.

    Args:
        mc_engine (MonteCarloEngine): equity engine.
        beliefs (dict): {opponent_id: BeliefState}.
        hero_hole (list[Card]): hero hole cards.
        community (list[Card]): board (0-5 cards).
        opponent_ids (list[int]): live opponents to model.
        rng (random.Random): random source for range sampling.
        n_combos (int): candidate hands sampled per opponent range.

    Returns:
        float: equity in [0, 1]. Falls back to the uniform unknown-opponent
            estimate when no opponent has a belief.
    """
    exclude = list(hero_hole) + list(community)
    ranges = []
    for oid in opponent_ids:
        belief = beliefs.get(oid)
        if belief is None:
            continue
        ranges.append(belief.range_sample(n_combos, exclude, rng))

    if not ranges:
        n_opp = max(1, len(opponent_ids))
        return mc_engine.estimate_equity_unknown_opponents(
            hero_hole, n_opp, community
        )
    return mc_engine.estimate_equity_vs_range(hero_hole, ranges, community)


def tilt_adjusted_p_fold(base_p_fold, beliefs, opponent_ids, strength=0.5):
    """
    Deflate a fold probability by opponent tilt.

        adjusted = base * (1 - strength * max_i P(tilted_i))

    A tilting opponent folds less, so high P(tilted) lowers fold equity.

    Returns:
        float: adjusted p_fold in [0, base_p_fold].
    """
    max_tilt = 0.0
    for oid in opponent_ids:
        belief = beliefs.get(oid)
        if belief is not None:
            max_tilt = max(max_tilt, belief.p_tilted())
    return base_p_fold * (1.0 - strength * max_tilt)
