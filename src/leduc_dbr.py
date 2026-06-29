"""
leduc_dbr.py
------------
Detect-then-exploit on Leduc Hold'em, measured EXACTLY (v2 Phase C3, PREREGISTRATION §15).

§14 (`leduc_rnr`) handed the RNR solver the opponent's EXACT strategy. That is the
easy case: the positive EV gain over Nash is essentially guaranteed by construction,
because best-responding to a known suboptimal opponent cannot do worse than Nash.

The honest, non-trivial question is whether exploitation survives DETECTION. Here the
hero plays Nash, OBSERVES the opponent for N hands, ESTIMATES its strategy (a
Dirichlet-smoothed empirical frequency table, `sigma_hat`), and only then computes
RNR(`sigma_hat`, p). We measure the REALIZED EV against the TRUE opponent, exactly.

This is the data-biased-response setting (Johanson & Bowling, AISTATS 2009): with
finite data a raw best response (p=1) overfits the estimate, while a restricted
response (0<p<1) trades exploitation for robustness to estimation error. Everything
downstream of the estimate is exact (120-deal enumeration), so the only randomness is
the N-hand observation sample, which we average over multiple seeds.

The opponent's own randomness matters. A deterministic opponent (station, maniac) is
pinned by one visit to an info-set, so the only estimation error is the info-sets that
N hands never reached. A stochastic opponent (uniform) needs many samples per info-set
to estimate its mixing, so it is where high p overfits the noise.
"""
import random

from src.leduc_cfr import (
    NUM_ACTIONS, FOLD, CALL, RAISE,
    _is_terminal, _available_actions,
)
from src.leduc_rnr import (
    _deals, _info_key, LeducRNR, ev_player0,
    station, maniac, uniform, OPPONENTS, nash_strategy,
)


def loose_passive(info_set, avail):
    """A STOCHASTIC, NON-uniform opponent: calls most, raises rarely, folds rarely.
    Unlike `uniform`, the uniform smoothing prior is a WRONG estimate of it, and unlike
    the deterministic station/maniac it cannot be pinned from one visit. Estimating it
    therefore needs many samples per info-set, so this is the opponent on which a raw
    best response to a finite-sample estimate (p=1) is expected to overfit."""
    d = [0.0] * NUM_ACTIONS
    if RAISE in avail and FOLD in avail:
        d[FOLD], d[CALL], d[RAISE] = 0.05, 0.70, 0.25
    elif RAISE in avail:
        d[CALL], d[RAISE] = 0.75, 0.25
    elif FOLD in avail:
        d[CALL], d[FOLD] = 0.85, 0.15
    else:
        d[CALL] = 1.0
    return d


def _sample_action(dist, avail, rng):
    """Sample one legal action from `dist` (a length-NUM_ACTIONS distribution)."""
    r = rng.random()
    cum = 0.0
    for a in avail:
        cum += dist[a]
        if r < cum:
            return a
    return avail[-1]


def _play_record(c0, c1, board, r1, r2, s0, s1, rng, counts):
    """Play one deal (s0 vs s1, actions sampled), recording PLAYER 1's chosen action
    at each info-set it acts into counts[info_set] = [n_fold, n_call, n_raise].
    Mirrors the round-1 -> round-2 transition of `leduc_rnr._value0`."""
    in_r2 = (r2 is not None)
    curr = r2 if in_r2 else r1
    player = len(curr) % 2
    if _is_terminal(curr, in_r2):
        if not in_r2 and curr[-1] == 'c':
            _play_record(c0, c1, board, r1, '', s0, s1, rng, counts)
        return
    avail = _available_actions(curr, in_r2)
    info_set = _info_key(c0, c1, board, r1, curr, player, in_r2)
    dist = (s0 if player == 0 else s1)(info_set, avail)
    a = _sample_action(dist, avail, rng)
    if player == 1:
        counts.setdefault(info_set, [0, 0, 0])[a] += 1
    ch = curr + ('f' if a == FOLD else 'c' if a == CALL else 'r')
    if in_r2:
        _play_record(c0, c1, board, r1, ch, s0, s1, rng, counts)
    else:
        _play_record(c0, c1, board, ch, None, s0, s1, rng, counts)


def observe(sigma_true, n_hands, seed, reference):
    """Play `reference` (player 0) against `sigma_true` (player 1) for n_hands deals,
    sampled uniformly from the 120 physical deals, recording player 1's actions.

    Returns the counts dict (info_set -> [n_fold, n_call, n_raise]). This is the
    realistic detection setting: the hero plays a reference strategy (Nash) and only
    sees the opponent at the info-sets their joint play actually reaches."""
    rng = random.Random(seed)
    deals = list(_deals())
    counts = {}
    for _ in range(n_hands):
        c0, c1, board = deals[rng.randrange(len(deals))]
        _play_record(c0, c1, board, '', None, reference, sigma_true, rng, counts)
    return counts


def sigma_hat_fn(counts, alpha=1.0):
    """Dirichlet(alpha)-smoothed empirical estimate of the opponent's strategy as a
    callable (info_set, avail) -> distribution. An unseen info-set falls back to the
    uniform prior (alpha on every legal action with no data)."""
    def f(info_set, avail):
        d = [0.0] * NUM_ACTIONS
        c = counts.get(info_set)
        if c is None:
            for a in avail:
                d[a] = 1.0 / len(avail)
            return d
        denom = sum(c[a] for a in avail) + alpha * len(avail)
        for a in avail:
            d[a] = (c[a] + alpha) / denom
        return d
    return f


def detect_then_exploit(sigma_true, n_hands, seed, reference, p, iters, alpha=1.0):
    """One end-to-end trial: observe `sigma_true` for n_hands (hero plays `reference`),
    estimate sigma_hat, solve RNR(sigma_hat, p), and return a dict with the REALIZED
    exact EV of the counter against the TRUE opponent, the counter's exploitability,
    and the number of distinct opponent info-sets actually observed.

    The EV is exact (120-deal enumeration); the only randomness is the observation."""
    counts = observe(sigma_true, n_hands, seed, reference)
    sigma_hat = sigma_hat_fn(counts, alpha)
    rnr = LeducRNR(sigma_hat, p)
    rnr.train(iters)
    counter = rnr.counter_fn()
    return {
        "ev_vs_true": ev_player0(counter, sigma_true),
        "exploitability": rnr.counter_exploitability(),
        "infosets_seen": len(counts),
    }
