"""
fold_equity.py
--------------
Fold-equity estimation (Phase A).

`FoldEquityModel.estimate_p_fold` turns the table state + opponent beliefs into
the probability that everyone folds to a raise. With a diffuse belief it falls
back to the GTO-neutral prior b/(pot+b); with a warmed-up looseness posterior
it integrates the opponent's continue-propensity against the bet "price"
x = b/(pot+b) using the regularized incomplete beta function:

    p_fold_i = I_x(alpha_loose_i, beta_loose_i)

A looser opponent (Beta mass toward 1) has a small CDF at x, so folds less; a
bigger bet (larger x) raises the CDF, so induces more folds. Multi-way fold
equity is the product of per-opponent fold probabilities (independence
assumption — documented as a lower bound).

scipy.special.betainc is used when available; otherwise a self-contained
continued-fraction implementation (Numerical Recipes `betai`) is used, so the
module imports with no hard scipy dependency.
"""

import math

try:  # scipy is optional — pure-Python fallback below
    from scipy.special import betainc as _scipy_betainc
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - exercised only without scipy
    _HAVE_SCIPY = False


# Minimum observations before a belief is considered informative (else diffuse).
FOLD_EQUITY_MIN_OBS = 5


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    MAXIT = 300
    EPS = 3.0e-16
    FPMIN = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def regularized_incomplete_beta(a, b, x):
    """
    Regularized incomplete beta I_x(a, b) in [0, 1].

    Uses scipy.special.betainc when available; otherwise a continued-fraction
    fallback. Both agree to ~1e-12.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if _HAVE_SCIPY:
        return float(_scipy_betainc(a, b, x))

    ln_beta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
               + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(ln_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


class FoldEquityModel:
    """
    Estimates the probability opponents fold to a raise.

    Args:
        beliefs (dict | None): {opponent_id: BeliefState}. Opponents without a
            belief (or with a diffuse one) use the GTO-neutral prior.
    """

    def __init__(self, beliefs=None):
        self.beliefs = beliefs if beliefs is not None else {}

    def estimate_p_fold(self, opponent_ids, raise_amount, pot,
                        n_opponents=None):
        """
        Probability that ALL opponents fold to a raise of `raise_amount`.

        Args:
            opponent_ids (list[int] | None): Live opponents facing the raise.
            raise_amount (int): Raise size b (chips).
            pot (int): Pot before the raise.
            n_opponents (int | None): Override opponent count when ids are
                unavailable; defaults to len(opponent_ids) or 1.

        Returns:
            float: P(all fold) in [0, 1]. With no belief info this is the
                GTO-neutral b/(pot+b) (raised to the opponent count).
        """
        if raise_amount <= 0:
            raise ValueError(
                f"raise_amount must be positive, got {raise_amount}."
            )
        x = raise_amount / (pot + raise_amount)  # bet "price" / GTO p_fold

        ids = list(opponent_ids) if opponent_ids else []
        if n_opponents is None:
            n_opponents = len(ids) if ids else 1

        per_opponent = []
        for oid in ids:
            belief = self.beliefs.get(oid)
            per_opponent.append(self._p_fold_one(belief, x))

        # Pad with neutral-prior opponents if ids are missing.
        while len(per_opponent) < n_opponents:
            per_opponent.append(x)

        if not per_opponent:
            return x

        p_all_fold = 1.0
        for pf in per_opponent:
            p_all_fold *= pf
        return p_all_fold

    def _p_fold_one(self, belief, x):
        """Per-opponent fold probability; neutral prior when belief is diffuse."""
        if belief is None or belief.n_observations < FOLD_EQUITY_MIN_OBS:
            return x  # GTO-neutral fold probability b/(pot+b)
        # Opponent continues when their looseness theta ~ Beta exceeds the price
        # x; they fold with probability P(theta < x) = I_x(alpha, beta).
        return regularized_incomplete_beta(
            belief.alpha_loose, belief.beta_loose, x
        )
