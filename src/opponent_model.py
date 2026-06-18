"""
opponent_model.py
-----------------
Static Bayesian opponent model (Phase 3).

`BeliefState` maintains two Beta posteriors about an opponent — one over their
aggression (P[aggressive action]) and one over their looseness (P[play a
hand]) — and turns the looseness estimate into a concrete hand range. Hero
conditions its equity on that range via
`MonteCarloEngine.estimate_equity_vs_range`, so a tightly-modeled opponent
(narrow, strong range) lowers hero's equity relative to a loosely-modeled one.

This class defines the FROZEN INTERFACE that Phases A, B and C depend on:

    update(action, n_aggressive, n_actions, delta_stack=0) -> None
    observe_pnl(delta_stack) -> None  # hand-boundary tilt transition (HMM only)
    update_from_showdown(hole_cards) -> None
    posterior_mean() -> float        # E[aggression]
    p_tilted() -> float              # 0.0 for the static model
    range_sample(n_combos, exclude_cards, rng) -> list[list[Card]]

Phase B subclasses (DynamicBeliefState, HMMBeliefState, ParticleBelief) must
honour the same method signatures.
"""

from src.card import Card, RANKS, SUITS

# Actions that count as voluntarily putting money in the pot.
VOLUNTARY_ACTIONS = {"call", "raise", "all_in"}
AGGRESSIVE_ACTIONS = {"raise", "all_in"}
FOLD_ACTION = "fold"

# Range fraction is clamped so a sampled range is never empty nor the whole deck.
MIN_RANGE_FRACTION = 0.05
MAX_RANGE_FRACTION = 1.0

# Showdown strength cutoff: hands weaker than this are evidence of a loose
# opponent (they were willing to play a weak holding).
WEAK_STRENGTH_CUTOFF = 35.0


def combo_strength(c1, c2):
    """
    Heuristic pre-flop strength score for a two-card combo (higher = stronger).

    Monotone enough to rank starting hands sensibly: pairs dominate, then
    high cards, with bonuses for suitedness and connectedness. Used only to
    order hands when carving a range — not as an equity estimate.
    """
    r1 = RANKS.index(c1.rank)
    r2 = RANKS.index(c2.rank)
    hi, lo = max(r1, r2), min(r1, r2)
    pair = r1 == r2
    suited = c1.suit == c2.suit

    score = 2.0 * hi + lo
    if pair:
        score += 30.0 + hi
    if suited:
        score += 4.0
    if not pair:
        gap = hi - lo
        if gap == 1:
            score += 3.0
        elif gap == 2:
            score += 1.5
    return score


class BeliefState:
    """
    Static Beta-Bernoulli belief about one opponent.

    Attributes:
        alpha_aggr / beta_aggr: Beta posterior over aggression.
        alpha_loose / beta_loose: Beta posterior over looseness (VPIP-like).
        n_observations: number of update / showdown observations folded in.
    """

    def __init__(self, prior_aggression=0.5, prior_looseness=0.5,
                 prior_strength=2.0):
        """
        Args:
            prior_aggression (float): Prior mean E[aggression] in (0, 1).
            prior_looseness (float): Prior mean E[looseness] in (0, 1).
            prior_strength (float): Pseudo-count weight of the priors (higher =
                stickier prior). Beta(a, b) with a+b = prior_strength.
        """
        self.alpha_aggr = prior_strength * prior_aggression
        self.beta_aggr = prior_strength * (1.0 - prior_aggression)
        self.alpha_loose = prior_strength * prior_looseness
        self.beta_loose = prior_strength * (1.0 - prior_looseness)

        self.n_observations = 0
        self.total_actions = 0
        self.shown_hands = []

    # -- frozen interface --------------------------------------------------

    def update(self, action, n_aggressive, n_actions, delta_stack=0):
        """
        Fold one batch of observed actions into the posteriors.

        Args:
            action (str): The most recent action string (for the looseness
                signal: voluntary vs fold).
            n_aggressive (int): Aggressive actions in this batch.
            n_actions (int): Total actions in this batch.
            delta_stack (int): Opponent stack change; unused by the static
                model (Phase B's tilt trigger consumes it).
        """
        self.alpha_aggr += n_aggressive
        self.beta_aggr += max(0, n_actions - n_aggressive)
        self._observe_looseness(action)

        self.n_observations += 1
        self.total_actions += n_actions

    def _observe_looseness(self, action):
        """Fold one action into the looseness Beta (voluntary vs fold)."""
        if action in VOLUNTARY_ACTIONS:
            self.alpha_loose += 1.0
        elif action == FOLD_ACTION:
            self.beta_loose += 1.0

    def observe_pnl(self, delta_stack):
        """
        Hand-boundary hook: a PnL-driven regime transition.

        Called once per hand with the opponent's realised per-hand chip delta.
        The static / Beta models have no tilt regime, so this is a no-op; only
        `HMMBeliefState` consumes realised PnL (see its override).
        """
        pass

    def update_from_showdown(self, hole_cards):
        """
        Refine the looseness estimate from a revealed holding.

        A weak shown hand is evidence the opponent plays loosely; a strong one
        is weak evidence of tightness.
        """
        if len(hole_cards) != 2:
            return
        self.shown_hands.append(tuple(hole_cards))
        strength = combo_strength(hole_cards[0], hole_cards[1])
        if strength < WEAK_STRENGTH_CUTOFF:
            self.alpha_loose += 1.0
        else:
            self.beta_loose += 0.5
        self.n_observations += 1

    def posterior_mean(self):
        """E[aggression] under the current posterior."""
        return self.alpha_aggr / (self.alpha_aggr + self.beta_aggr)

    def p_tilted(self):
        """Probability the opponent is tilted. Always 0.0 for the static model."""
        return 0.0

    def range_sample(self, n_combos, exclude_cards, rng):
        """
        Sample `n_combos` opponent hands from the estimated playing range.

        The played range is the top `range_fraction()` of all two-card combos
        by `combo_strength`, after removing any combo that uses an excluded
        card. Tighter opponents -> smaller, stronger range.

        Args:
            n_combos (int): Number of sampled hands to return (the MC engine
                draws from these per simulation).
            exclude_cards (list[Card]): Known cards (hero hole + board) that
                cannot appear in an opponent hand.
            rng (random.Random): Random source.

        Returns:
            list[list[Card]]: `n_combos` hands, each a [Card, Card] pair.
        """
        excl = set(c.treys_int for c in exclude_cards)
        available = [Card(r, s) for s in SUITS for r in RANKS
                     if Card(r, s).treys_int not in excl]

        combos = []
        m = len(available)
        for i in range(m):
            for j in range(i + 1, m):
                combos.append((available[i], available[j]))
        combos.sort(key=lambda pair: combo_strength(pair[0], pair[1]),
                    reverse=True)

        k = max(1, int(round(self.range_fraction() * len(combos))))
        top = combos[:k]
        return [list(rng.choice(top)) for _ in range(n_combos)]

    # -- helpers -----------------------------------------------------------

    def looseness_mean(self):
        """E[looseness] under the current posterior."""
        return self.alpha_loose / (self.alpha_loose + self.beta_loose)

    def range_fraction(self):
        """Fraction of all combos in the opponent's range, clamped to bounds."""
        f = self.looseness_mean()
        return min(MAX_RANGE_FRACTION, max(MIN_RANGE_FRACTION, f))

    def __repr__(self):
        return (f"BeliefState(E[aggr]={self.posterior_mean():.3f}, "
                f"E[loose]={self.looseness_mean():.3f}, "
                f"n_obs={self.n_observations})")


# ===========================================================================
# Phase B — stochastic / adaptive opponent models
# ===========================================================================

import math
import random as _random

# How much a fully-tilted regime widens the sampled range.
TILT_RANGE_WIDEN = 0.45


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _binom_pmf(k, n, p):
    """Binomial pmf with clamped p (avoids 0/1 degeneracy)."""
    if n <= 0:
        return 1.0
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


class DynamicBeliefState(BeliefState):
    """
    Discounted-Beta (EWMA) belief: recent actions weighted more heavily.

    Each update first discounts the existing pseudo-counts by `decay` before
    adding the new observation:
        alpha <- decay*alpha + k ,  beta <- decay*beta + (n - k)
    so the effective memory has half-life -1/log(decay).

    Quant parallel: RiskMetrics EWMA volatility estimation.
    """

    def __init__(self, decay=0.94, **kwargs):
        super().__init__(**kwargs)
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}.")
        self.decay = decay

    def update(self, action, n_aggressive, n_actions, delta_stack=0):
        d = self.decay
        self.alpha_aggr = d * self.alpha_aggr + n_aggressive
        self.beta_aggr = d * self.beta_aggr + max(0, n_actions - n_aggressive)

        # Discount looseness counts, then add the new evidence.
        self.alpha_loose *= d
        self.beta_loose *= d
        self._observe_looseness(action)

        self.n_observations += 1
        self.total_actions += n_actions

    def half_life(self):
        """Half-life of the discount in number of updates."""
        return -math.log(2.0) / math.log(self.decay)


class TiltTrigger:
    """
    PnL-driven tilt transition probability.

        P(normal -> tilted) = min(1, epsilon + kappa * max(0, -delta_stack/stack0))

    A losing opponent (delta_stack < 0) becomes more likely to tilt; a winning
    or break-even opponent stays at the baseline epsilon.

    Quant parallel: drawdown-triggered risk-regime shift.
    """

    def __init__(self, epsilon=0.04, kappa=1.5, stack0=1000):
        self.epsilon = epsilon
        self.kappa = kappa
        self.stack0 = max(1, stack0)

    def transition_prob(self, delta_stack):
        loss_fraction = max(0.0, -delta_stack / self.stack0)
        return min(1.0, self.epsilon + self.kappa * loss_fraction)


class HMMBeliefState(BeliefState):
    """
    Two-state HMM regime switch: {normal, tilted}.

    Forward filter on the regime belief pi = (P[normal], P[tilted]):
        predict:  pi_pred = Aᵀ pi      (A from the TiltTrigger + recovery rate)
        update:   pi = normalize( emission ⊙ pi_pred )
    Emission per state is Binomial(n_aggressive; n_actions, mu_state), with the
    tilted state more aggressive (mu_tilted > mu_normal). E[aggression] is the
    regime-weighted emission mean, and the sampled range widens with P(tilted).

    Quant parallel: Hamilton regime-switching model.
    """

    def __init__(self, mu_normal=0.30, mu_tilted=0.80, recover=0.15,
                 prior_tilted=0.1, tilt_trigger=None, use_pnl=True, **kwargs):
        super().__init__(**kwargs)
        self.mu_normal = mu_normal
        self.mu_tilted = mu_tilted
        self.recover = recover            # P(tilted -> normal)
        self.tilt_trigger = tilt_trigger or TiltTrigger()
        # When True, observe_pnl() fires the TiltTrigger at each hand boundary
        # from the opponent's realised PnL (earlier detection). False reproduces
        # the aggression-emission-only detector (the dormant pre-fix behaviour).
        self.use_pnl = use_pnl
        self.pi_normal = 1.0 - prior_tilted
        self.pi_tilted = prior_tilted

    def update(self, action, n_aggressive, n_actions, delta_stack=0):
        p_nt = self.tilt_trigger.transition_prob(delta_stack)  # normal->tilted
        p_tn = self.recover                                    # tilted->normal

        # Predict (apply transition).
        pred_normal = self.pi_normal * (1.0 - p_nt) + self.pi_tilted * p_tn
        pred_tilted = self.pi_normal * p_nt + self.pi_tilted * (1.0 - p_tn)

        # Emission likelihood of the observed aggression count per state.
        b_normal = _binom_pmf(n_aggressive, n_actions, self.mu_normal)
        b_tilted = _binom_pmf(n_aggressive, n_actions, self.mu_tilted)

        post_normal = b_normal * pred_normal
        post_tilted = b_tilted * pred_tilted
        z = post_normal + post_tilted
        if z <= 0:
            self.pi_normal, self.pi_tilted = pred_normal, pred_tilted
        else:
            self.pi_normal = post_normal / z
            self.pi_tilted = post_tilted / z

        # Track looseness for range sampling; aggression posterior is regime-led.
        self.alpha_aggr += n_aggressive
        self.beta_aggr += max(0, n_actions - n_aggressive)
        self._observe_looseness(action)
        self.n_observations += 1
        self.total_actions += n_actions

    def observe_pnl(self, delta_stack):
        """
        Hand-boundary regime transition driven by the opponent's realised
        per-hand PnL — the HMM *predict* step only (no emission).

        A losing opponent's P(normal -> tilted) rises via the TiltTrigger, so
        p_tilted climbs the moment a hand is lost, *before* the opponent's
        aggression emissions reveal the regime. Because this trigger shares the
        opponent's epsilon/kappa/stack0, the predicted P(tilted) matches the
        opponent's TRUE transition probability — the belief leads the
        aggression-only detector by one hand. A winning / break-even hand only
        applies the recovery drift (toward normal), mirroring the opponent's own
        per-hand recovery. The 2x2 transition is stochastic, so the regime belief
        stays normalised without a renorm step.
        """
        if not self.use_pnl:
            return
        p_nt = self.tilt_trigger.transition_prob(delta_stack)  # normal->tilted
        p_tn = self.recover                                    # tilted->normal
        pred_normal = self.pi_normal * (1.0 - p_nt) + self.pi_tilted * p_tn
        pred_tilted = self.pi_normal * p_nt + self.pi_tilted * (1.0 - p_tn)
        self.pi_normal, self.pi_tilted = pred_normal, pred_tilted

    def posterior_mean(self):
        """E[aggression] = regime-weighted emission mean."""
        return self.pi_normal * self.mu_normal + self.pi_tilted * self.mu_tilted

    def p_tilted(self):
        return self.pi_tilted

    def range_fraction(self):
        # Widen the played range as tilt probability rises.
        widened = self.looseness_mean() + self.p_tilted() * TILT_RANGE_WIDEN
        return min(MAX_RANGE_FRACTION, max(MIN_RANGE_FRACTION, widened))


class ParticleBelief(BeliefState):
    """
    Particle filter over an OU latent aggression process.

        x_t = phi * x_{t-1} + sigma * eps ,   aggression_t = sigmoid(x_t)
    Each particle is weighted by the Bernoulli likelihood of the observed
    aggressive actions; the filter resamples (systematic) with roughening
    jitter when the effective sample size drops below N/2.

    Quant parallel: stochastic-volatility particle filtering.
    """

    def __init__(self, n_particles=200, phi=0.95, sigma=0.3, rng=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_particles = n_particles
        self.phi = phi
        self.sigma = sigma
        # Default to a DETERMINISTIC rng so the "one seed reproduces a session"
        # invariant holds even when no rng is injected. Pass a per-opponent
        # seeded rng (e.g. random.Random(session_seed + player_id)) for
        # independent, still-reproducible particle trajectories.
        self._rng = rng if rng is not None else _random.Random(0)
        self.x = [self._rng.gauss(0.0, 1.0) for _ in range(n_particles)]
        self.w = [1.0 / n_particles] * n_particles

    def update(self, action, n_aggressive, n_actions, delta_stack=0):
        # Propagate the OU latent state.
        self.x = [self.phi * xi + self.sigma * self._rng.gauss(0.0, 1.0)
                  for xi in self.x]

        # Weight by Bernoulli likelihood of the observed aggression.
        new_w = []
        for xi, wi in zip(self.x, self.w):
            a = min(max(_sigmoid(xi), 1e-6), 1.0 - 1e-6)
            like = (a ** n_aggressive) * ((1.0 - a) ** (n_actions - n_aggressive))
            new_w.append(wi * like)
        z = sum(new_w)
        if z <= 0:
            self.w = [1.0 / self.n_particles] * self.n_particles
        else:
            self.w = [w / z for w in new_w]

        ess = 1.0 / sum(w * w for w in self.w)
        if ess < self.n_particles / 2.0:
            self._resample()

        self._observe_looseness(action)
        self.n_observations += 1
        self.total_actions += n_actions

    def posterior_mean(self):
        """E[aggression] = weighted mean of sigmoid(latent state)."""
        return sum(w * _sigmoid(xi) for xi, w in zip(self.x, self.w))

    def effective_sample_size(self):
        return 1.0 / sum(w * w for w in self.w)

    def _resample(self):
        """Systematic resampling with roughening jitter to fight degeneracy."""
        n = self.n_particles
        positions = [(self._rng.random() + i) / n for i in range(n)]
        cumulative = []
        c = 0.0
        for w in self.w:
            c += w
            cumulative.append(c)
        new_x = []
        i = 0
        for pos in positions:
            while i < n - 1 and pos > cumulative[i]:
                i += 1
            new_x.append(self.x[i])
        # Roughening: jitter proportional to the spread of resampled particles.
        mean_x = sum(new_x) / n
        var = sum((xi - mean_x) ** 2 for xi in new_x) / n
        jitter = math.sqrt(var) * (n ** (-1.0 / 3.0)) if var > 0 else 1e-3
        self.x = [xi + self._rng.gauss(0.0, jitter) for xi in new_x]
        self.w = [1.0 / n] * n
