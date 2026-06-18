"""
test_phase_b.py
---------------
Tests for Phase B: stochastic / adaptive opponent modeling.

DoD (ROADMAP §10):
    - discounted-Beta recency,
    - HMM recovers an injected regime switch (P(tilted) > 0.8 after switch),
    - particle filter tracks a synthetic OU path within an RMSE bound,
    - tilted range is wider than normal,
    - one ParticleBelief.update() < 5 ms.

Run with: python -m pytest tests/test_phase_b.py -v
"""

import sys
import os
import math
import time
import random
import inspect
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.card import Card
from src.monte_carlo import MonteCarloEngine
from src.player import BotPlayer
from src.game import GameEngine
from src.opponent_model import (
    BeliefState, DynamicBeliefState, HMMBeliefState, ParticleBelief,
    TiltTrigger, _sigmoid,
)
from src.belief_conditioned_equity import (
    conditioned_equity, tilt_adjusted_p_fold,
)
from src.simulation import simulate_session
from src.analytics import belief_trace_dataframe


def make_cards(card_strs):
    return [Card(r, s) for r, s in card_strs]


# ===========================================================================
# Frozen interface conformance for all Phase B models
# ===========================================================================

class TestInterfaceConformance:
    @pytest.mark.parametrize("cls,kwargs", [
        (DynamicBeliefState, {}),
        (HMMBeliefState, {}),
        (ParticleBelief, {"n_particles": 50}),
    ])
    def test_has_frozen_interface(self, cls, kwargs):
        b = cls(**kwargs)
        for name in ("update", "update_from_showdown", "posterior_mean",
                     "p_tilted", "range_sample"):
            assert callable(getattr(b, name))
        # smoke: update + sample
        b.update("raise", 1, 1)
        assert 0.0 <= b.posterior_mean() <= 1.0
        assert 0.0 <= b.p_tilted() <= 1.0
        hands = b.range_sample(20, [], random.Random(0))
        assert len(hands) == 20


# ===========================================================================
# DynamicBeliefState (EWMA)
# ===========================================================================

class TestDynamicBelief:
    def test_recency_weights_recent_actions(self):
        dyn = DynamicBeliefState(decay=0.85)
        sta = BeliefState()
        for _ in range(30):
            dyn.update("raise", 1, 1)
            sta.update("raise", 1, 1)
        for _ in range(30):
            dyn.update("call", 0, 1)
            sta.update("call", 0, 1)
        # Dynamic discounts the old aggression, so it reads the recent passive
        # behavior; static averages everything to ~0.5.
        assert dyn.posterior_mean() < sta.posterior_mean()
        assert dyn.posterior_mean() < 0.35

    def test_reacts_faster_to_a_switch(self):
        dyn = DynamicBeliefState(decay=0.8)
        sta = BeliefState()
        for _ in range(20):
            dyn.update("call", 0, 1)
            sta.update("call", 0, 1)
        for _ in range(5):
            dyn.update("raise", 1, 1)
            sta.update("raise", 1, 1)
        assert dyn.posterior_mean() > sta.posterior_mean()

    def test_half_life(self):
        assert DynamicBeliefState(decay=0.5).half_life() == pytest.approx(1.0)

    def test_invalid_decay_raises(self):
        with pytest.raises(ValueError):
            DynamicBeliefState(decay=1.5)


# ===========================================================================
# TiltTrigger
# ===========================================================================

class TestTiltTrigger:
    def test_baseline_when_not_losing(self):
        t = TiltTrigger(epsilon=0.04, kappa=1.5, stack0=1000)
        assert t.transition_prob(0) == pytest.approx(0.04)
        assert t.transition_prob(200) == pytest.approx(0.04)  # a gain

    def test_loss_raises_tilt_probability(self):
        t = TiltTrigger(epsilon=0.04, kappa=1.5, stack0=1000)
        assert t.transition_prob(-500) > 0.04
        assert t.transition_prob(-500) == pytest.approx(0.04 + 1.5 * 0.5)

    def test_clamped_to_one(self):
        t = TiltTrigger(epsilon=0.04, kappa=1.5, stack0=1000)
        assert t.transition_prob(-10_000_000) == 1.0


# ===========================================================================
# HMM regime switch
# ===========================================================================

class TestHMM:
    def test_recovers_injected_regime_switch(self):
        hmm = HMMBeliefState()
        for _ in range(8):
            hmm.update("call", 0, 1)   # passive -> normal regime
        assert hmm.p_tilted() < 0.3
        for _ in range(15):
            hmm.update("raise", 1, 1)  # aggressive burst -> tilted regime
        assert hmm.p_tilted() > 0.8

    def test_posterior_mean_between_regime_means(self):
        hmm = HMMBeliefState(mu_normal=0.3, mu_tilted=0.8)
        for _ in range(5):
            hmm.update("call", 0, 1)
        assert 0.3 <= hmm.posterior_mean() <= 0.8

    def test_tilted_range_wider_than_normal(self):
        # Isolate the tilt effect: identical looseness (neutral checks),
        # different regime belief.
        h_normal = HMMBeliefState()
        h_tilted = HMMBeliefState()
        for _ in range(10):
            h_normal.update("check", 0, 1)
            h_tilted.update("check", 0, 1)
        h_normal.pi_tilted, h_normal.pi_normal = 0.0, 1.0
        h_tilted.pi_tilted, h_tilted.pi_normal = 0.9, 0.1
        assert h_tilted.range_fraction() > h_normal.range_fraction()


# ===========================================================================
# Particle filter (OU drift)
# ===========================================================================

class TestParticleFilter:
    def test_tracks_synthetic_ou_path(self):
        pf = ParticleBelief(n_particles=400, phi=0.95, sigma=0.3,
                            rng=random.Random(1))
        gen = random.Random(2)
        x_true = 0.0
        sq_errors = []
        for t in range(80):
            x_true = 0.95 * x_true + 0.3 * gen.gauss(0.0, 1.0)
            p_true = _sigmoid(x_true)
            aggressive = 1 if gen.random() < p_true else 0
            pf.update("raise" if aggressive else "call", aggressive, 1)
            if t >= 15:  # after burn-in
                sq_errors.append((pf.posterior_mean() - p_true) ** 2)
        rmse = math.sqrt(sum(sq_errors) / len(sq_errors))
        assert rmse < 0.3, f"OU tracking RMSE too high: {rmse:.3f}"

    def test_update_under_5ms(self):
        pf = ParticleBelief(n_particles=200, rng=random.Random(0))
        for _ in range(5):  # warm up
            pf.update("call", 0, 1)
        start = time.perf_counter()
        n = 50
        for _ in range(n):
            pf.update("raise", 1, 1)
        per_update = (time.perf_counter() - start) / n
        assert per_update < 0.005, f"update took {per_update*1000:.2f} ms"

    def test_resampling_keeps_weights_normalised(self):
        pf = ParticleBelief(n_particles=100, rng=random.Random(0))
        for _ in range(30):
            pf.update("raise", 1, 1)
        assert abs(sum(pf.w) - 1.0) < 1e-9
        assert pf.effective_sample_size() > 1.0


# ===========================================================================
# belief_conditioned_equity adapter
# ===========================================================================

class TestBeliefConditionedEquity:
    def test_wider_range_gives_higher_equity(self):
        mc = MonteCarloEngine(n_simulations=1200, rng=random.Random(3))
        rng = random.Random(0)
        tight = BeliefState()
        for _ in range(40):
            tight.update("fold", 0, 1)         # narrow strong range
        wide = HMMBeliefState()
        for _ in range(15):
            wide.update("raise", 1, 1)         # tilted + loose -> wide range
        hero = make_cards([("K", "d"), ("Q", "c")])
        eq_tight = conditioned_equity(mc, {2: tight}, hero, [], [2], rng)
        eq_wide = conditioned_equity(mc, {2: wide}, hero, [], [2], rng)
        assert eq_wide > eq_tight

    def test_no_belief_falls_back(self):
        mc = MonteCarloEngine(n_simulations=300, rng=random.Random(3))
        rng = random.Random(0)
        hero = make_cards([("A", "s"), ("K", "s")])
        eq = conditioned_equity(mc, {}, hero, [], [2, 3], rng)
        assert 0.0 <= eq <= 1.0

    def test_tilt_deflates_p_fold(self):
        hmm = HMMBeliefState()
        for _ in range(15):
            hmm.update("raise", 1, 1)
        assert hmm.p_tilted() > 0.5
        adjusted = tilt_adjusted_p_fold(0.6, {2: hmm}, [2])
        assert adjusted < 0.6


# ===========================================================================
# SessionResult belief traces
# ===========================================================================

class TestBeliefTraces:
    def test_traces_populated(self):
        rng = random.Random(5)
        mc = MonteCarloEngine(n_simulations=120, rng=rng)
        players = [
            BotPlayer(i, f"B{i}", 1000, 0.3, 0.5, mc, rng,
                      belief_state=DynamicBeliefState())
            for i in range(1, 4)
        ]
        result = simulate_session(players, n_hands=20, seed=5)
        assert len(result.opponent_belief_traces) > 0
        df = belief_trace_dataframe(result)
        assert set(df.columns) == {
            "hand_number", "player_id", "posterior_mean", "p_tilted"}
        assert df["posterior_mean"].between(0.0, 1.0).all()


# ===========================================================================
# Chip conservation with each adaptive model
# ===========================================================================

class TestSessionIntegrity:
    @pytest.mark.parametrize("factory", [
        lambda: DynamicBeliefState(),
        lambda: HMMBeliefState(),
        lambda: ParticleBelief(n_particles=60),
    ])
    def test_chip_conservation(self, factory):
        rng = random.Random(6)
        mc = MonteCarloEngine(n_simulations=120, rng=rng)
        players = [
            BotPlayer(i, f"B{i}", 1000, 0.3, 0.5, mc, rng,
                      belief_state=factory())
            for i in range(1, 4)
        ]
        engine = GameEngine(players, 10, 20, verbose=False, rng=rng)
        total = sum(p.stack for p in players)
        for _ in range(25):
            if sum(1 for p in players if p.stack > 0) < 2:
                break
            engine.play_hand()
        assert sum(p.stack for p in players) == total


# ===========================================================================
# PnL -> tilt belief feed (hand-boundary realised-PnL transition)
# ===========================================================================

class TestPnLBeliefFeed:
    def test_loss_raises_p_tilted(self):
        """A realised loss fed via observe_pnl pushes the regime toward tilted."""
        hmm = HMMBeliefState()  # default use_pnl=True
        before = hmm.p_tilted()
        for _ in range(4):
            hmm.observe_pnl(-200)   # lost 200 of a 1000 stack each hand
        assert hmm.p_tilted() > before + 0.3

    def test_win_recovers_toward_normal(self):
        """A winning hand only applies the recovery drift (toward normal)."""
        hmm = HMMBeliefState()
        for _ in range(4):
            hmm.observe_pnl(-200)   # tilt it up first
        tilted = hmm.p_tilted()
        for _ in range(8):
            hmm.observe_pnl(+200)   # winning hands -> recover
        assert hmm.p_tilted() < tilted

    def test_transition_stays_normalised(self):
        """The 2x2 regime transition keeps pi a probability vector."""
        hmm = HMMBeliefState()
        for d in (-300, +50, -120, 0, -900, +400):
            hmm.observe_pnl(d)
            assert hmm.pi_normal + hmm.pi_tilted == pytest.approx(1.0)
            assert 0.0 <= hmm.p_tilted() <= 1.0

    def test_use_pnl_false_is_inert(self):
        """Ablation: use_pnl=False reproduces the dormant (no-PnL) detector."""
        hmm = HMMBeliefState(use_pnl=False)
        before = hmm.p_tilted()
        for _ in range(6):
            hmm.observe_pnl(-500)
        assert hmm.p_tilted() == before

    def test_static_belief_observe_pnl_is_noop(self):
        """The static / Beta models have no tilt regime: observe_pnl is a no-op."""
        b = BeliefState()
        b.observe_pnl(-500)
        assert b.p_tilted() == 0.0  # unchanged

    def test_calibrated_to_opponent_trigger(self):
        """
        With a shared TiltTrigger, one observe_pnl predict-step reproduces the
        opponent's own P(normal->tilted) exactly (the belief is calibrated to the
        true transition, not a heuristic).
        """
        trig = TiltTrigger(epsilon=0.04, kappa=1.5, stack0=1000)
        hmm = HMMBeliefState(prior_tilted=0.0, recover=0.0, tilt_trigger=trig)
        delta = -200  # loss fraction 0.2 -> p_nt = 0.04 + 1.5*0.2 = 0.34
        hmm.observe_pnl(delta)
        assert hmm.p_tilted() == pytest.approx(trig.transition_prob(delta))

    def test_engine_feeds_realised_delta(self):
        """
        The engine delivers each opponent's realised per-hand chip delta to the
        other player's belief at the hand boundary (and stays chip-neutral).
        """
        rng = random.Random(11)
        mc = MonteCarloEngine(n_simulations=100, rng=rng)
        hero = BotPlayer(1, "Hero", 1000, 0.3, 0.5, mc, rng,
                         belief_state=HMMBeliefState())
        villain = BotPlayer(2, "Villain", 1000, 0.3, 0.5, mc, rng)

        captured = []
        orig = hero.observe_hand_result
        hero.observe_hand_result = lambda obs: (captured.append(dict(obs)),
                                                orig(obs))[1]

        engine = GameEngine([hero, villain], 10, 20, verbose=False, rng=rng)
        total = sum(p.stack for p in (hero, villain))
        v_before = villain.stack
        engine.play_hand()

        assert sum(p.stack for p in (hero, villain)) == total  # chip-neutral
        assert len(captured) == 1                              # one opponent
        obs = captured[0]
        assert obs["player_id"] == 2
        assert obs["delta_stack"] == villain.stack - v_before  # realised PnL

    def test_pnl_feed_leads_aggression_only_detector(self):
        """
        Headline: against a real tilting opponent, the PnL-fed belief reaches a
        higher mean p_tilted than the aggression-emission-only belief over the
        same seeded match (earlier / stronger detection).
        """
        from src.adaptive_agent import AdaptiveBotPlayer

        def run(use_pnl):
            rng = random.Random(7)
            mc = MonteCarloEngine(n_simulations=100, rng=rng)
            hero = BotPlayer(1, "Hero", 1000, 0.2, 0.5, mc, rng,
                             belief_state=HMMBeliefState(
                                 mu_normal=0.25, mu_tilted=0.92, recover=0.05,
                                 use_pnl=use_pnl))
            villain = AdaptiveBotPlayer(2, "Tilt", 1000, mode="tilt",
                                        mc_engine=mc, rng=rng)
            engine = GameEngine([hero, villain], 10, 20, verbose=False, rng=rng)
            ps = []
            for _ in range(60):
                if sum(1 for p in (hero, villain) if p.stack > 0) < 2:
                    break
                engine.play_hand()
                ps.append(hero.belief_state.p_tilted())
            return sum(ps) / len(ps)

        assert run(use_pnl=True) > run(use_pnl=False)

    def test_pnl_feed_fires_once_per_hand_in_3plus_player(self):
        """In 3+ player the engine notifies once per opponent, but the single
        belief's observe_pnl must fire AT MOST once per hand (deduped) — else the
        regime transition compounds N-1 times."""
        rng = random.Random(3)
        mc = MonteCarloEngine(n_simulations=100, rng=rng)
        hmm = HMMBeliefState()
        calls = []
        orig = hmm.observe_pnl
        hmm.observe_pnl = lambda d: (calls.append(d), orig(d))[1]
        hero = BotPlayer(1, "Hero", 1000, 0.3, 0.5, mc, rng, belief_state=hmm)
        villains = [BotPlayer(i, f"V{i}", 1000, 0.3, 0.5, mc, rng) for i in (2, 3)]
        engine = GameEngine([hero] + villains, 10, 20, verbose=False, rng=rng)
        engine.play_hand()
        assert len(calls) == 1   # one transition for the hand, not 2 opponents
