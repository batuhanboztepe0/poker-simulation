"""
test_real_data_tilt.py
----------------------
Tests for the real-data tilt validation (src/real_data_tilt.py).

The analysis logic (sequence building, the phenomenon / detector / regime tests)
is exercised on SYNTHETIC sequences so it is deterministic and needs no external
data. PHH parsing is checked on a tiny committed fixture of 6 real hands
(tests/fixtures/sample.phhs, from the PHH dataset, Kim 2024, CC-BY-4.0) and is
skipped if pokerkit is not installed.

These are additive tests for a new, offline, opt-in module; no existing engine,
training, or baseline behaviour is touched.
"""

import os

import pytest

from src.real_data_tilt import (build_sequences, phenomenon_test,
                                within_player_loss_vs_win,
                                detector_separation, fit_regime_hmm)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.phhs")


def _obs(player, t, n_aggr, n_dec, vpip, net_bb, table="T1", hand=0):
    return {"player": player, "table": table, "hand": hand,
            "ts": (2009, 7, 1, t, 0), "n_aggr": n_aggr, "n_dec": n_dec,
            "vpip": vpip, "pfr": bool(n_aggr) and vpip, "net_bb": net_bb}


def _tilt_session(player, post_rate, base_rate, n=24, n_dec=10):
    """A session where the hand AFTER a 20bb loss is aggressive at `post_rate`
    and the hand after a +5bb result is passive at `base_rate`. Even hands lose,
    odd hands (preceded by a loss) are the aggressive 'post-loss' hands."""
    sess = []
    for i in range(n):
        loss = (i % 2 == 0)
        rate = base_rate if loss else post_rate     # cur reflects PREV result
        net = -20.0 if loss else 5.0
        n_a = round(rate * n_dec)
        sess.append(_obs(player, i, n_a, n_dec, vpip=(n_a > 0),
                         net_bb=net, hand=i))
    return sess


# --------------------------------------------------------------------------
# PHH parsing (real fixture, pokerkit-dependent)
# --------------------------------------------------------------------------

def test_parse_fixture_chip_conservation():
    pytest.importorskip("pokerkit")
    from src.real_data_tilt import parse_phhs
    recs = parse_phhs(FIXTURE)
    assert recs, "fixture parsed to no records"
    by_hand = {}
    for r in recs:
        by_hand.setdefault(r["hand"], 0.0)
        by_hand[r["hand"]] += r["net_bb"]
    # Every hand conserves chips: per-hand net (in bb) sums to ~0.
    assert max(abs(v) for v in by_hand.values()) < 1e-6
    assert len(by_hand) == 6


def test_parse_fixture_record_invariants():
    pytest.importorskip("pokerkit")
    from src.real_data_tilt import parse_phhs
    recs = parse_phhs(FIXTURE)
    for r in recs:
        assert 0 <= r["n_aggr"] <= r["n_dec"]
        assert isinstance(r["vpip"], bool)
        assert not r["pfr"] or r["vpip"]            # PFR implies VPIP
        assert r["net_bb"] == r["net_bb"]           # finite (not NaN)
    assert any(r["n_aggr"] > 0 for r in recs)       # some aggression present


# --------------------------------------------------------------------------
# Sequence building
# --------------------------------------------------------------------------

def test_build_sequences_gap_split_and_minlen():
    # One player, one table, two sittings separated by a 2h gap (> max_gap_s).
    early = [_obs("p", t=100 + i, n_aggr=0, n_dec=1, vpip=False, net_bb=0.0)
             for i in range(25)]
    late = [_obs("p", t=8000 + i, n_aggr=0, n_dec=1, vpip=False, net_bb=0.0)
            for i in range(25)]
    sessions = build_sequences(early + late, min_len=20, max_gap_s=3600)
    assert len(sessions) == 2                        # split into two sittings

    # A 10-hand sitting is dropped at min_len=20.
    short = [_obs("q", t=i, n_aggr=0, n_dec=1, vpip=False, net_bb=0.0)
             for i in range(10)]
    assert build_sequences(short, min_len=20) == []


# --------------------------------------------------------------------------
# Test A — phenomenon + placebo
# --------------------------------------------------------------------------

def test_phenomenon_detects_injected_tilt():
    seqs = [_tilt_session(f"p{k}", post_rate=0.7 + 0.03 * k, base_rate=0.0)
            for k in range(8)]
    real = phenomenon_test(seqs, loss_bb=10.0)
    assert real["n_players"] == 8
    assert real["aggr"]["lo"] > 0                    # post-loss more aggressive
    assert real["vpip"]["lo"] > 0                    # ... and looser

    # Placebo: shuffling the post/baseline labels must kill the effect.
    plac = phenomenon_test(seqs, loss_bb=10.0, placebo_seed=7)
    assert not (plac["aggr"]["lo"] > 0)              # CI no longer excludes 0


def test_phenomenon_null_when_flat():
    # Aggression independent of the previous result -> ~0 effect.
    flat = []
    for k in range(8):
        sess = [_obs(f"q{k}", t=i, n_aggr=3, n_dec=10, vpip=True,
                     net_bb=(-20.0 if i % 2 == 0 else 5.0), hand=i)
                for i in range(24)]
        flat.append(sess)
    res = phenomenon_test(flat, loss_bb=10.0)
    assert abs(res["aggr"]["mean"]) < 1e-9
    assert not (res["aggr"]["lo"] > 0)


# --------------------------------------------------------------------------
# Test A' — symmetric within-player control (post-loss vs post-WIN)
# --------------------------------------------------------------------------

def _swing_session(player, after_loss_rate, after_win_rate, n=24, n_dec=10):
    """Even hands are big-swing events (alternating −20bb / +20bb); the odd hand
    right after each is the 'response', aggressive at `after_loss_rate` when the
    swing was a loss and `after_win_rate` when it was a win. Response hands net 0
    so they are not themselves classified as swings -> per player 6 post-loss and
    6 post-win response hands."""
    sess = []
    for i in range(n):
        if i % 2 == 0:
            swing_loss = ((i // 2) % 2 == 0)
            net, n_a = (-20.0 if swing_loss else 20.0), round(0.3 * n_dec)
        else:
            prev_loss = (((i - 1) // 2) % 2 == 0)
            net = 0.0
            n_a = round((after_loss_rate if prev_loss else after_win_rate) * n_dec)
        sess.append(_obs(player, i, n_a, n_dec, vpip=(n_a > 0),
                         net_bb=net, hand=i))
    return sess


def test_within_player_loss_vs_win_detects_asymmetry():
    seqs = [_swing_session(f"p{k}", after_loss_rate=0.7 + 0.02 * k,
                           after_win_rate=0.3) for k in range(8)]
    r = within_player_loss_vs_win(seqs, swing_bb=10.0, min_per_group=5)
    assert r["n_players"] == 8
    assert r["n_loss"] >= 40 and r["n_win"] >= 40
    assert r["aggr"]["lo"] > 0               # more aggressive after a loss than an equal win
    assert r["aggr_cohen_d"] is not None and r["aggr_cohen_d"] > 0

    # Placebo: permuting the loss/win labels within each player kills the effect.
    p = within_player_loss_vs_win(seqs, swing_bb=10.0, min_per_group=5,
                                  placebo_seed=7)
    assert not (p["aggr"]["lo"] > 0)


def test_within_player_loss_vs_win_null_when_symmetric():
    # Identical response after a loss and after an equal win -> no asymmetry,
    # even though both differ from the player's ordinary baseline (arousal, not
    # loss-aversion).
    seqs = [_swing_session(f"q{k}", after_loss_rate=0.5, after_win_rate=0.5)
            for k in range(8)]
    r = within_player_loss_vs_win(seqs, swing_bb=10.0, min_per_group=5)
    assert abs(r["aggr"]["mean"]) < 1e-9
    assert not (r["aggr"]["lo"] > 0)


# --------------------------------------------------------------------------
# Test B — the project's HMM detector (emission-only)
# --------------------------------------------------------------------------

def test_detector_separation_positive_then_placebo():
    seqs = [_tilt_session(f"p{k}", post_rate=0.8, base_rate=0.05)
            for k in range(8)]
    real = detector_separation(seqs, loss_bb=10.0)
    assert real["n_players"] == 8
    assert real["separation"]["mean"] > 0           # P(tilted) up post-loss

    plac = detector_separation(seqs, loss_bb=10.0, placebo_seed=7)
    assert abs(plac["separation"]["mean"]) < real["separation"]["mean"]


# --------------------------------------------------------------------------
# Test C — regime fit (hmmlearn-dependent)
# --------------------------------------------------------------------------

def _regime_sessions(plo, phi, blocks=4, blen=15, nsess=16):
    """Sessions that alternate between a low- and high-aggression regime
    (stochastic Bernoulli emissions, seeded for determinism)."""
    import random
    seqs = []
    for k in range(nsess):
        rng = random.Random(100 + k)
        sess, t, hi = [], 0, False
        for _ in range(blocks):
            p = phi if hi else plo
            for _ in range(blen):
                a = 1 if rng.random() < p else 0
                sess.append(_obs(f"p{k}", t=t, n_aggr=a, n_dec=1, vpip=bool(a),
                                 net_bb=(-20.0 if a else 5.0), hand=t))
                t += 1
            hi = not hi
        seqs.append(sess)
    return seqs


def test_regime_fit_prefers_two_states_on_regime_data():
    pytest.importorskip("hmmlearn")
    res = fit_regime_hmm(_regime_sessions(0.1, 0.9), seed=0)
    assert res["ok"] and res.get("two_state_found")
    assert res["bic_gain"] > 0                       # 2-state beats 1-state
    assert res["p_aggr_high"] > res["p_aggr_low"]


def test_regime_fit_rejects_two_states_on_iid_data():
    pytest.importorskip("hmmlearn")
    import random
    seqs = []
    for k in range(16):
        rng = random.Random(7 + k)
        sess = [_obs(f"q{k}", t=i, n_aggr=(1 if rng.random() < 0.3 else 0),
                     n_dec=1, vpip=False, net_bb=0.0, hand=i)
                for i in range(60)]
        seqs.append(sess)
    res = fit_regime_hmm(seqs, seed=0)
    assert res["ok"]
    assert not (res["bic_gain"] > 0)                 # no structure -> 1-state


def test_regime_fit_graceful_when_too_few_sessions():
    res = fit_regime_hmm([], seed=0)
    assert res["ok"] is False
