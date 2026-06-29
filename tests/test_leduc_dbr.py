"""
Tests for src/leduc_dbr.py (detect-then-exploit, PREREGISTRATION §15).

Validates the estimation building blocks and two endpoints: p=0 ignores the estimate
(recovers Nash, ~0 gain), and plentiful observations recover the exact RNR-on-true
result. The full multi-seed frontier lives in scripts/measure_dbr.py.
"""
import pytest

from src.leduc_cfr import FOLD, CALL, RAISE
from src.leduc_rnr import station, uniform, ev_player0, nash_strategy, LeducRNR
from src.leduc_dbr import (
    observe, sigma_hat_fn, detect_then_exploit, loose_passive,
)


@pytest.fixture(scope="module")
def nash():
    fn, _ = nash_strategy(2000)
    return fn


def test_sigma_hat_valid_distribution():
    """sigma_hat is a proper distribution over the legal actions, and an unseen
    info-set falls back to the uniform prior."""
    counts = {"Qcr": [0, 3, 1]}                       # 3 calls, 1 raise observed
    sh = sigma_hat_fn(counts, alpha=1.0)
    d = sh("Qcr", [FOLD, CALL, RAISE])
    assert abs(sum(d) - 1.0) < 1e-9
    assert d[CALL] > d[RAISE] > d[FOLD] > 0           # smoothed toward the data
    # unseen info-set -> uniform over avail
    u = sh("never_seen", [CALL, RAISE])
    assert u[CALL] == pytest.approx(0.5) and u[RAISE] == pytest.approx(0.5)
    assert u[FOLD] == 0.0


def test_observe_station_is_deterministic(nash):
    """The station only ever calls, so its observed counts must be CALL-only."""
    counts = observe(station, 200, seed=0, reference=nash)
    assert counts, "expected some observed info-sets"
    for iset, c in counts.items():
        assert c[FOLD] == 0 and c[RAISE] == 0 and c[CALL] > 0, iset


def test_p0_ignores_estimate_and_recovers_nash(nash):
    """p=0 is Nash regardless of the estimate, so the realized gain over Nash is ~0
    even from few, noisy observations."""
    base = ev_player0(nash, loose_passive)
    r = detect_then_exploit(loose_passive, 20, seed=1, reference=nash, p=0.0, iters=400)
    assert abs(r["ev_vs_true"] - base) < 0.05


def test_large_N_recovers_exact_ceiling(nash):
    """With many observations the estimate approaches the true opponent, so the
    realized gain approaches the exact ceiling (RNR handed the true opponent)."""
    base = ev_player0(nash, station)
    ceil_solver = LeducRNR(station, 1.0); ceil_solver.train(600)
    ceiling = ev_player0(ceil_solver.counter_fn(), station) - base
    r = detect_then_exploit(station, 4000, seed=0, reference=nash, p=1.0, iters=600)
    realized = r["ev_vs_true"] - base
    assert realized > 0
    assert abs(realized - ceiling) < 0.15
