"""
test_leduc_rnr.py
-----------------
v2 Phase C3: Restricted Nash Response on Leduc (PREREGISTRATION.md §14).

Regression-tests the exactness of the RNR frontier at modest iteration counts:
the opponent strategies are valid, p=0 reproduces Nash (gain ~0, exploitability
~0), p=1 reproduces the independently-computed exact best response (large gain),
EV is monotone in p, and exploitability rises with p (the tradeoff). Exact metrics,
no sampling, so the only looseness is CFR convergence at the small test iteration
count.
"""

import unittest

from src.leduc_cfr import LeducCFR, NUM_ACTIONS, FOLD, CALL, RAISE
from src.leduc_rnr import (
    LeducRNR, nash_strategy, ev_player0, station, maniac, uniform, OPPONENTS,
    _avail_from_key,
)

ITERS = 500   # small for test speed; exact metrics, so endpoints still track


def _br_exact_vs(opp_fn, iters=1):
    """Exact best-response EV for player 0 against a fixed opponent, via the
    load-trick: set every node's average to opp, then best-respond player 0."""
    cfr = LeducCFR(); cfr.train(iters)
    for iset, node in cfr.nodes.items():
        av = _avail_from_key(iset)
        dist = opp_fn(iset, av)
        node.strategy_sum = [dist[a] for a in range(NUM_ACTIONS)]
    return cfr._best_response_value(0) / 120.0


class TestOpponentStrategies(unittest.TestCase):
    def test_valid_distributions(self):
        for av in ([CALL, RAISE], [FOLD, CALL, RAISE], [FOLD, CALL]):
            for fn in (station, maniac, uniform):
                d = fn("dummy", av)
                self.assertEqual(len(d), NUM_ACTIONS)
                self.assertAlmostEqual(sum(d), 1.0, places=9)
                for a in range(NUM_ACTIONS):
                    if a not in av:
                        self.assertEqual(d[a], 0.0)
                    self.assertGreaterEqual(d[a], 0.0)

    def test_archetypes(self):
        self.assertEqual(station("x", [FOLD, CALL, RAISE])[CALL], 1.0)   # never folds/raises
        self.assertEqual(maniac("x", [FOLD, CALL, RAISE])[RAISE], 1.0)   # raises when able
        self.assertEqual(maniac("x", [FOLD, CALL])[CALL], 1.0)           # else calls


class TestRNRFrontier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nash_fn, cls.nash_cfr = nash_strategy(ITERS)
        cls.opp = station
        cls.ev_nash = ev_player0(cls.nash_fn, cls.opp)
        cls.br_exact = _br_exact_vs(cls.opp)
        cls.frontier = {}
        for p in (0.0, 0.5, 1.0):
            r = LeducRNR(cls.opp, p); r.train(ITERS)
            cls.frontier[p] = (ev_player0(r.counter_fn(), cls.opp),
                               r.counter_exploitability())

    def test_p0_reproduces_nash(self):
        ev0, ex0 = self.frontier[0.0]
        self.assertAlmostEqual(ev0, self.ev_nash, places=6)   # p=0 IS standard CFR
        self.assertLess(abs(ex0), 0.1)                        # ~unexploitable

    def test_p1_matches_exact_best_response(self):
        ev1, _ = self.frontier[1.0]
        self.assertAlmostEqual(ev1, self.br_exact, delta=0.05)

    def test_reliable_positive_gain(self):
        ev1, _ = self.frontier[1.0]
        self.assertGreater(ev1 - self.ev_nash, 0.5)           # large exact EV gain over Nash

    def test_ev_monotone_in_p(self):
        ev0 = self.frontier[0.0][0]
        ev_half = self.frontier[0.5][0]
        ev1 = self.frontier[1.0][0]
        self.assertLessEqual(ev0, ev_half + 1e-9)
        self.assertLessEqual(ev_half, ev1 + 1e-9)

    def test_exploitability_tradeoff(self):
        # the price of exploitation: p=1 is far more exploitable than p=0
        self.assertGreater(self.frontier[1.0][1], self.frontier[0.0][1])


if __name__ == "__main__":
    unittest.main()
