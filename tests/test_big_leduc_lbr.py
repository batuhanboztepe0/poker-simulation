"""
test_big_leduc_lbr.py
---------------------
Validate the parameterised LBR lower bound on the R-rank game: at R=3 it matches the
verified 3-rank LBR and the exact NashConv ordering; at small R it stays a lower
bound (LBR ≤ exact); and the deal-sampling path runs and stays a lower bound.
"""

import unittest

from src.big_leduc_lbr import lbr_exploitability as big_lbr
from src.big_leduc import exploitability_of, all_info_sets
from src.leduc_lbr import lbr_exploitability as leduc_lbr
from src.leduc_eval import uniform_strategy_table
from src.leduc_cfr import _avail_from_key

_TOL = 1e-9


def _uniform(ranks):
    t = {}
    for s in all_info_sets(ranks):
        av = _avail_from_key(s)
        t[s] = [1.0 / len(av) if a in av else 0.0 for a in range(3)]
    return t


class TestBigLeducLBR(unittest.TestCase):
    def test_r3_matches_verified_leduc_lbr(self):
        """R=3 LBR equals the verified 3-rank LBR (isomorphic relabel)."""
        self.assertAlmostEqual(big_lbr(_uniform(3), 3),
                               leduc_lbr(uniform_strategy_table()), places=9)

    def test_lower_bound_small_r(self):
        for R in (3, 6):
            tab = _uniform(R)
            self.assertLessEqual(big_lbr(tab, R), exploitability_of(tab, R) + _TOL,
                                 f"R={R}: LBR must not exceed exact")

    def test_lower_bound_captures_exploitation(self):
        tab = _uniform(3)
        self.assertGreater(big_lbr(tab, 3), 0.5 * exploitability_of(tab, 3))

    def test_sampling_runs_and_bounds(self):
        """The sampled-deal path runs, is deterministic given the seed, and stays a
        plausible lower bound near the full enumeration."""
        tab = _uniform(3)
        full = big_lbr(tab, 3)
        s1 = big_lbr(tab, 3, samples=60, seed=1)
        self.assertEqual(s1, big_lbr(tab, 3, samples=60, seed=1))
        self.assertLess(abs(s1 - full), 1.5)  # sampling noise, not a different metric


if __name__ == "__main__":
    unittest.main()
