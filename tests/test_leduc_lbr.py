"""
test_leduc_lbr.py
-----------------
Validate the Local Best Response (LBR) lower bound on Leduc against the EXACT
NashConv metric. The defining guarantee — LBR is a LOWER bound, never an upper
bound — is checked on policies spanning far-from-Nash to near-equilibrium. This is
the Phase 1 validation that licenses using LBR on a larger game where exact
exploitability is infeasible.
"""

import unittest

from src.leduc_lbr import lbr_exploitability
from src.leduc_eval import exploitability_of, uniform_strategy_table
from src.leduc_cfr import LeducCFR

_TOL = 1e-9


def _cfr_table(iters):
    cfr = LeducCFR()
    cfr.train(iters)
    return cfr.strategy_table()


class TestLeducLBR(unittest.TestCase):
    def test_lbr_is_a_lower_bound(self):
        """LBR ≤ exact NashConv on every policy — the core guarantee."""
        tables = {
            "uniform": uniform_strategy_table(),
            "cfr_early": _cfr_table(20),
            "cfr_mid": _cfr_table(200),
        }
        for name, tab in tables.items():
            lbr = lbr_exploitability(tab)
            exact = exploitability_of(tab)
            self.assertLessEqual(
                lbr, exact + _TOL,
                f"{name}: LBR {lbr:.4f} must not exceed exact {exact:.4f}")

    def test_lbr_captures_real_exploitation_far_from_nash(self):
        """For the uniform strategy (far from Nash) LBR must find substantial
        exploitation — a useful, non-vacuous lower bound, not ~0."""
        uni = uniform_strategy_table()
        lbr = lbr_exploitability(uni)
        exact = exploitability_of(uni)
        self.assertGreater(lbr, 0.5 * exact,
                           f"LBR {lbr:.4f} should capture a large share of exact "
                           f"{exact:.4f} for the uniform policy")

    def test_lbr_near_zero_for_near_equilibrium(self):
        """Against a converged CFR average (near Nash) LBR finds almost no
        exploitation, so its lower bound sits near 0 (it may be marginally
        negative — the honest meaning of 'no profitable deviation found')."""
        tab = _cfr_table(2000)
        lbr = lbr_exploitability(tab)
        self.assertLess(lbr, 0.1,
                        f"LBR {lbr:.4f} should be near 0 for a near-equilibrium")
        self.assertLessEqual(lbr, exploitability_of(tab) + _TOL)

    def test_lbr_deterministic(self):
        """LBR is a pure function of the strategy table (no RNG)."""
        uni = uniform_strategy_table()
        self.assertEqual(lbr_exploitability(uni), lbr_exploitability(uni))


if __name__ == "__main__":
    unittest.main()
