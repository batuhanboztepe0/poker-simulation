"""
test_big_leduc_nfsp.py
----------------------
Light smoke test for neural NFSP on the parameterised R-rank game: the feature
dimension is 2R+15, training runs, and the average policy is a valid masked
distribution over every info-set. Skipped when torch is absent.
"""

import unittest

import pytest

from src import big_leduc_nfsp as bn
from src.big_leduc import all_info_sets
from src.leduc_cfr import _avail_from_key, NUM_ACTIONS


class TestBigLeducNFSP(unittest.TestCase):
    def test_feat_dim(self):
        self.assertEqual(bn.feat_dim(3), 21)
        self.assertEqual(bn.feat_dim(6), 27)

    @pytest.mark.skipif(not bn._HAVE_TORCH, reason="torch not installed")
    def test_trains_and_emits_valid_policy(self):
        m = bn.BigLeducNeuralNFSP(ranks=3, seed=0)
        m.train(300)
        table = m.average_strategy_table()
        keys = all_info_sets(3)
        self.assertEqual(set(table), set(keys))
        for s, row in table.items():
            av = _avail_from_key(s)
            self.assertAlmostEqual(sum(row), 1.0, places=5)
            for a in range(NUM_ACTIONS):
                if a not in av:
                    self.assertEqual(row[a], 0.0)


if __name__ == "__main__":
    unittest.main()
