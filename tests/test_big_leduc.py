"""
test_big_leduc.py
-----------------
Anchor the parameterised R-rank Leduc against the verified 3-rank Leduc before it is
scaled: at R=3 the generalised game must reproduce the original game value and exact
exploitability bit-for-bit (it is the same game with relabelled ranks and the same
deal-enumeration order), and the info-set / deal counts must grow with R.
"""

import unittest

from src.leduc_cfr import LeducCFR
from src.big_leduc import BigLeducCFR, all_info_sets, num_deals, deck, char_rank


class TestBigLeduc(unittest.TestCase):
    def test_r3_reproduces_original_leduc(self):
        """R=3 == the verified 3-rank Leduc, bit-for-bit, at equal iterations."""
        orig = LeducCFR()
        big = BigLeducCFR(3)
        gv_o = orig.train(100)
        gv_b = big.train(100)
        self.assertAlmostEqual(gv_o, gv_b, places=9,
                               msg="R=3 game value must match the original Leduc")
        self.assertAlmostEqual(orig.exploitability(), big.exploitability(), places=9,
                               msg="R=3 exploitability must match the original Leduc")

    def test_r3_info_set_count(self):
        self.assertEqual(len(all_info_sets(3)), 288)

    def test_counts_scale_with_ranks(self):
        self.assertEqual(num_deals(3), 120)
        self.assertEqual(num_deals(6), 1320)
        self.assertGreater(len(all_info_sets(6)), len(all_info_sets(3)))

    def test_deck_and_encoding(self):
        self.assertEqual(deck(3), ["A", "A", "B", "B", "C", "C"])
        self.assertEqual(char_rank("A"), 0)
        self.assertEqual(char_rank("C"), 2)


if __name__ == "__main__":
    unittest.main()
