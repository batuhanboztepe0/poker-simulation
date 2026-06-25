"""
big_leduc.py
------------
v2 Phase 2 scale-up: a PARAMETERISED R-rank Leduc Hold'em, so we can grow the game
until tabular CFR is infeasible and demonstrate that a neural method still learns
there (measured by a sampled LBR lower bound, since exact NashConv is also
infeasible at scale).

Same structure as the verified 3-rank Leduc (`leduc_cfr`): R ranks × 2 suits, ante
1, round-1 bet 2, round-2 bet 4, max 2 raises, pair-beats-higher showdown. The ONLY
generalisation is the number of ranks. Ranks are encoded as ordered uppercase
characters 'A','B',... so each card is a single character (the info-set key format
and `leduc_cfr._avail_from_key` keep working for any R) and ordered comparison
equals rank order (so `leduc_cfr._showdown` works unchanged). Uppercase avoids any
collision with the lowercase betting-history characters 'c'/'r'/'f'.

`BigLeducCFR` SUBCLASSES `LeducCFR` and overrides only the three deck-enumerating
methods (`train`, `exploitability`, `_best_response_value`); the correctness-critical
traversal, best-response policy iteration, and node/strategy logic are inherited
unchanged and were independently verified at R=3. `test_big_leduc.py` re-checks that
R=3 reproduces the exact 3-rank Leduc game value and exploitability, anchoring the
generalisation before it is scaled.

Pure analysis, side-effect free; imported nowhere in the engine.
"""

from src.leduc_cfr import LeducCFR


def rank_char(r):
    """Rank index -> ordered single-character card label ('A','B','C',...)."""
    return chr(ord("A") + r)


def char_rank(ch):
    """Card-label character -> rank index."""
    return ord(ch) - ord("A")


def deck(ranks):
    """Physical deck: two suits of each of `ranks` ranks (e.g. AABBCC at R=3)."""
    return [rank_char(r) for r in range(ranks) for _ in range(2)]


def num_deals(ranks):
    n = 2 * ranks
    return n * (n - 1) * (n - 2)


class BigLeducCFR(LeducCFR):
    """Vanilla CFR for R-rank Leduc. Inherits the verified `_cfr`, `_br_value`,
    `_br_gather`, `_info_key`, `Node` machinery from `LeducCFR`; only the deck and
    its enumeration (which were hard-coded to the 6-card deck) are overridden."""

    def __init__(self, ranks=3):
        super().__init__()
        self.ranks = ranks
        self.deck = deck(ranks)
        self.n = len(self.deck)

    def _deals(self):
        d = self.deck
        for i in range(self.n):
            for j in range(self.n):
                if j == i:
                    continue
                for k in range(self.n):
                    if k == i or k == j:
                        continue
                    yield d[i], d[j], d[k]

    def train(self, iterations):
        """`iterations` full-enumeration CFR passes over every deal; returns the
        running average game value to player 0."""
        total = 0.0
        nd = num_deals(self.ranks)
        for _ in range(iterations):
            for c0, c1, board in self._deals():
                total += self._cfr(c0, c1, board, "", None, 1.0, 1.0)
        return total / (iterations * nd)

    def exploitability(self):
        """NashConv averaged over all deals (0 iff exact equilibrium)."""
        return ((self._best_response_value(0) + self._best_response_value(1))
                / num_deals(self.ranks))

    def _best_response_value(self, br_player):
        """Exact best-response value over all deals via the inherited policy
        iteration (`_br_gather`/`_br_value`), with the parameterised deck."""
        from src.leduc_cfr import _avail_from_key
        br_action = {}
        for _sweep in range(self._BR_SWEEPS):
            q = {}
            for c0, c1, board in self._deals():
                self._br_gather(c0, c1, board, "", None, br_player, 1.0,
                                br_action, q)
            changed = False
            for info_set, qa in q.items():
                avail = _avail_from_key(info_set)
                best = max(avail, key=lambda a: qa[a])
                if br_action.get(info_set) != best:
                    br_action[info_set] = best
                    changed = True
            if not changed:
                break
        total = 0.0
        for c0, c1, board in self._deals():
            total += self._br_value(c0, c1, board, "", None, br_player, br_action)
        return total


def all_info_sets(ranks):
    """Every reachable info-set key for R-rank Leduc (one cheap CFR enumeration)."""
    cfr = BigLeducCFR(ranks)
    cfr.train(1)
    return sorted(cfr.nodes.keys())


def exploitability_of(strategy_table, ranks):
    """Exact NashConv of an arbitrary R-rank Leduc strategy (the parameterised
    analogue of `leduc_eval.exploitability_of`). Tractable only at small R; at the
    scaled R it is replaced by the LBR lower bound (`big_leduc_lbr`)."""
    from src.leduc_eval import _FixedNode
    cfr = BigLeducCFR(ranks)
    cfr.nodes = {iset: _FixedNode(dist) for iset, dist in strategy_table.items()}
    return cfr.exploitability()
