"""
leduc_nfsp.py
-------------
Tabular Neural Fictitious Self-Play (NFSP; Heinrich & Silver 2016, references.md
§1) on Leduc Hold'em — the policy-AVERAGING learner that fixes the exact
non-convergence the greedy value learner in `leduc_q.LeducQLearner` (the DQN
regime) exhibits on this very game.

It is deliberately the SAME value learner with ONE thing added. `LeducNFSP`
reuses `leduc_q`'s Q table, epsilon-greedy behaviour, and TD update verbatim
(same alpha, eps, gamma) and adds only NFSP's second policy:

  - beta  = epsilon-greedy over Q  (the best response; the off-policy TD learner)
  - Pi    = the empirical action distribution of beta's OWN past moves
            (fictitious-play averaging; the supervised "average-policy network"
            in the neural version).

Each seat, each episode, follows beta with probability `eta` (the anticipatory
parameter) and Pi otherwise. Transitions always update Q (the RL memory M_RL);
`(info_set, action)` pairs are recorded into Pi ONLY on beta episodes (the
supervised memory M_SL). For tabular Leduc the reservoir + classifier reduces
exactly to per-info-set action counts whose normalisation is Pi.

The point (the loop THESIS §4/§6 leaves open): CFR's time-AVERAGE reaches Nash
but the greedy LAST-ITERATE — the DQN regime, measured directly by `leduc_q` —
does not (it oscillates around 1.15 and never converges). Holding alpha, eps and
gamma at `leduc_q`'s values and adding ONLY this averaging, the AVERAGE policy
converges instead: `average_strategy_table()` feeds Pi to
`leduc_eval.exploitability_of` — the same exact metric on which the greedy
learner fails — so the comparison isolates the single variable (averaging) and
success/failure are reported on identical, non-tunable footing.

Pure-Python, seeded from a single seed, offline analysis only (touches no engine,
training, or baseline behaviour). Reuses the verified game logic and TD update in
`leduc_q` rather than duplicating the correctness-critical payoff code.
"""

from src.leduc_cfr import (
    NUM_ACTIONS, _available_actions, _is_terminal, _avail_from_key,
)
from src.leduc_q import (
    LeducQLearner, _payoff0, _DECK, _ACT_CHAR, _all_info_sets,
)


class LeducNFSP:
    """Tabular NFSP self-play: `leduc_q`'s epsilon-greedy Q best-response (beta)
    plus a counts-based average policy (Pi), mixed per seat per episode by `eta`.
    alpha/eps/gamma default to `leduc_q`'s values so the only difference from the
    non-converging greedy learner is the averaging itself."""

    def __init__(self, alpha=0.1, eps=0.1, eta=0.1, gamma=1.0, seed=0):
        # beta: the value-based learner reused verbatim from leduc_q; its rng is
        # shared so the WHOLE run reproduces from this one seed.
        self.ql = LeducQLearner(alpha=alpha, eps=eps, gamma=gamma, seed=seed)
        self.rng = self.ql.rng
        self.eta = eta
        self.avg_counts = {}            # info_set -> [c_fold, c_call, c_raise]

    def _row_counts(self, info_set):
        row = self.avg_counts.get(info_set)
        if row is None:
            row = [0.0, 0.0, 0.0]
            self.avg_counts[info_set] = row
        return row

    def _pi_action(self, info_set, avail):
        """Sample an action from the current AVERAGE policy Pi at `info_set`
        (uniform over available actions where Pi has no mass yet)."""
        row = self.avg_counts.get(info_set)
        total = sum(row[a] for a in avail) if row is not None else 0.0
        if total <= 0:
            return self.rng.choice(avail)
        r = self.rng.random() * total
        upto = 0.0
        for a in avail:
            upto += row[a]
            if r < upto:
                return a
        return avail[-1]

    def _play(self, c0, c1, board, modes):
        """Play one hand; seat `p` follows beta (eps-greedy Q) when `modes[p]` is
        True, else Pi. Returns (decisions, payoff0) with decisions an ordered list
        of (player, info_set, action) — the same shape `leduc_q._update` expects."""
        cards, r1, r2, decisions = (c0, c1), "", None, []
        while True:
            in_r2 = r2 is not None
            curr = r2 if in_r2 else r1
            if _is_terminal(curr, in_r2):
                if curr[-1] == "f" or in_r2:
                    break                       # fold or round-2 showdown ends it
                r2 = ""                          # round-1 betting closed -> round 2
                continue
            player = len(curr) % 2
            info_set = (str(cards[player]) + str(board) + r1 + "/" + r2
                        if in_r2 else str(cards[player]) + curr)
            avail = _available_actions(curr, in_r2)
            if modes[player]:
                a = self.ql._eps_greedy(info_set, avail)
            else:
                a = self._pi_action(info_set, avail)
            decisions.append((player, info_set, a))
            if in_r2:
                r2 = curr + _ACT_CHAR[a]
            else:
                r1 = curr + _ACT_CHAR[a]
        return decisions, _payoff0(c0, c1, board, r1, r2)

    def train(self, episodes):
        for _ in range(episodes):
            i, j, k = self.rng.sample(range(6), 3)
            modes = {p: (self.rng.random() < self.eta) for p in (0, 1)}
            decisions, payoff0 = self._play(_DECK[i], _DECK[j], _DECK[k], modes)
            self.ql._update(decisions, payoff0)         # M_RL: off-policy TD on Q
            for (player, info_set, a) in decisions:     # M_SL: record beta moves
                if modes[player]:
                    self._row_counts(info_set)[a] += 1.0

    def average_strategy_table(self):
        """The AVERAGE (fictitious-play) policy over EVERY Leduc info-set as a
        {info_set: [p_fold, p_call, p_raise]} table for `exploitability_of`;
        uniform at info-sets with no recorded best-response action yet."""
        table = {}
        for s in _all_info_sets():
            av = _avail_from_key(s)
            row = self.avg_counts.get(s)
            total = sum(row[a] for a in av) if row is not None else 0.0
            probs = [0.0] * NUM_ACTIONS
            if total <= 0:
                for a in av:
                    probs[a] = 1.0 / len(av)
            else:
                for a in av:
                    probs[a] = row[a] / total
            table[s] = probs
        return table
