"""
big_leduc_nfsp.py
-----------------
v2 Phase 2 scale-up: neural NFSP on the parameterised R-rank Leduc (`big_leduc`).

Identical algorithm to `leduc_neural_nfsp.LeducNeuralNFSP` (DQN best-response beta +
supervised average-policy Pi, mixed by eta) — generalised so the featurizer and the
self-play loop work for any number of ranks. The feature is R+R+1+8+6 = 2R+15 dims
(private-card one-hot, board one-hot, round flag, and the two betting histories,
which are R-independent since the betting structure does not change with ranks). The
network is a fixed-width MLP, so training cost per episode is independent of R — the
property that lets it scale where tabular CFR's deal enumeration (~R^3) does not.

A separate module from `leduc_neural_nfsp` so that the verified 3-rank module and its
committed result are untouched; the algorithm is intentionally the same.

OFF by default, imported nowhere in the engine. Requires torch (optional dep).
"""

import random
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

from src.leduc_cfr import NUM_ACTIONS, _available_actions, _is_terminal, _avail_from_key
from src.leduc_q import _ACT_CHAR, _payoff0
from src.big_leduc import deck, char_rank, all_info_sets

_MAX_R1, _MAX_R2 = 4, 3
_NEG = -1e9


def feat_dim(ranks):
    return 2 * ranks + 1 + _MAX_R1 * 2 + _MAX_R2 * 2


def _onehot(i, n):
    v = [0.0] * n
    if i is not None:
        v[i] = 1.0
    return v


def _enc_hist(h, maxlen):
    out = []
    for t in range(maxlen):
        if t < len(h):
            out += [1.0 if h[t] == "c" else 0.0, 1.0 if h[t] == "r" else 0.0]
        else:
            out += [0.0, 0.0]
    return out


def featurize(info_set, ranks):
    """R-rank Leduc info-set key -> fixed (2R+15)-d feature vector."""
    if "/" in info_set:
        s = info_set.index("/")
        card, board = char_rank(info_set[0]), char_rank(info_set[1])
        r1, r2, is_r2 = info_set[2:s], info_set[s + 1:], 1.0
    else:
        card, board = char_rank(info_set[0]), None
        r1, r2, is_r2 = info_set[1:], "", 0.0
    return (_onehot(card, ranks) + _onehot(board, ranks) + [is_r2]
            + _enc_hist(r1, _MAX_R1) + _enc_hist(r2, _MAX_R2))


def _mlp(din, hidden):
    return nn.Sequential(nn.Linear(din, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, NUM_ACTIONS))


def _avail_mask(avail):
    return [1.0 if a in avail else 0.0 for a in range(NUM_ACTIONS)]


class BigLeducNeuralNFSP:
    """Neural NFSP on R-rank Leduc (see leduc_neural_nfsp for the algorithm)."""

    def __init__(self, ranks=3, hidden=64, eta=0.1, eps_start=0.06, eps_end=0.0,
                 gamma=1.0, lr_rl=0.01, lr_sl=0.01, batch=128,
                 rl_capacity=200_000, sl_capacity=1_000_000,
                 target_update=1000, min_buffer=1000, seed=0):
        if not _HAVE_TORCH:
            raise ImportError("BigLeducNeuralNFSP requires torch (optional dep)")
        torch.manual_seed(seed)
        self.ranks = ranks
        self.deck = deck(ranks)
        self.din = feat_dim(ranks)
        self.rng = random.Random(seed)
        self.eta, self.eps_start, self.eps_end = eta, eps_start, eps_end
        self.gamma, self.batch = gamma, batch
        self.target_update, self.min_buffer = target_update, min_buffer
        self.q = _mlp(self.din, hidden)
        self.q_target = _mlp(self.din, hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.pi = _mlp(self.din, hidden)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=lr_rl)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr_sl)
        self.m_rl = deque(maxlen=rl_capacity)
        self.m_sl, self.sl_capacity, self._sl_seen = [], sl_capacity, 0
        self._eps, self._steps = eps_start, 0

    def _feat(self, info_set):
        return featurize(info_set, self.ranks)

    def _q_action(self, feat, avail):
        if self.rng.random() < self._eps:
            return self.rng.choice(avail)
        with torch.no_grad():
            qv = self.q(torch.tensor(feat).float()).tolist()
        return max(avail, key=lambda a: qv[a])

    def _pi_action(self, feat, avail):
        with torch.no_grad():
            logits = self.pi(torch.tensor(feat).float()).tolist()
        masked = [logits[a] if a in avail else _NEG for a in range(NUM_ACTIONS)]
        m = max(masked)
        import math
        exps = [math.exp(x - m) for x in masked]
        tot = sum(exps)
        r = self.rng.random() * tot
        upto = 0.0
        for a in range(NUM_ACTIONS):
            upto += exps[a]
            if r < upto:
                return a
        return avail[-1]

    def _play(self, c0, c1, board, modes):
        cards, r1, r2, decisions = (c0, c1), "", None, []
        while True:
            in_r2 = r2 is not None
            curr = r2 if in_r2 else r1
            if _is_terminal(curr, in_r2):
                if curr[-1] == "f" or in_r2:
                    break
                r2 = ""
                continue
            player = len(curr) % 2
            info_set = (str(cards[player]) + str(board) + r1 + "/" + r2
                        if in_r2 else str(cards[player]) + curr)
            avail = _available_actions(curr, in_r2)
            feat = self._feat(info_set)
            a = (self._q_action(feat, avail) if modes[player]
                 else self._pi_action(feat, avail))
            decisions.append((player, feat, a, avail, modes[player]))
            if in_r2:
                r2 = curr + _ACT_CHAR[a]
            else:
                r1 = curr + _ACT_CHAR[a]
        return decisions, _payoff0(c0, c1, board, r1, r2)

    def _store(self, decisions, payoff0):
        for p in (0, 1):
            seq = [(f, a, av, b) for (pl, f, a, av, b) in decisions if pl == p]
            reward = payoff0 if p == 0 else -payoff0
            for i, (f, a, av, b) in enumerate(seq):
                if i + 1 < len(seq):
                    self.m_rl.append((f, a, 0.0, seq[i + 1][0],
                                      _avail_mask(seq[i + 1][2]), 0.0))
                else:
                    self.m_rl.append((f, a, reward, [0.0] * self.din,
                                      [0.0] * NUM_ACTIONS, 1.0))
                if b:
                    entry = (f, a, _avail_mask(av))
                    self._sl_seen += 1
                    if len(self.m_sl) < self.sl_capacity:
                        self.m_sl.append(entry)
                    else:
                        j = self.rng.randint(0, self._sl_seen - 1)
                        if j < self.sl_capacity:
                            self.m_sl[j] = entry

    def _learn_q(self):
        if len(self.m_rl) < self.min_buffer:
            return
        b = self.rng.sample(self.m_rl, self.batch)
        feat = torch.tensor([x[0] for x in b]).float()
        act = torch.tensor([x[1] for x in b]).long()
        rew = torch.tensor([x[2] for x in b]).float()
        nfeat = torch.tensor([x[3] for x in b]).float()
        nmask = torch.tensor([x[4] for x in b]).float()
        done = torch.tensor([x[5] for x in b]).float()
        with torch.no_grad():
            nq = self.q_target(nfeat) + (nmask - 1.0) * 1e9
            target = rew + self.gamma * (1.0 - done) * nq.max(dim=1).values
        qv = self.q(feat).gather(1, act.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(qv, target)
        self.opt_q.zero_grad(); loss.backward(); self.opt_q.step()

    def _learn_pi(self):
        if len(self.m_sl) < self.min_buffer:
            return
        b = self.rng.sample(self.m_sl, self.batch)
        feat = torch.tensor([x[0] for x in b]).float()
        act = torch.tensor([x[1] for x in b]).long()
        mask = torch.tensor([x[2] for x in b]).float()
        logits = self.pi(feat) + (mask - 1.0) * 1e9
        loss = F.cross_entropy(logits, act)
        self.opt_pi.zero_grad(); loss.backward(); self.opt_pi.step()

    def train(self, episodes, eval_every=None, eval_hook=None):
        history = []
        for ep in range(episodes):
            frac = ep / max(1, episodes - 1)
            self._eps = self.eps_start + (self.eps_end - self.eps_start) * frac
            i, j, k = self.rng.sample(range(len(self.deck)), 3)
            modes = {p: (self.rng.random() < self.eta) for p in (0, 1)}
            decisions, payoff0 = self._play(self.deck[i], self.deck[j],
                                            self.deck[k], modes)
            self._store(decisions, payoff0)
            self._learn_q()
            self._learn_pi()
            self._steps += 1
            if self._steps % self.target_update == 0:
                self.q_target.load_state_dict(self.q.state_dict())
            if eval_every and eval_hook and (ep + 1) % eval_every == 0:
                history.append(eval_hook(ep + 1, self))
        return history

    def average_strategy_table(self):
        keys = all_info_sets(self.ranks)
        feats = torch.tensor([self._feat(s) for s in keys]).float()
        with torch.no_grad():
            logits = self.pi(feats)
        import math
        table = {}
        for idx, s in enumerate(keys):
            av = _avail_from_key(s)
            row = logits[idx].tolist()
            m = max(row[a] for a in av)
            exps = [math.exp(row[a] - m) if a in av else 0.0
                    for a in range(NUM_ACTIONS)]
            tot = sum(exps)
            table[s] = [e / tot if tot > 0 else (1.0 / len(av) if a in av else 0.0)
                        for a, e in enumerate(exps)]
        return table
