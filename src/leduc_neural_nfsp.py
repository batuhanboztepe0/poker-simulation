"""
leduc_neural_nfsp.py
--------------------
v2 Phase 2: NEURAL Fictitious Self-Play on Leduc Hold'em — the first neural
equilibrium method in the repo, validated by the SAME exact NashConv metric that
scores the tabular learners.

This is the neural counterpart of `leduc_nfsp.LeducNFSP` (tabular NFSP). The
algorithm (Heinrich & Silver 2016, NFSP; REFERENCES.md §1) is unchanged — a
best-response value learner (beta) plus a supervised average-policy learner (Pi),
mixed per seat per episode by the anticipatory parameter eta. What changes is the
function approximation: the per-info-set Q table and count table become two MLPs
that consume a STRUCTURED 21-d feature of the info-set (private card, board card,
round, and the two betting histories), so the method GENERALISES across info-sets
instead of tabulating them. That generalisation is the whole point of going neural:
it is what lets the same algorithm scale to games too large to tabulate. On Leduc
it is a correctness check — the exact exploitability of Pi must fall toward Nash,
on the identical, non-tunable metric (`leduc_eval.exploitability_of`) the tabular
NFSP / CFR / Q-learner are scored on, so the comparison is apples-to-apples.

Honest caveat: Leduc is small enough that the TABULAR learners are near-optimal, so
neural function approximation is not expected to beat them HERE; its advantage is
scalability, demonstrated by converging on the exactly-measurable game first.

Self-contained beyond the verified Leduc game logic in `leduc_cfr` / `leduc_q`;
OFF by default and imported nowhere in the engine, so the baseline is byte-identical
and all existing tests are untouched. Seeded (torch + Python RNG) from one seed.
Requires torch (optional dependency, like `src.rl_agent`); guarded by `_HAVE_TORCH`.
"""

import random
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAVE_TORCH = True
except ImportError:  # torch is an optional dependency (see requirements.txt)
    _HAVE_TORCH = False

from src.leduc_cfr import NUM_ACTIONS, _avail_from_key
from src.leduc_q import _DECK, _ACT_CHAR, _payoff0, _all_info_sets
from src.leduc_cfr import _available_actions, _is_terminal

# --- info-set featurizer ----------------------------------------------------
# Leduc info-set key (see leduc_cfr): round 1 "<card><r1>", round 2
# "<card><board><r1>/<r2>". Cards are J=0/Q=1/K=2 (suits collapsed); histories
# contain only 'c'/'r' (a fold ends the hand). Empirically max |r1|=4, |r2|=3.
_MAX_R1 = 4
_MAX_R2 = 3
FEAT_DIM = 3 + 3 + 1 + _MAX_R1 * 2 + _MAX_R2 * 2   # 21


def _onehot3(i):
    v = [0.0, 0.0, 0.0]
    if i is not None:
        v[i] = 1.0
    return v


def _enc_hist(h, maxlen):
    """Each betting slot -> [is_call, is_raise]; absent slots are zeros."""
    out = []
    for t in range(maxlen):
        if t < len(h):
            out += [1.0 if h[t] == "c" else 0.0, 1.0 if h[t] == "r" else 0.0]
        else:
            out += [0.0, 0.0]
    return out


def featurize(info_set):
    """Map a Leduc info-set key to a fixed 21-d feature vector (list of float)."""
    if "/" in info_set:
        s = info_set.index("/")
        card, board = int(info_set[0]), int(info_set[1])
        r1, r2, is_r2 = info_set[2:s], info_set[s + 1:], 1.0
    else:
        card, board = int(info_set[0]), None
        r1, r2, is_r2 = info_set[1:], "", 0.0
    return (_onehot3(card) + _onehot3(board) + [is_r2]
            + _enc_hist(r1, _MAX_R1) + _enc_hist(r2, _MAX_R2))


def _avail_mask(avail):
    """3-slot 0/1 mask of available actions."""
    return [1.0 if a in avail else 0.0 for a in range(NUM_ACTIONS)]


# --- networks ---------------------------------------------------------------

def _mlp(hidden):
    return nn.Sequential(
        nn.Linear(FEAT_DIM, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, NUM_ACTIONS),
    )


_NEG = -1e9   # logit/Q mask for unavailable actions


class LeducNeuralNFSP:
    """Neural NFSP on Leduc. beta = DQN best response (replay buffer + target
    net, epsilon-greedy); Pi = supervised average policy (reservoir buffer,
    cross-entropy on beta's own moves). Per episode each seat follows beta with
    probability eta, else Pi (anticipatory dynamics). Output: the average policy
    Pi over every info-set, scored by `leduc_eval.exploitability_of`."""

    def __init__(self, hidden=64, eta=0.1, eps_start=0.06, eps_end=0.0,
                 gamma=1.0, lr_rl=0.01, lr_sl=0.01, batch=128,
                 rl_capacity=200_000, sl_capacity=1_000_000,
                 target_update=1000, min_buffer=1000, seed=0):
        if not _HAVE_TORCH:
            raise ImportError("LeducNeuralNFSP requires torch (optional dep)")
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.eta = eta
        self.eps_start, self.eps_end = eps_start, eps_end
        self.gamma = gamma
        self.batch = batch
        self.target_update = target_update
        self.min_buffer = min_buffer

        self.q = _mlp(hidden)
        self.q_target = _mlp(hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.pi = _mlp(hidden)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=lr_rl)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr_sl)

        self.m_rl = deque(maxlen=rl_capacity)   # (feat,a,r,next_feat,next_mask,done)
        self.m_sl = []                          # reservoir of (feat,a,mask)
        self.sl_capacity = sl_capacity
        self._sl_seen = 0
        self._eps = eps_start
        self._steps = 0

    # --- behaviour ----------------------------------------------------------
    def _q_action(self, feat, avail):
        """epsilon-greedy best response over available actions (beta)."""
        if self.rng.random() < self._eps:
            return self.rng.choice(avail)
        with torch.no_grad():
            qv = self.q(torch.tensor(feat).float()).tolist()
        best = max(avail, key=lambda a: qv[a])
        return best

    def _pi_action(self, feat, avail):
        """sample from the average policy Pi, masked to available actions."""
        with torch.no_grad():
            logits = self.pi(torch.tensor(feat).float()).tolist()
        masked = [logits[a] if a in avail else _NEG for a in range(NUM_ACTIONS)]
        m = max(masked)
        exps = [pow(2.718281828, x - m) for x in masked]
        tot = sum(exps)
        r = self.rng.random() * tot
        upto = 0.0
        for a in range(NUM_ACTIONS):
            upto += exps[a]
            if r < upto:
                return a
        return avail[-1]

    def _play(self, c0, c1, board, modes):
        """One hand; seat p follows beta when modes[p] else Pi. Returns
        (decisions, payoff0) with decisions a list of (player, feat, action,
        avail, is_beta)."""
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
            feat = featurize(info_set)
            if modes[player]:
                a = self._q_action(feat, avail)
            else:
                a = self._pi_action(feat, avail)
            decisions.append((player, feat, a, avail, modes[player]))
            if in_r2:
                r2 = curr + _ACT_CHAR[a]
            else:
                r1 = curr + _ACT_CHAR[a]
        return decisions, _payoff0(c0, c1, board, r1, r2)

    # --- memory -------------------------------------------------------------
    def _store(self, decisions, payoff0):
        for p in (0, 1):
            seq = [(feat, a, avail, is_b)
                   for (pl, feat, a, avail, is_b) in decisions if pl == p]
            reward = payoff0 if p == 0 else -payoff0
            for i, (feat, a, avail, is_b) in enumerate(seq):
                if i + 1 < len(seq):
                    nfeat, navail = seq[i + 1][0], seq[i + 1][2]
                    self.m_rl.append((feat, a, 0.0, nfeat,
                                      _avail_mask(navail), 0.0))
                else:
                    self.m_rl.append((feat, a, reward,
                                      [0.0] * FEAT_DIM, [0.0] * NUM_ACTIONS, 1.0))
                if is_b:   # M_SL: average ONLY the best-response (beta) moves
                    entry = (feat, a, _avail_mask(avail))
                    self._sl_seen += 1
                    if len(self.m_sl) < self.sl_capacity:
                        self.m_sl.append(entry)
                    else:
                        j = self.rng.randint(0, self._sl_seen - 1)
                        if j < self.sl_capacity:
                            self.m_sl[j] = entry

    # --- learning -----------------------------------------------------------
    def _learn_q(self):
        if len(self.m_rl) < self.min_buffer:
            return
        batch = self.rng.sample(self.m_rl, self.batch)
        feat = torch.tensor([b[0] for b in batch]).float()
        act = torch.tensor([b[1] for b in batch]).long()
        rew = torch.tensor([b[2] for b in batch]).float()
        nfeat = torch.tensor([b[3] for b in batch]).float()
        nmask = torch.tensor([b[4] for b in batch]).float()
        done = torch.tensor([b[5] for b in batch]).float()
        with torch.no_grad():
            nq = self.q_target(nfeat)
            nq = nq + (nmask - 1.0) * 1e9        # -inf on unavailable
            nmax = nq.max(dim=1).values
            target = rew + self.gamma * (1.0 - done) * nmax
        qv = self.q(feat).gather(1, act.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(qv, target)
        self.opt_q.zero_grad()
        loss.backward()
        self.opt_q.step()

    def _learn_pi(self):
        if len(self.m_sl) < self.min_buffer:
            return
        batch = self.rng.sample(self.m_sl, self.batch)
        feat = torch.tensor([b[0] for b in batch]).float()
        act = torch.tensor([b[1] for b in batch]).long()
        mask = torch.tensor([b[2] for b in batch]).float()
        logits = self.pi(feat) + (mask - 1.0) * 1e9   # -inf on unavailable
        loss = F.cross_entropy(logits, act)
        self.opt_pi.zero_grad()
        loss.backward()
        self.opt_pi.step()

    def train(self, episodes, eval_every=None, eval_hook=None):
        """Run `episodes` of NFSP self-play. If `eval_every` and `eval_hook` are
        given, call eval_hook(episode, self) at those checkpoints (used to record
        the exact-exploitability convergence curve)."""
        history = []
        for ep in range(episodes):
            frac = ep / max(1, episodes - 1)
            self._eps = self.eps_start + (self.eps_end - self.eps_start) * frac
            i, j, k = self.rng.sample(range(6), 3)
            modes = {p: (self.rng.random() < self.eta) for p in (0, 1)}
            decisions, payoff0 = self._play(_DECK[i], _DECK[j], _DECK[k], modes)
            self._store(decisions, payoff0)
            self._learn_q()
            self._learn_pi()
            self._steps += 1
            if self._steps % self.target_update == 0:
                self.q_target.load_state_dict(self.q.state_dict())
            if eval_every and eval_hook and (ep + 1) % eval_every == 0:
                history.append(eval_hook(ep + 1, self))
        return history

    # --- output policy ------------------------------------------------------
    def average_strategy_table(self):
        """Pi over EVERY Leduc info-set as {info_set: [p_fold,p_call,p_raise]},
        masked to available actions, for `leduc_eval.exploitability_of`."""
        keys = _all_info_sets()
        feats = torch.tensor([featurize(s) for s in keys]).float()
        with torch.no_grad():
            logits = self.pi(feats)
        table = {}
        for idx, s in enumerate(keys):
            av = _avail_from_key(s)
            row = logits[idx].tolist()
            masked = [row[a] if a in av else _NEG for a in range(NUM_ACTIONS)]
            m = max(masked)
            exps = [pow(2.718281828, x - m) if a in av else 0.0
                    for a, x in enumerate(masked)]
            tot = sum(exps)
            table[s] = [e / tot if tot > 0 else (1.0 / len(av) if a in av else 0.0)
                        for a, e in enumerate(exps)]
        return table
