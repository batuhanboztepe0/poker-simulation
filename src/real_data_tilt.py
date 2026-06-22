"""
real_data_tilt.py
-----------------
Validate the HMM tilt detector on REAL human hands.

This module is the real-data counterpart to the synthetic
`scripts/probe_tilt_detection.py`. The synthetic probe scores the detector
against a bot whose hidden `is_tilted` flag is ground truth; real human logs
carry no tilt label, so here the *behavioural definition* of tilt is the proxy
ground truth: a player who has just suffered a (big) loss is hypothesised to
play looser and more aggressively. The thesis link is Kyle (1985) /
Glosten-Milgrom (1985): a predictable, post-loss deviation is the poker analog
of the adverse-selection signal an opponent model exists to read (references.md
§3) — the cross-domain mapping to real order flow is an untested hypothesis. The
label is observational ("post-loss"/"loss-associated"), not causal.

IMPORTANT — real hands feed the OPPONENT MODEL ONLY, never the DQN policy.
Training a policy on human logs would make the self-play agent exploitable; the
policy stays self-play. This module is offline, opt-in, and imports nothing from
the training path: only `HMMBeliefState` / `TiltTrigger` from `opponent_model`
(both torch-free) and `bootstrap_ci` from the dependency-light `src.stats` (NOT
from `src.evaluation`, which would transitively pull `rl_agent` -> torch).

Three honest tests, run by two DISTINCT HMMs (do not conflate them):
  A. PHENOMENON (model-free)  -- within-player, is aggression / VPIP higher in
     the hand after a big loss than otherwise? Cluster-bootstrap CI over players.
  B. DETECTOR (the project's forward-filter HMM) -- run `HMMBeliefState` as a
     forward filter in EMISSION-ONLY mode: use_pnl=False AND a zero-rate trigger
     (TiltTrigger(epsilon=0.0)) with delta_stack=0, so the transition has no
     PnL-independent drift and P(tilted) rises ONLY when observed aggression
     emissions support the tilted state — never on a PnL feed (which would make
     it rise after losses by construction). Measure P(tilted) separation post-loss.
  C. REGIME-FIT (a SEPARATE Baum-Welch HMM) -- fit a 2-state CategoricalHMM
     (hmmlearn) on the binary aggressive-hand sequences and ask whether it beats
     a 1-state i.i.d. model (held-out log-likelihood; a held-out, BIC-penalised
     score) and whether its high-aggression state aligns with recent losses. This
     CORROBORATES the phenomenon with a different method; it does NOT validate the
     Test-B detector architecture.

Data: PHH Dataset (Kim 2024, Zenodo DOI 10.5281/zenodo.13997158, CC-BY-4.0).
The NLHE hands originate from a 2009 HandHQ scrape, redistributed under
CC-BY-4.0; see references.md.
"""

from src.stats import bootstrap_ci
from src.opponent_model import HMMBeliefState, TiltTrigger

# pokerkit action verbs (PHH notation).
AGGRESSIVE_VERB = "cbr"          # complete / bet / raise (the only aggressive verb)
PASSIVE_VERB = "cc"              # check / call
FOLD_VERB = "f"


# --------------------------------------------------------------------------
# 1. Parsing PHH hands -> per-(hand, player) observations
# --------------------------------------------------------------------------

def _ts_key(hh, fallback):
    """A sortable timestamp key for one hand (year, month, day, seconds)."""
    t = getattr(hh, "time", None)
    secs = (t.hour * 3600 + t.minute * 60 + t.second) if t is not None else 0
    return (getattr(hh, "year", 0) or 0, getattr(hh, "month", 0) or 0,
            getattr(hh, "day", 0) or 0, secs, fallback)


def _hand_observations(hh, fallback_idx):
    """
    Reduce one parsed PHH hand to a per-seated-player observation.

    Aggression / VPIP / PFR come from the action string sequence; the exact net
    chip result comes from replaying the hand with pokerkit (chip-conserving, so
    rake-free and correct for uncalled-bet returns and all-ins).

    Returns list[dict]: one dict per player who acted or posted a blind, with
    keys player, table, ts, n_aggr, n_dec, vpip, pfr, net_bb.
    """
    players = list(hh.players)
    n = len(players)
    blinds = [float(b) for b in hh.blinds_or_straddles[:n]]
    bb = max(blinds) if blinds else 1.0
    bb = bb if bb > 0 else 1.0

    # Light pre-flop bet tracker (just enough to know whether a `cc` voluntarily
    # put money in pre-flop -> VPIP). Post the blinds first.
    street_put = [0.0] * n
    for i, b in enumerate(blinds):
        if b > 0:
            street_put[i] = b
    cur = max(street_put) if street_put else 0.0

    n_aggr = [0] * n
    n_dec = [0] * n
    vpip = [False] * n
    pfr = [False] * n
    street = 0
    idx = {f"p{i + 1}": i for i in range(n)}

    for a in hh.actions:
        tok = a.split()
        if tok[0] == "d":                       # dealer op
            if tok[1] == "db":                  # board card(s) -> next street
                street += 1
            continue
        p = idx.get(tok[0])
        if p is None:
            continue
        verb = tok[1]
        if verb == FOLD_VERB:
            n_dec[p] += 1
        elif verb == PASSIVE_VERB:
            n_dec[p] += 1
            if street == 0 and cur > street_put[p]:   # a real pre-flop call
                vpip[p] = True
                street_put[p] = cur
        elif verb == AGGRESSIVE_VERB:
            n_dec[p] += 1
            n_aggr[p] += 1
            if street == 0:
                vpip[p] = True
                pfr[p] = True
                to = float(tok[2])
                street_put[p] = to
                cur = to
        # 'sm' (show/muck) and others are not decisions.

    start = [float(s) for s in hh.starting_stacks]
    final = _replay_final_stacks(hh, start)
    table = getattr(hh, "table", None) or "?"
    hand_id = getattr(hh, "hand", None) or fallback_idx
    ts = _ts_key(hh, fallback_idx)

    out = []
    for i in range(n):
        if n_dec[i] == 0 and blinds[i] == 0:
            continue                             # never in the hand
        net_bb = (final[i] - start[i]) / bb
        out.append({"player": players[i], "table": table, "hand": hand_id,
                    "ts": ts, "n_aggr": n_aggr[i], "n_dec": n_dec[i],
                    "vpip": vpip[i], "pfr": pfr[i], "net_bb": net_bb})
    return out


def _replay_final_stacks(hh, start):
    """Replay a pokerkit HandHistory to terminal and return final stacks."""
    state = None
    for state in hh:
        pass
    if state is None:
        return list(start)
    return [float(s) for s in state.stacks]


def parse_phhs(path):
    """
    Parse one .phhs file (a PHH series of NLHE hands) into per-player
    observations. Requires pokerkit (analysis-only dependency).

    Returns list[dict] (see `_hand_observations`).
    """
    import pokerkit  # lazy: keeps the module importable without pokerkit

    records = []
    with open(path, "rb") as f:
        for k, hh in enumerate(pokerkit.HandHistory.load_all(f)):
            try:
                records.extend(_hand_observations(hh, k))
            except Exception:
                # A malformed / unsupported hand is skipped, not fatal.
                continue
    return records


# --------------------------------------------------------------------------
# 2. Per-player session sequences
# --------------------------------------------------------------------------

def build_sequences(records, min_len=15, max_gap_s=3600):
    """
    Group per-hand observations into time-ordered (player, table) sittings.

    A "session" is one player at one table; consecutive hands more than
    `max_gap_s` seconds apart are split into separate sessions (a player who
    leaves and returns). Sessions shorter than `min_len` are dropped — regime
    detection needs a few dozen hands to mean anything.

    Ordering / dataset note (PHH HandHQ scrape): every hand carries the SAME
    placeholder calendar date (2009-07-01) with a second-resolution time-of-day,
    so `_gap_seconds` orders and gap-splits hands by that within-day time (the
    cross-day `None` branch never fires here). That time field is reliable for
    ordering: sorting a table's hands by it reproduces the monotonic PokerStars
    hand-id order on ~99.9% of adjacent pairs (measured), so `max_gap_s` splits at
    genuine within-day breaks. (A handful of hands stamped exactly 00:00:00 are a
    negligible edge.)

    Returns list[list[dict]]: each inner list is one session's observations in
    time order.
    """
    by_pt = {}
    for r in records:
        by_pt.setdefault((r["player"], r["table"]), []).append(r)

    sessions = []
    for obs in by_pt.values():
        obs.sort(key=lambda r: r["ts"])
        cur = [obs[0]]
        for prev, nxt in zip(obs, obs[1:]):
            gap = _gap_seconds(prev["ts"], nxt["ts"])
            if gap is None or gap > max_gap_s:
                if len(cur) >= min_len:
                    sessions.append(cur)
                cur = [nxt]
            else:
                cur.append(nxt)
        if len(cur) >= min_len:
            sessions.append(cur)
    return sessions


def _gap_seconds(a, b):
    """Seconds between two ts keys, or None if they cross a day boundary. In this
    single-date dataset (all hands 2009-07-01) the `None` branch never fires, so
    gaps are the within-day time-of-day differences (see `build_sequences`)."""
    if a[:3] != b[:3]:           # different (year, month, day)
        return None
    return b[3] - a[3]


def _aggr_rate(o):
    """Aggression rate for one hand observation, or None if no decisions."""
    return o["n_aggr"] / o["n_dec"] if o["n_dec"] > 0 else None


# --------------------------------------------------------------------------
# 3. Test A — the phenomenon (model-free)
# --------------------------------------------------------------------------

def phenomenon_test(sequences, loss_bb=10.0, min_per_group=5, placebo_seed=None):
    """
    Within-player, is play more aggressive / looser in the hand *after* a big
    loss than otherwise?

    For each session, hand i is labelled "post-loss" if hand i-1 had
    net_bb <= -loss_bb, else "baseline". Per player we average the metric in each
    group (requiring >= `min_per_group` hands in BOTH groups so the player's two
    means are stable), then bootstrap the per-player paired difference
    (post-loss − baseline). Clustering by player is the honest unit: hands within
    a player are correlated, so the player — not the hand — is the sample.

    `placebo_seed`: if set, the post/baseline labels are randomly permuted within
    each player (same group sizes, temporal link broken). This is the negative
    control — the effect must collapse to ~0, proving the signal is the loss
    timing and not an artifact of the unequal group sizes or the estimator.

    Returns {aggr: {...ci...}, vpip: {...ci...}, n_players, n_post, n_base}.
    """
    import random
    rng = random.Random(placebo_seed) if placebo_seed is not None else None
    agg_diffs, vpip_diffs = [], []
    n_post = n_base = 0
    for entries in _per_player_entries(sequences, loss_bb).values():
        posts = [e[2] for e in entries]
        if rng is not None:
            rng.shuffle(posts)
        ap = [e[0] for e, p in zip(entries, posts) if p]
        ab = [e[0] for e, p in zip(entries, posts) if not p]
        vp = [e[1] for e, p in zip(entries, posts) if p]
        vb = [e[1] for e, p in zip(entries, posts) if not p]
        if len(ap) < min_per_group or len(ab) < min_per_group:
            continue
        agg_diffs.append(sum(ap) / len(ap) - sum(ab) / len(ab))
        vpip_diffs.append(sum(vp) / len(vp) - sum(vb) / len(vb))
        n_post += len(ap)
        n_base += len(ab)
    return {
        "aggr": bootstrap_ci(agg_diffs),
        "vpip": bootstrap_ci(vpip_diffs),
        "n_players": len(agg_diffs),
        "n_post": n_post,
        "n_base": n_base,
        "loss_bb": loss_bb,
        "placebo": placebo_seed is not None,
    }


def _per_player_entries(sequences, loss_bb):
    """player -> list of (aggr_rate, vpip01, post_loss_bool) for each hand that
    follows another hand in a session. Pools a player's sessions."""
    by_player = {}
    for sess in sequences:
        for prev, cur in zip(sess, sess[1:]):
            rate = _aggr_rate(cur)
            if rate is None:
                continue
            post = prev["net_bb"] <= -loss_bb
            by_player.setdefault(cur["player"], []).append(
                (rate, 1.0 if cur["vpip"] else 0.0, post))
    return by_player


def _cohen_d(diffs):
    """Paired Cohen's d (mean / SD of the per-player paired differences).

    None if < 2 values or zero variance. This is the standardised effect size of
    the loss-vs-win asymmetry; |d| ~ 0.2 small, 0.5 medium, 0.8 large.
    """
    n = len(diffs)
    if n < 2:
        return None
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    if var <= 0:
        return None
    return m / (var ** 0.5)


def _per_player_swing_entries(sequences):
    """player -> list of (aggr_rate, vpip01, prev_net_bb) for each hand that
    follows another hand in a session (the previous hand's net result is kept so
    the caller can classify the swing sign)."""
    by_player = {}
    for sess in sequences:
        for prev, cur in zip(sess, sess[1:]):
            rate = _aggr_rate(cur)
            if rate is None:
                continue
            by_player.setdefault(cur["player"], []).append(
                (rate, 1.0 if cur["vpip"] else 0.0, prev["net_bb"]))
    return by_player


def within_player_loss_vs_win(sequences, swing_bb=10.0, min_per_group=5,
                              placebo_seed=None):
    """
    The SYMMETRIC within-player control for the post-loss tilt phenomenon.

    `phenomenon_test` compares a player's post-(big-)loss hands against ALL their
    other hands — a baseline dominated by ordinary, no-recent-swing hands. So that
    contrast can reflect *any* big-pot arousal (not loss specifically), and it
    does not fully close the player-type confound (looser players both lose more
    and play looser to begin with).

    This test compares, WITHIN each player, the hand after a >= `swing_bb` LOSS
    against the hand after a >= `swing_bb` WIN. Both groups are "the hand after an
    equal-magnitude decisive pot for the same player", so player identity, big-pot
    arousal, and event magnitude are all matched — the only thing that differs is
    the SIGN of the swing. A positive difference (more aggressive / looser after a
    loss than after an equal win) is the prospect-theory loss-aversion asymmetry;
    a null says the post-loss shift is big-pot arousal, not loss-specific.

    Per player we require >= `min_per_group` hands in BOTH the post-loss and
    post-win groups, average each metric per group, and bootstrap the per-player
    paired difference (post-loss − post-win). Cohen's d (paired) is the effect
    size. `placebo_seed` permutes the loss/win labels within each player among the
    swing hands (group sizes preserved) — the negative control, which must
    collapse to ~0.

    Returns {aggr, vpip, aggr_cohen_d, vpip_cohen_d, n_players, n_loss, n_win,
             swing_bb, placebo}.
    """
    import random
    rng = random.Random(placebo_seed) if placebo_seed is not None else None
    agg_diffs, vpip_diffs = [], []
    n_loss = n_win = 0
    for entries in _per_player_swing_entries(sequences).values():
        labels = [(-1 if prev <= -swing_bb else 1 if prev >= swing_bb else 0)
                  for (_r, _v, prev) in entries]
        if rng is not None:
            # Permute only the loss/win labels among the swing hands (the
            # 'neither' hands stay out), so group sizes are preserved.
            idx = [i for i, l in enumerate(labels) if l != 0]
            shuffled = [labels[i] for i in idx]
            rng.shuffle(shuffled)
            for i, l in zip(idx, shuffled):
                labels[i] = l
        al = [entries[i][0] for i, l in enumerate(labels) if l == -1]
        aw = [entries[i][0] for i, l in enumerate(labels) if l == 1]
        vl = [entries[i][1] for i, l in enumerate(labels) if l == -1]
        vw = [entries[i][1] for i, l in enumerate(labels) if l == 1]
        if len(al) < min_per_group or len(aw) < min_per_group:
            continue
        agg_diffs.append(sum(al) / len(al) - sum(aw) / len(aw))
        vpip_diffs.append(sum(vl) / len(vl) - sum(vw) / len(vw))
        n_loss += len(al)
        n_win += len(aw)
    return {
        "aggr": bootstrap_ci(agg_diffs),
        "vpip": bootstrap_ci(vpip_diffs),
        "aggr_cohen_d": _cohen_d(agg_diffs),
        "vpip_cohen_d": _cohen_d(vpip_diffs),
        "n_players": len(agg_diffs),
        "n_loss": n_loss,
        "n_win": n_win,
        "swing_bb": swing_bb,
        "placebo": placebo_seed is not None,
    }


# --------------------------------------------------------------------------
# 4. Test B — the project's HMM detector, emission-only
# --------------------------------------------------------------------------

def detector_separation(sequences, loss_bb=10.0, min_per_group=5,
                        mu_normal=0.30, mu_tilted=0.80, recover=0.15,
                        placebo_seed=None):
    """
    Run the project's `HMMBeliefState` over each real session as a forward filter
    in EMISSION-ONLY mode so its P(tilted) responds only to observed aggression —
    never to a PnL feed, which would make it rise after losses by construction
    (the circularity this test must avoid). Then measure the per-player paired
    separation of P(tilted) between post-(big-)loss hands and baseline hands.

    EMISSION-ONLY means three things together: use_pnl=False (no hand-boundary
    PnL transition), delta_stack=0 on every update, AND a zero-rate transition
    trigger (TiltTrigger(epsilon=0.0)). The last one matters: with the default
    epsilon=0.04 the constant-transition predict step drifts the prior toward its
    stationary P(tilted) (~0.21) regardless of any emission, so P(tilted) would
    climb on hand COUNT alone. Zeroing epsilon makes the only path into the tilted
    state the aggression emission likelihood (recovery still pulls back toward
    normal), so the separation is genuinely emission-driven.

    The emission parameters should be set from a population/train summary BEFORE
    looking at this separation, not tuned to maximise it (the Kim & Sandholm
    post-hoc-tuning footgun, references.md §2).

    `placebo_seed`: if set, the post/baseline labels are permuted within each
    player (negative control) — the separation must collapse to ~0.

    Returns {separation: {...ci...}, n_players, mu_normal, mu_tilted, recover}.
    """
    import random
    rng = random.Random(placebo_seed) if placebo_seed is not None else None
    by_player = {}
    for sess in sequences:
        belief = HMMBeliefState(mu_normal=mu_normal, mu_tilted=mu_tilted,
                                recover=recover, use_pnl=False,
                                tilt_trigger=TiltTrigger(epsilon=0.0))
        prev = None
        for o in sess:
            if o["n_dec"] > 0:
                belief.update(action=None, n_aggressive=o["n_aggr"],
                              n_actions=o["n_dec"], delta_stack=0)
            p_t = belief.p_tilted()
            if prev is not None and o["n_dec"] > 0:
                post = prev["net_bb"] <= -loss_bb
                by_player.setdefault(o["player"], []).append((p_t, post))
            prev = o

    diffs = []
    for entries in by_player.values():
        posts = [e[1] for e in entries]
        if rng is not None:
            rng.shuffle(posts)
        p = [e[0] for e, q in zip(entries, posts) if q]
        b = [e[0] for e, q in zip(entries, posts) if not q]
        if len(p) >= min_per_group and len(b) >= min_per_group:
            diffs.append(sum(p) / len(p) - sum(b) / len(b))
    return {
        "separation": bootstrap_ci(diffs),
        "n_players": len(diffs),
        "mu_normal": mu_normal,
        "mu_tilted": mu_tilted,
        "recover": recover,
        "loss_bb": loss_bb,
        "placebo": placebo_seed is not None,
    }


# --------------------------------------------------------------------------
# 5. Test C — fit a 2-state HMM and compare to 1-state
# --------------------------------------------------------------------------

def _aggr_symbol(o):
    """Binary aggressive-hand indicator (1 if >=1 aggressive action), or None."""
    if o["n_dec"] <= 0:
        return None
    return 1 if o["n_aggr"] >= 1 else 0


def fit_regime_hmm(sequences, loss_bb=10.0, seed=0, test_frac=0.3,
                   max_sessions=3000):
    """
    Fit a 2-state CategoricalHMM (Baum-Welch) on per-hand BINARY aggressive-hand
    sequences and compare it to a 1-state i.i.d. Bernoulli. A binary emission is
    used deliberately: per-hand aggression *rate* is spiky-at-zero (most hands
    are folds), which collapses a Gaussian state onto 0 with near-zero variance
    and produces a meaningless BIC. The binary "was this hand aggressive?" symbol
    is the honest discrete observable.

    Reports held-out log-likelihood and BIC for both models, and whether the
    learned high-aggression state aligns with recent losses (P[prev hand was a
    big loss | high state] vs the base rate).

    Returns a dict of metrics, or {"ok": False, ...} if hmmlearn is unavailable
    or there is too little data.
    """
    try:
        import math
        import numpy as np
        from hmmlearn.hmm import CategoricalHMM
    except Exception as e:                       # pragma: no cover - env guard
        return {"ok": False, "reason": f"hmmlearn unavailable: {e}"}

    def _symbols(sess):
        return [_aggr_symbol(o) for o in sess if _aggr_symbol(o) is not None]

    sessions_f = [s for s in sequences if len(_symbols(s)) >= 10]
    if len(sessions_f) < 8:
        return {"ok": False, "reason": "too few sessions"}

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(sessions_f)).tolist()
    # Cap the working set: a few thousand sessions is ample for a 2-symbol BIC
    # comparison, and keeps the multi-restart fit fast and reproducible.
    working = [sessions_f[i] for i in order[:max_sessions]]
    n_test = max(1, int(len(working) * test_frac))
    test_sess = working[:n_test]
    train_sess = working[n_test:]

    def _pack(sessions_subset):
        sym_lists = [_symbols(s) for s in sessions_subset]
        X = np.array([[s] for sl in sym_lists for s in sl], dtype=int)
        lengths = [len(sl) for sl in sym_lists]
        return X, lengths

    Xtr, Ltr = _pack(train_sess)
    Xte, Lte = _pack(test_sess)

    def _fit(k, n_restarts=12, floor=0.02):
        """Best-of-`n_restarts` Baum-Welch fit. A 2-state CategoricalHMM has two
        traps: a SYMMETRIC fixed point where both states collapse to the marginal
        (so it never sees structure that IS there), and a DEGENERATE corner where
        an emission hits 0/1 (a numerical artifact that fakes structure that is
        NOT there). Asymmetric random inits escape the first; rejecting fits with
        an emission outside [floor, 1-floor] avoids the second. Among the valid
        fits we keep the highest TRAIN log-likelihood."""
        best = None
        for r in range(n_restarts):
            m = CategoricalHMM(n_components=k, n_features=2, n_iter=200,
                               random_state=seed + r, init_params="st",
                               params="ste", tol=1e-4)
            rs = np.random.RandomState(seed + r)
            if k == 1:
                m.emissionprob_ = np.array([[0.5, 0.5]])
            else:
                lo = 0.05 + 0.25 * rs.random()
                hi = 0.60 + 0.35 * rs.random()
                m.emissionprob_ = np.array([[1 - lo, lo], [1 - hi, hi]])
            try:
                m.fit(Xtr, Ltr)
            except Exception:
                continue
            e = m.emissionprob_
            if k > 1 and (e.min() < floor or e.max() > 1 - floor):
                continue                          # reject degenerate corner
            s = m.score(Xtr, Ltr)
            if best is None or s > best[0]:
                best = (s, m)
        return best[1] if best else None

    m1, m2 = _fit(1), _fit(2)
    if m1 is None or m2 is None:
        # No non-degenerate 2-state fit -> structure is not supported.
        return {"ok": True, "bic_gain": None, "two_state_found": False,
                "n_sessions": len(sessions_f), "n_train": len(train_sess),
                "n_test": len(test_sess)}
    ll1 = m1.score(Xte, Lte)
    ll2 = m2.score(Xte, Lte)

    # Free params for CategoricalHMM (2 symbols), k states:
    #   (k-1) start + k(k-1) transition + k emission(P[symbol=1]).
    def _nparams(k):
        return (k - 1) + k * (k - 1) + k

    # BIC-style penalised score. NOTE: `loglik` here is the HELD-OUT (test-fold)
    # log-likelihood, not the in-sample train likelihood of textbook BIC; the
    # complexity penalty uses the fit-set size (len(Xtr)) as BIC prescribes. So
    # `bic_gain` is a generalisation-penalised model-selection score, and its sign
    # agrees with `heldout_ll_gain` (the primary, assumption-free metric). Using
    # held-out LL is deliberate: it is what lets the iid case reject a spurious
    # 2-state fit (an in-sample LL would always favour the larger model).
    def _bic(loglik, k):
        return -2.0 * loglik + _nparams(k) * math.log(max(1, len(Xtr)))

    bic1, bic2 = _bic(ll1, 1), _bic(ll2, 2)
    align = _state_loss_alignment(m2, test_sess, loss_bb, np)
    p_aggr = sorted(float(x) for x in m2.emissionprob_[:, 1])
    return {
        "ok": True,
        "two_state_found": True,
        "heldout_ll_1state": float(ll1),
        "heldout_ll_2state": float(ll2),
        "heldout_ll_gain": float(ll2 - ll1),
        "bic_1state": float(bic1),
        "bic_2state": float(bic2),
        "bic_gain": float(bic1 - bic2),     # >0 => 2-state preferred
        "p_aggr_low": p_aggr[0],            # P[aggressive hand] in the calm state
        "p_aggr_high": p_aggr[-1],          # ... in the active state
        "n_sessions": len(sessions_f),
        # Working set: only `max_sessions` of the eligible sessions are used (a
        # few thousand 2-symbol sequences are ample for the BIC comparison and
        # keep the multi-restart fit fast/reproducible). Surfaced for honesty.
        "sessions_available": len(sessions_f),
        "sessions_used": len(working),
        "max_sessions": max_sessions,
        "n_train": len(train_sess),
        "n_test": len(test_sess),
        **align,
    }


def _state_loss_alignment(model, test_sessions, loss_bb, np):
    """
    P(prev hand was a big loss | Viterbi state == high-aggression state) vs the
    overall base rate of post-big-loss hands, on the held-out test sessions. The
    post-loss label uses the literal preceding hand in the full session.
    """
    high = int(np.argmax(model.emissionprob_[:, 1]))
    in_high_loss = in_high = total_loss = total = 0
    for sess in test_sessions:
        syms, labels = [], []
        for j, o in enumerate(sess):
            if _aggr_symbol(o) is None:
                continue
            syms.append(_aggr_symbol(o))
            prev = sess[j - 1] if j > 0 else None
            labels.append(prev is not None and prev["net_bb"] <= -loss_bb)
        if len(syms) < 2:
            continue
        states = model.predict(np.array([[s] for s in syms], dtype=int),
                               [len(syms)])
        for st, lost in zip(states, labels):
            total += 1
            total_loss += 1 if lost else 0
            if st == high:
                in_high += 1
                in_high_loss += 1 if lost else 0
    return {
        "p_loss_given_high": (in_high_loss / in_high) if in_high else 0.0,
        "p_loss_base": (total_loss / total) if total else 0.0,
        "high_state": high,
    }
