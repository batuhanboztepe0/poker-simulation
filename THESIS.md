# Poker as a Market-Microstructure Sandbox — thesis, method & honest results

*Project narrative for a quant-research portfolio. Bibliography:
[references.md](references.md). Figures: [figures/](figures/) (start with
[`figures/exec_summary.png`](figures/exec_summary.png)). This document is the
narrative; the repository `README.md` is intentionally left for a human to write
last.*

---

## 1. Thesis — what poker and market-making share (and what they don't)

Both poker and market-making extract edge from **predictable counterparty
behaviour against a background of noise**: *predictable deviations are
exploitable; pure randomness is not.*

The defensible version of this is **not** a marketing analogy — it is the
informed-vs-uninformed-trader distinction of **Kyle (1985)** and the
adverse-selection model of **Glosten & Milgrom (1985)** (references.md §3). An
exploitable, predictable opponent is the analog of *informed / toxic flow* you
can read; a uniformly-random opponent is *noise flow* you cannot. Opponent
modelling ↔ detecting adverse selection; Kelly / log-bankroll growth is the
shared capital-allocation objective in both domains.

**Honest boundary of the claim.** I do *not* claim a validated tradable signal.
The popular "toxic-flow metric" mapping (VPIN) does **not** survive scrutiny: the
specific claims that VPIN is parameter-free and an empirically validated
volatility predictor were *refuted* under adversarial verification
(references.md §3). The analogy holds at the **decision-theory** level (EV under
uncertainty, bankroll growth, adverse-selection detection), not as alpha.

**Institutional grounding.** This is how a top-tier market maker actually
trains. SIG uses poker as formal trader pedagogy — a mandatory ~100-hour
requirement, with *The Mathematics of Poker* authors on staff — to teach
"the same thought process as evaluating the expected value of a trade and
pricing risk" (references.md §4).

## 2. What was built (method)

- A **seeded, chip-conserving** Texas Hold'em engine (2–9 players, side pots)
  with a Monte-Carlo equity calculator. One top-level seed reproduces a full
  session.
- A **self-play RL agent**: DQN over equity + opponent-belief features, TD
  bootstrapping, risk-neutral *and* Kelly/log-bankroll *and* ICM reward variants,
  and an HMM "tilt" opponent-belief used as a feature.
- An **evaluation layer built like a quant backtest** ([src/evaluation.py](src/evaluation.py)):
  each seed is one *paired* scenario; it reports per-seed PnL diffs, paired
  t-tests, **bootstrap 95% confidence intervals**, and opt-in
  **duplicate/mirror matching** (same deck, swapped seats — the variance-reduction
  protocol behind DIVAT/AIVAT, references.md §2).
- **Engineering discipline**: every advanced feature is opt-in / default-off
  (baseline byte-identical), 480 tests pass, and a **multi-agent adversarial
  audit** of both the code and the figures caught and fixed overclaims *before*
  they were committed.

## 3. Honest results — the headline is the honesty

The lead figure, [`figures/exec_summary.png`](figures/exec_summary.png), shows
every claimed edge as a point with its 95% bootstrap CI:

- **RL beats the myopic baseline directionally** (+560 chips/match) — but the
  95% CI's lower bound sits at ~0. Real, but **marginal**.
- A **belief + opponent-mix generalist** tops a cross-agent leaderboard (+209)
  and beats each adaptive opponent head-to-head — but its leaderboard CI
  **includes 0** at 16 seeds, and it loses head-to-head to an analytic **Kelly**
  agent.
- A **risk-averse ICM/Kelly reward shows no robust edge** over a risk-neutral
  chip reward heads-up — it is *significantly negative* on the mild prize ladder.
- A rigorous **Block-B sweep** (finer action grid, un-truncated bust clip,
  tilt-bonus decoupling, snapshot self-play, warmed fold-equity) found every
  RL-*mechanics* lever works mechanically but **none moves the headline**
  ([figures/](figures/), B1–B5 panels).

This matches the field: even an **80,000-hand** human-AI match with a margin
"huge" by professional standards was only **at the edge of significance** without
variance reduction (Claudico 2015; references.md §2). Marginal,
honestly-measured edges are the *correct* finding for a single-developer DQN in a
high-variance game — not a failure to hide.

## 4. Where this sits vs the state of the art

DQN self-play here is a **deliberate pragmatic baseline, not a claim to compete
with the CFR family.** The superhuman poker AIs — DeepStack (2017), Libratus
(2018), **Pluribus** (2019, first superhuman multiplayer, >30 mbb/g), ReBeL
(2020) — are CFR / deep-RL-plus-search with game-theoretic grounding. Plain DQN
self-play even **violates the stationarity assumption** Q-learning needs (the
opponent moves while you learn). This repo makes that concrete and *exact* on
Leduc Hold'em ([`figures/exploitability.png`](figures/exploitability.png)): the
time-AVERAGE strategy's exploitability falls toward 0 (Nash), but the greedy
LAST-ITERATE — the regime DQN self-play plays in — stays exploitable and does not
converge. **NFSP** (Heinrich & Silver 2016) is the theoretically-grounded next
step: it brings exactly this averaging to large games via a neural average-policy
network (references.md §1).

## 5. Honest-negative as a feature, not a bug

The deliverable is **method + intellectual honesty**: reproducible, paired,
variance-accounted, adversarially-audited, with null/marginal results reported as
such. *(Caveat: that quant firms specifically reward null results is practitioner
signal, not documented policy — the defensible claim is that rigour + honesty
signal statistical maturity, references.md §5.)*

## 6. What the rigor layer ships, and what remains

**Shipped (the deep-research rigor recommendations, references.md §1–§2):**
- **Variance reduction** — bootstrap CIs, duplicate/mirror matching (cuts the
  heads-up CI to ~65% of raw), and the **all-in EV control variate** (the
  AIVAT-family chance-node adjustment; unbiased, chip-conserving, opt-in). See
  [`figures/variance_reduction.png`](figures/variance_reduction.png) and
  [`figures/exec_summary.png`](figures/exec_summary.png).
- **An exact exploitability metric** ([`src/leduc_eval.py`](src/leduc_eval.py))
  for any Leduc strategy, and the verifiable demonstration
  ([`figures/exploitability.png`](figures/exploitability.png)) that the
  time-AVERAGE strategy converges to the Nash equilibrium while the greedy
  LAST-ITERATE — the regime DQN self-play plays in — stays exploitable. This is
  the exact reason DQN self-play does not reach Nash and averaging methods do.

**Remaining (the genuine next steps):**
- **Full decision-node AIVAT** — extend the chance-node (all-in) control variate
  with a *pre-committed* value function at decision nodes for the ~10–44× CI
  reduction; the precondition is to fix the heuristic before seeing the
  evaluation data (the documented footgun).
- **A neural NFSP learner on NLHE** — carry the averaging that reaches Nash on
  Leduc to the large game via a neural average-policy network, evaluated with the
  exploitability metric now in place. (DQN here stays a deliberate baseline.)
- **Thesis** — keep the markets connection on **Kyle / Glosten-Milgrom** (drop
  VPIN); if pursued, *test* the exploit-predictable-deviations idea on real
  order-flow data rather than asserting the analogy.

---

*Every empirical number above traces to committed data under
[results/](results/) and a figure under [figures/](figures/); every external
claim traces to [references.md](references.md), where claims that did **not**
survive adversarial verification are kept visible on purpose.*
