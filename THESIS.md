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
opponent moves while you learn). **NFSP** (Heinrich & Silver 2016) is the
theoretically-grounded next step (references.md §1).

## 5. Honest-negative as a feature, not a bug

The deliverable is **method + intellectual honesty**: reproducible, paired,
variance-accounted, adversarially-audited, with null/marginal results reported as
such. *(Caveat: that quant firms specifically reward null results is practitioner
signal, not documented policy — the defensible claim is that rigour + honesty
signal statistical maturity, references.md §5.)*

## 6. Concrete next steps (prioritized)

- **Rigor (highest value).** Implement **full AIVAT** — a *pre-committed* equity
  value function as a control variate at chance *and* decision nodes — to cut the
  CIs by ~10–44× (references.md §2) and resolve whether the marginal edges are
  real or zero. This repo already ships the first layer (bootstrap CIs +
  duplicate/mirror matching); the AIVAT precondition is to **fix the heuristic
  before seeing the evaluation data** (post-hoc tuning voids the guarantee — the
  documented footgun).
- **Method.** Swap DQN self-play for **NFSP/PSRO** (better-grounded) and add an
  **exploitability / best-response** metric alongside raw win rate.
- **Thesis.** Keep the markets connection strictly on **Kyle / Glosten-Milgrom**
  (drop VPIN); if pursued, *test* the exploit-predictable-deviations idea on a
  real order-flow dataset rather than asserting the analogy.

---

*Every empirical number above traces to committed data under
[results/](results/) and a figure under [figures/](figures/); every external
claim traces to [references.md](references.md), where claims that did **not**
survive adversarial verification are kept visible on purpose.*
