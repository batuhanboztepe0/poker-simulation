# Poker as a Decision-Science Sandbox — finding edge under uncertainty, measured honestly

*Project narrative for a trader / quant-research portfolio. Bibliography:
[references.md](references.md). Figures: [figures/](figures/) (start with
[`figures/exec_summary.png`](figures/exec_summary.png)). Reading order:
[README.md](README.md) (entry) → [GUIDE.md](GUIDE.md) (picture-first tour) → this
document (full narrative) → [notebooks/](notebooks/) (reproduce from committed
data).*

---

## 1. Thesis — poker as decision-making under uncertainty

Poker and trading reward the same core skill: **finding and sizing edge under
uncertainty** — expected value against a noisy, adversarial background, where
*predictable deviations are exploitable and pure randomness is not.* This is a
decision-science claim, not a marketing metaphor.

**This is how a top-tier market maker actually trains.** SIG uses poker as formal
trader pedagogy — a mandatory ~100-hour requirement, with *The Mathematics of
Poker* authors (Bill Chen, Jerrod Ankenman) on staff in quant roles — to teach
"the same thought process as evaluating the expected value of a trade and pricing
risk" (references.md §4, ✓ confirmed 3-0). The poker→quant career path is
documented, not asserted — the strongest, *non-metaphorical* version of the
poker↔trading link.

**The market-microstructure analogy (motivation, not alpha).** At the
decision-theory level the parallel sharpens into market microstructure: an
exploitable, predictable opponent is the analog of *informed / toxic flow* you can
read; a uniformly-random opponent is *noise flow* you cannot. It **draws on** the
informed-vs-uninformed-trader distinction of **Kyle (1985)** and the
adverse-selection model of **Glosten & Milgrom (1985)** (references.md §3) as a
*structural parallel* — opponent modelling ↔ detecting adverse selection; Kelly /
log-bankroll growth is the shared capital-allocation objective. It is offered as
**motivation**, a hypothesis not yet tested on real order-flow data (§6).

**Honest boundary of the claim.** I do *not* claim a validated tradable signal.
The popular "toxic-flow metric" mapping (VPIN) does **not** survive scrutiny: the
specific claims that VPIN is parameter-free and an empirically validated
volatility predictor were *refuted* under adversarial verification
(references.md §3). The analogy holds at the **decision-theory** level (EV under
uncertainty, bankroll growth, adverse-selection detection), not as alpha.

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
  (baseline byte-identical), 504 tests pass, and a **multi-agent adversarial
  audit** of both the code and the figures caught and fixed overclaims *before*
  they were committed.

## 3. Honest results — what is genuinely new, and the honesty around it

Two results here are genuinely novel and statistically clean: (1) an **exact**
demonstration on Leduc Hold'em of *why* DQN-style self-play cannot reach Nash
while averaging methods can (§4, §6), and (2) a **within-player loss-aversion
asymmetry** on 777k real human hand-rows — players are more aggressive after a
loss than after an equal-size win (§6). The self-play RL agent is a resolved-but-modest edge
over a myopic baseline and, above all, an **end-to-end engineering**
demonstration — it loses head-to-head to a zero-parameter Kelly bot.

The lead figure, [`figures/exec_summary.png`](figures/exec_summary.png), shows
every claimed edge as a point with its 95% bootstrap CI — **2 of 4 resolve** (CI
excludes 0), 2 are within per-seed noise:

- **RL beats the myopic baseline** (+500 chips/match, 125/200 matches; exact
  binomial sign test **p=0.0005**, 95% bootstrap CI **[+240, +760]** — excludes
  0). The matches are binary bust outcomes, so the sign test is the right test (a
  paired t-test on the ±2000 spread agrees, p=0.0003). The win rate is stable
  across samples (64% at 50 seeds, 62.5% at 200): the 50-seed eval lacked the
  power to resolve a real ~63% edge and 200 paired seeds **do** — correct
  powering, not optional stopping. It is an edge over the *myopic baseline*, not
  the field (next bullet).
- A **belief + opponent-mix generalist** tops a cross-agent leaderboard (+209)
  and beats two of three adaptive opponents head-to-head (13-3 vs myopic, 12-4 vs
  random; 9-7 vs tilt is within noise at n=16) — but its leaderboard CI
  **includes 0** at 16 seeds, it **loses 5-11 head-to-head to a zero-parameter,
  closed-form Kelly** agent, and it ranks **6th of 17** in a tight×aggressive
  static round-robin. The RL agent is a competent generalist, not the best agent
  in the pool.
- A **risk-averse ICM/Kelly reward shows no robust edge** over a risk-neutral
  chip reward heads-up — it is *directionally negative* on the mild prize ladder
  (mean −146 chips; 95% bootstrap CI [−249, −51] excludes 0, but at n=6 that is
  suggestive, not robust).
- A rigorous **Block-B sweep** (finer action grid, un-truncated bust clip,
  tilt-bonus decoupling, snapshot self-play, warmed fold-equity) found every
  RL-*mechanics* lever works mechanically but **none moves the headline**
  ([figures/](figures/), B1–B5 panels).

This matches the field: even an **80,000-hand** human-AI match with a margin
"huge" by professional standards was only **at the edge of significance** without
variance reduction (Claudico 2015; references.md §2). Honestly-measured edges —
one that resolves over the baseline with enough paired seeds, the rest marginal
or null — are the *correct* finding for a single-developer DQN in a high-variance
game, not a failure to hide.

## 4. Where this sits vs the state of the art

DQN self-play here is a **deliberate pragmatic baseline, not a claim to compete
with the CFR family.** The superhuman poker AIs — DeepStack (2017), Libratus
(2018), **Pluribus** (2019, first superhuman multiplayer, >30 mbb/g) — are CFR /
deep-RL-plus-search with game-theoretic grounding; ReBeL (2020) extends that
deep-RL-plus-search paradigm to imperfect-information games (its own HUNL
superhuman claim did not survive adversarial verification — cite it for the
paradigm extension only, references.md §1). Plain DQN
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
  time-AVERAGE strategy's exploitability falls from **0.433 to 0.0014**
  (effectively Nash) while the greedy LAST-ITERATE — the regime DQN self-play
  plays in — plateaus near **0.355** and does not converge. This is the exact
  reason DQN self-play does not reach Nash and averaging methods do.
- **Real-data tilt validation** ([`src/real_data_tilt.py`](src/real_data_tilt.py),
  [`figures/tilt_realdata.png`](figures/tilt_realdata.png)) — the
  exploit-predictable-deviations thesis tested on **777k hand-rows** of 2009
  online play (PHH / Kim 2024, CC-BY-4.0; the NLHE hands are a redistributed
  HandHQ scrape, used for the **opponent model ONLY, never the policy** — human
  logs in the policy would make the agent exploitable). After a ≥10bb loss,
  873 players play looser and more aggressively: **VPIP +2.8pp and aggression
  +1.6pp, both 95% CIs exclude 0 — real but small.** A **symmetric within-player
  control** ([`figures/tilt_lossvswin.png`](figures/tilt_lossvswin.png)) closes the
  player-type confound (looser players both lose more and play looser): comparing
  each player's hand after a ≥10bb LOSS to their hand after an **equal-size ≥10bb
  WIN** — matching player, big-pot arousal, and event magnitude, so only the swing
  *sign* differs — the loss side is **+3.6pp more aggressive and +2.9pp looser**
  (95% CIs exclude 0; Cohen d=0.25 / 0.14; n=685 matched players; shuffled-label
  placebo ~0). That is *larger* than the vs-baseline shift, because players also
  tighten after a win — a clean **prospect-theory loss-aversion asymmetry**, not
  generic big-pot arousal. The project's emission-only
  forward-filter HMM detector (Test B) registers a small but resolved shift
  (+0.011 in P(tilted), CI excludes 0; its emission means μ_normal=0.11,
  μ_tilted=0.64 are population statistics fixed before the separation is measured,
  not tuned to it). A **separate** Baum-Welch regime HMM (Test C, not the
  detector) corroborates the phenomenon with a different method, beating a 1-state
  model out-of-sample (held-out LL +681; held-out ΔBIC +1315) with its active
  regime ~1.9× enriched for a recent big loss (4.3% vs 2.3% base; the regime fit
  used 3000 of 12320 eligible sessions — 2100 train / 900 test — a computational
  cap). Each effect is placebo-controlled (shuffled labels → ~0) and the shifts
  are honestly small (1–3pp). This **post-loss** deviation (observational, not a
  causal claim) is the poker analog of the adverse-selection signal of §1 (Kyle /
  Glosten-Milgrom) — the cross-domain mapping to real order flow remains an
  untested hypothesis — corroborated for the human-vs-bot contrast by Haaf et al.
  2021 (references.md §6).

**Remaining (the genuine next steps):**
- **Full decision-node AIVAT** — extend the chance-node (all-in) control variate
  with a *pre-committed* value function at decision nodes for the ~10–44× CI
  reduction; the precondition is to fix the heuristic before seeing the
  evaluation data (the documented footgun).
- **A neural NFSP learner on NLHE** — carry the averaging that reaches Nash on
  Leduc to the large game via a neural average-policy network, evaluated with the
  exploitability metric now in place. (DQN here stays a deliberate baseline.)
- **Test the markets analogy on real data** — the §1 connection stays a
  *motivation* on Kyle / Glosten-Milgrom (not VPIN); the genuine next step is to
  *test* the exploit-predictable-deviations idea on real order-flow data rather
  than assert it.

---

*Every empirical number above traces to committed data under
[results/](results/) and a figure under [figures/](figures/); every external
claim traces to [references.md](references.md), where claims that did **not**
survive adversarial verification are kept visible on purpose.*
