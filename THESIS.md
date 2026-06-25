# Poker as a Decision-Science Sandbox: finding edge under uncertainty, measured honestly

*Project narrative for a trader / quant-research portfolio. Bibliography:
[REFERENCES.md](REFERENCES.md). Figures: [figures/](figures/) (start with
[`figures/exec_summary.png`](figures/exec_summary.png)). Reading order:
[README.md](README.md) (entry) → [GUIDE.md](GUIDE.md) (picture-first tour) → this
document (full narrative) → [notebooks/](notebooks/) (reproduce from committed
data).*

---

## 1. Thesis: poker as decision-making under uncertainty

Poker and trading reward the same core skill: finding and sizing edge under
uncertainty. That means expected value against a noisy, adversarial background,
where *predictable deviations are exploitable and pure randomness is not.* This is a
decision-science claim, not a marketing metaphor.

### How a top-tier market maker actually trains

SIG uses poker as formal trader pedagogy: a mandatory ~100-hour requirement, with
*The Mathematics of Poker* authors (Bill Chen, Jerrod Ankenman) on staff in quant
roles, to teach "the same thought process as evaluating the expected value of a
trade and pricing risk" (REFERENCES.md §4, ✓ confirmed 3-0). The poker→quant
career path is documented, not asserted. This is the strongest, *non-metaphorical*
version of the poker↔trading link.

### The market-microstructure analogy (motivation, not alpha)

At the decision-theory level the parallel sharpens into market microstructure: an
exploitable, predictable opponent is the analog of *informed / toxic flow* you can
read; a uniformly-random opponent is *noise flow* you cannot. It draws on the
informed-vs-uninformed-trader distinction of Kyle (1985) and the adverse-selection
model of Glosten & Milgrom (1985) (REFERENCES.md §3) as a *structural parallel*:
opponent modelling corresponds to detecting adverse selection, and Kelly /
log-bankroll growth is the shared capital-allocation objective. It is offered as
motivation, a hypothesis not yet tested on real order-flow data (§6).

### Honest boundary of the claim

I do *not* claim a validated tradable signal. The popular "toxic-flow metric"
mapping (VPIN) does not survive scrutiny: the specific claims that VPIN is
parameter-free and an empirically validated volatility predictor were *refuted*
under adversarial verification (REFERENCES.md §3). The analogy holds at the
decision-theory level (EV under uncertainty, bankroll growth, adverse-selection
detection), not as alpha.

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
  **duplicate/mirror matching** (same deck, swapped seats, the variance-reduction
  protocol behind DIVAT/AIVAT, REFERENCES.md §2).
- **Engineering discipline**: every advanced feature is opt-in / default-off
  (baseline byte-identical), 513 tests pass (509 without torch), and a multi-agent adversarial
  audit of both the code and the figures caught and fixed overclaims *before*
  they were committed.

## 3. Honest results: what was built, what is reproduced, and what is new

This section covers four results: the cross-cutting thesis, the lead Leduc exploitability finding, the post-loss risk-taking asymmetry from real hand data, and the RL edge with its limits.

**Cross-cutting thesis.** Predictable behavioral deviations are an exploitable,
measurable edge. The same paired, variance-reduced methodology detects them
whether the opponent is a tilted poker player or an uninformed market participant
(Bartlett & O'Hara 2026, SSRN:6615739, cite-with-caveat; confirmed at abstract
level, REFERENCES.md §3).

**Lead finding:** a clean, exact, pedagogically-valuable reproduction of the
known last-iterate vs. time-average exploitability result on Leduc Hold'em (§4,
§6), with an independent tabular NFSP fix. Furthermore, a 0-parameter closed-form
Kelly bot beats the DQN agent head-to-head. The convergence result faithfully
reproduces known theory (Freund & Schapire 1999 on time-average convergence;
Mertikopoulos, Papadimitriou & Piliouras SODA 2018 on Poincaré recurrence /
cycling of regularized learners; Bailey & Piliouras EC 2018 on MWU divergence
toward the simplex boundary; Daskalakis & Panageas ITCS 2019 on OMWU last-iterate
convergence; all in REFERENCES.md §7). It is *not* a novel theorem. The novelty
boundary is the exact numerical confirmation on Leduc with an independent
tabular-NFSP fix.

**Supporting finding:** a within-player post-loss risk-taking asymmetry on 777k real
human hand-rows. Players are +3.6 pp more aggressive and +2.9 pp looser after a
realized loss than after an equal-size win (Cohen d=0.25/0.14, n=685 matched
players, §6). The effect is small but real, well-calibrated to the behavioral
literature (Coval & Shumway 2005, ⚠ abstract-level claim only; Haaf et al. 2021
, REFERENCES.md §6), and
statistically disciplined (within-player paired design, bootstrap CIs). The
"novel" claim for this result is the within-player paired design on this dataset.
The directional finding is consistent with, not independent of, prior work.

The self-play RL agent is a resolved-but-modest edge over a myopic baseline and,
above all, an end-to-end engineering demonstration. It loses head-to-head to
a zero-parameter Kelly bot.

The lead figure, [`figures/exec_summary.png`](figures/exec_summary.png), shows
every claimed edge as a point with its 95% bootstrap CI. 2 of 4 resolve (CI
excludes 0), 2 are within per-seed noise:

- **RL beats the myopic baseline (pre-registered confirmatory).** The
  [pre-registered](PREREGISTRATION.md) confirmatory protocol (500 mirrored paired
  seeds × 100 hands via `evaluate_matchup`, PREREGISTRATION.md §4.3/§4.6) resolves
  a mean **+256 chips/match**, 95% bootstrap CI **[+144, +364]** (excludes 0;
  exact binomial sign test **p≈7×10⁻⁶** on the 200 decisive seeds, 132–68; the
  mirror splits the other 300 on pure seat-luck, which is exactly what duplicate
  matching is for; paired t agrees, p≈5×10⁻⁶). The single-orientation raw
  calibration arm agrees (+400 chips, CI [+232, +576]).

  **Robust across training seeds (v2 Phase 0).** That +256 rests on one training
  seed (`torch_seed=0`). Retraining the *identical* recipe over 20 seeds — eval
  protocol held byte-identical, only the seed varied, pre-registered before the run
  (PREREGISTRATION.md §10) — gives a per-seed edge of **mean +351, median +300,
  across-seed 95% CI [+244, +468]**: 16 of 20 seeds individually resolve a positive
  edge, **none resolve negative**. Seed 0 reproduces +256 exactly and sits at the
  **35th percentile, *below* the median**, so the published number was a
  conservative draw, not a cherry-picked one. This closes the single-training-seed
  limitation §0 of the pre-registration flagged
  ([`figures/seed_sweep.png`](figures/seed_sweep.png)).

  An earlier exploratory pilot (`evaluate_vs_baseline`, 200 seeds × 200 hands,
  single orientation) first surfaced the edge at +500 (125/200, CI [+240, +760]).
  The confirmatory number is smaller because mirror matching removes the
  seat-correlated deck luck the pilot left in.

  The pilot's 50-seed training monitor is noisy, not separate evidence. It was
  significantly negative mid-training (15/50, −800 at step 500) before recovering
  to 33/50 at the final checkpoint. The resolved claim rests on the 200-seed pilot
  and the separate 500-seed mirrored confirmatory, not on the 50-seed reading. It
  is an edge over the *myopic baseline*, not the field (next bullet).

- A **belief + opponent-mix generalist** tops a cross-agent leaderboard (+209)
  and beats two of its three pool opponents head-to-head (13-3 vs myopic, 12-4 vs
  random; 9-7 vs tilt is within noise at n=16). Its leaderboard CI
  **includes 0** at 16 seeds, it **loses 5-11 head-to-head to a zero-parameter,
  closed-form Kelly** agent, and it ranks **6th of 17** in a tight×aggressive
  static round-robin. The RL agent is a competent generalist, not the best agent
  in the pool.
- A **risk-averse ICM/Kelly reward shows no robust edge** over a risk-neutral
  chip reward heads-up. It is *directionally negative* on the mild prize ladder
  (mean −146 chips; 95% bootstrap CI [−249, −51] excludes 0, but at n=6 that is
  suggestive, not robust).
- A rigorous **Block-B sweep** (finer action grid, un-truncated bust clip,
  tilt-bonus decoupling, snapshot self-play, warmed fold-equity) found every
  RL-*mechanics* lever works mechanically but **none moves the headline**
  ([figures/](figures/), B1–B5 panels).

One resolved edge over the baseline and the rest marginal or null. For a
single-developer DQN in a high-variance game, this is the expected measurement.
It matches what the field shows: even an **80,000-hand** human-AI match with a
margin "huge" by professional standards was only at the edge of significance
without variance reduction (Claudico 2015; REFERENCES.md §2). The result is
reported as such.

## 4. Where this sits vs the state of the art

DQN self-play here is a **deliberate pragmatic baseline, not a claim to compete
with the CFR family.** The superhuman poker AIs (DeepStack (2017), Libratus
(2018), **Pluribus** (2019, first superhuman multiplayer, >30 mbb/g)) are CFR /
deep-RL-plus-search with game-theoretic grounding. ReBeL (2020) extends that
deep-RL-plus-search paradigm to imperfect-information games (its own HUNL
superhuman claim did not survive adversarial verification; cite it for the
paradigm extension only, REFERENCES.md §1). Plain DQN
self-play even **violates the stationarity assumption** Q-learning needs (the
opponent moves while you learn). This repo makes that concrete and *exact* on
Leduc Hold'em ([`figures/exploitability.png`](figures/exploitability.png)): the
time-AVERAGE strategy's exploitability falls toward 0 (Nash), but the greedy
LAST-ITERATE stays exploitable and does not converge. An **independent
tabular Q-learning self-play** (an actual DQN-regime learner, not a proxy)
confirms it directly, its greedy last-iterate oscillating around **3.40** and
never approaching Nash.

This last-iterate vs. time-average gap is **known theory**, faithfully reproduced
here with exact Leduc exploitability numbers. Freund & Schapire (1999) proved
time-average convergence of no-regret dynamics to Nash in zero-sum games;
Mertikopoulos et al. (SODA 2018) showed regularized learners / Mirror Descent /
FTRL exhibit Poincaré recurrence (cycling); Bailey & Piliouras (EC 2018) showed
MWU iterates diverge monotonically toward the simplex boundary; Daskalakis &
Panageas (ITCS 2019) proved OMWU achieves last-iterate convergence (REFERENCES.md
§7). The contribution here is a clean, exact, end-to-end pedagogical
demonstration on Leduc with a tabular-NFSP fix, not a novel theorem.

**Neural NFSP** (Heinrich & Silver 2016) brings exactly this averaging to large
games via a neural average-policy network (REFERENCES.md §1). v2 Phase 2 implements
it ([`src/leduc_neural_nfsp.py`](src/leduc_neural_nfsp.py)): two MLPs over a
structured 21-d info-set feature, scored by the *same* exact NashConv metric as the
tabular learners, pre-registered before the run (PREREGISTRATION.md §11). The honest
result over 5 seeds: neural NFSP **converges** (exploitability 4.75 → mean **1.46**
at 200k episodes) but does **not** beat tabular NFSP on Leduc — it wins only at the
smallest budget (50k: 1.94 vs 2.40) and tabular edges ahead from 100k on
([`figures/neural_nfsp.png`](figures/neural_nfsp.png)). This is the expected null on
a game small enough to tabulate exactly: neural function approximation has no edge
where the tabular policy is already exact. Its value — generalising across info-sets
in games too large to tabulate — is the next step (a larger game scored by a
validated LBR lower bound, where "beats tabular" becomes meaningful). Notably, a
single-seed look briefly *suggested* a neural win at 100k; the 5-seed mean erased it,
the same multi-seed discipline §10 (the Phase 0 robustness sweep) established.

## 5. Honest-negative as a feature, not a bug

The deliverable is **method + intellectual honesty**: reproducible, paired,
variance-accounted, adversarially-audited, with null/marginal results reported as
such. *(Caveat: that quant firms specifically reward null results is practitioner
signal, not documented policy. The defensible claim is that rigour + honesty
signal statistical maturity, REFERENCES.md §5.)*

The confirmatory analysis plan (frozen seeds, opponent configuration, test
files, and the power calculation) is **pre-registered** in
[`PREREGISTRATION.md`](PREREGISTRATION.md) (registered *concurrently*, dated
2026-06-23; §0 states plainly that nothing is claimed to have been registered
earlier). The pre-registered RL-vs-myopic confirmatory run **has now been executed
exactly as frozen** (500 mirrored seeds × 100 hands; +256 chips, CI [+144, +364],
resolved; see PREREGISTRATION.md §4.6, [`results/confirmatory.json`](results/confirmatory.json)),
with the outcome reported in full regardless of magnitude (§8). The plan, the run
script, and the result are committed together (the registration is concurrent, not
sequenced ahead of the run in git history; PREREGISTRATION.md §0 says so plainly).
What makes the posture honest is not a git-provable time gap but the pre-committed
rule to report whatever the frozen protocol returns. It returned an edge
*smaller* than the exploratory pilot, reported as such rather than hidden. The
later multi-seed robustness sweep (PREREGISTRATION.md §10) goes one step further:
its protocol was committed in a *separate, earlier* commit than its result, so for
that claim the freeze-before-run gap **is** git-provable — and it confirmed the
+256 headline is representative across training seeds, not a single-seed artifact.

## 6. What the rigor layer ships, and what remains

**Shipped (the deep-research rigor recommendations, REFERENCES.md §1–§2):**
- **Variance reduction:** bootstrap CIs, duplicate/mirror matching (cuts the
  heads-up CI to ~65% of raw), and the **all-in EV control variate** (the
  AIVAT-family chance-node adjustment; unbiased, chip-conserving, opt-in). See
  [`figures/variance_reduction.png`](figures/variance_reduction.png) and
  [`figures/exec_summary.png`](figures/exec_summary.png).
- **An exact exploitability metric** ([`src/leduc_eval.py`](src/leduc_eval.py))
  for any Leduc strategy, and the verifiable demonstration
  ([`figures/exploitability.png`](figures/exploitability.png)) that the
  time-AVERAGE strategy's exploitability falls from **0.695 to 0.009**
  (toward Nash) while the greedy LAST-ITERATE (the regime DQN self-play
  plays in) stays exploitable around **2.2** and does not converge. An **independent
  tabular Q-learning self-play** (a tabular analog of the DQN regime: greedy,
  off-policy, no averaging) exhibits the same behavior: its greedy last-iterate
  oscillates around **3.40** (range [1.70, 5.53] over 1M+ episodes) and never
  approaches Nash. This is genuine non-convergence of the greedy iterate, not a
  CFR artifact. It illustrates why a greedy, non-averaging self-play learner (the
  family DQN self-play belongs to) need not reach Nash while averaging methods do.

  **Correction history.** An earlier version of this solver and metric were both
  wrong in mutually-masking ways. A sign error in the CFR round-1→round-2
  transition made the time-average converge to a degenerate all-call profile (a
  King that never raised). A reachability lock-out in the best-response made
  the metric underestimate exploitability, even returning impossible negative
  NashConv values. Both errors surfaced while baselining NFSP against the metric.
  They were pinned down by three independent reimplementations of the game and
  best response, all agreeing to machine precision, plus a four-way adversarial
  verification. After fixing both, the corrected game value is ≈**−0.0862**, which
  matches the ~−0.0856 known for Leduc. The average strategy's exploitability
  now strictly decreases toward 0. The qualitative claim held throughout. Only
  the magnitudes were wrong. The catch is the rigor layer working.

- **Tabular NFSP on Leduc** ([`src/leduc_nfsp.py`](src/leduc_nfsp.py)): the
  *learned* averaging method that closes the loop above, in-repo. It is the SAME
  value learner as the Q-curve (identical α, ε, γ) with ONLY NFSP's policy-
  averaging added (Heinrich & Silver 2016): its AVERAGE policy's exploitability
  falls **~2.40 → ~0.86** over 50k → 1M episodes, converging toward Nash on the
  same exact metric on which the greedy last-iterate oscillates around 3.40.
  It remains sample-based, so it stays above CFR's full-enumeration ~0.01. Adding only
  averaging flips non-convergence into convergence. The fourth curve on
  [`figures/exploitability.png`](figures/exploitability.png) shows it directly.
- **Real-data tilt validation** ([`src/real_data_tilt.py`](src/real_data_tilt.py),
  [`figures/tilt_realdata.png`](figures/tilt_realdata.png)): the
  exploit-predictable-deviations thesis tested on **777k hand-rows** of 2009
  online play (PHH / Kim 2024, CC-BY-4.0; the NLHE hands are a redistributed
  HandHQ scrape, used for the **opponent model ONLY, never the policy**, as
  human logs in the policy would make the agent exploitable). After a ≥10bb loss,
  873 players play looser and more aggressively: **VPIP +2.8pp and aggression
  +1.6pp, both 95% CIs exclude 0. Real but small.** A **symmetric within-player
  control** ([`figures/tilt_lossvswin.png`](figures/tilt_lossvswin.png)) closes the
  player-type confound (looser players both lose more and play looser): comparing
  each player's hand after a ≥10bb LOSS to their hand after an **equal-size ≥10bb
  WIN**, matching player, big-pot arousal, and event magnitude, so only the swing
  *sign* differs. The loss side is **+3.6pp more aggressive and +2.9pp looser**
  (95% CIs exclude 0; Cohen d=0.25 / 0.14; n=685 matched players; shuffled-label
  placebo ~0). That is *larger* than the vs-baseline shift, because players also
  tighten after a win. This is a clean **within-player post-loss risk-taking asymmetry**,
  not generic big-pot arousal. The label is deliberately descriptive: the mechanism is
  discussed below, and the finding runs opposite to the realization effect, so it is not
  asserted to be prospect-theory loss aversion.

  **Positioning against the behavioral literature.** The ≥10bb chip loss in this
  design is a *realized* loss (chips removed from stack at showdown), which puts
  it squarely in the domain of Imas (2016, AER 106(8):2086–2109): the realization
  effect predicts that after a *realized* loss people take **less** risk, while
  paper (unrealized) losses produce loss-chasing. The finding here is rising
  aggression after a realized chip loss, which runs in the **opposite** direction
  to the Imas prediction, placing it closer to Coval & Shumway (2005, J. Finance
  60(1):1–34), who document CBOT proprietary traders becoming ~16% more likely to
  take above-average afternoon risk after morning losses. Rather than weakening the
  contribution, this tension sharpens it: the result is not a replication of
  a known effect but a potentially distinct phenomenon (emotional loss-chasing
  under competitive-game pressure), where the social and competitive context of a
  poker session may override the realization-effect brake. The mechanism
  distinguishing the two predictions (realized-loss risk-reduction vs.
  competitive-context loss-chasing) is an open question, not a resolved claim.

  **On lambda=2.25.** Any loss-aversion reading of the asymmetry above does not depend on
  the canonical Tversky-Kahneman (1992) loss-aversion coefficient of λ=2.25.
  That estimate comes from a small, unincentivized student sample, and recent
  meta-analyses report substantially lower and widely heterogeneous estimates.
  The broad verdict is that 2.25 is probably too high. The qualitative asymmetry
  (losses loom larger than equal-size gains) is well-supported; the precise
  magnitude is contested, and this thesis does not rely on 2.25 as a fixed
  parameter. (Specific meta-analytic point estimates are deliberately not quoted
  here. They were not primary-source-verified in this project's reference sweep.)

  **Pre-registered confound controls** (PREREGISTRATION.md §5.3)**.** Three
  confounds could explain the asymmetry without any loss-aversion mechanism, and
  the project pre-commits to reporting honestly if they absorb the effect:
  (i) *Session time-of-day:* fatigue in late-session hands could co-vary with
  cumulative loss position; the primary robustness check is to re-run the
  within-player comparison controlling for hand-number within session and
  time-of-day quantile. If the asymmetry vanishes under time-of-day control,
  that null will be reported.
  (ii) *Stake-size:* the 10bb threshold is fixed, but its real-money magnitude
  varies by stake; the check is to stratify by stake level and test whether the
  effect persists in each stratum independently.
  (iii) *Survivorship / selection:* players who remain after a big loss are
  those who kept playing; session-level dropout is a residual threat even in the
  within-player matched design. The project will report the fraction of eligible
  loss events where the player left the session immediately, and test robustness
  after excluding sessions with very few post-event hands.

  The project's emission-only
  forward-filter HMM detector (Test B) registers a small but resolved shift
  (+0.011 in P(tilted), CI excludes 0; its emission means μ_normal=0.11,
  μ_tilted=0.643 are population statistics fixed before the separation is measured,
  not tuned to it). A **separate** Baum-Welch regime HMM (Test C, not the
  detector) corroborates the phenomenon with a different method, beating a 1-state
  model out-of-sample (held-out LL +681; held-out ΔBIC +1315) with its active
  regime ~1.9× enriched for a recent big loss (4.3% vs 2.3% base; the regime fit
  used 3000 of 12320 eligible sessions (2100 train / 900 test), a computational
  cap). Each effect is placebo-controlled (shuffled labels → ~0) and the shifts
  are honestly small (1–3pp). This **post-loss** deviation (observational, not a
  causal claim) is the poker analog of the adverse-selection signal of §1 (Kyle /
  Glosten-Milgrom). The cross-domain mapping to real order flow remains an
  untested hypothesis, corroborated for the human-vs-bot contrast by Haaf et al.
  2021 (REFERENCES.md §6).

**Remaining (the genuine next steps):**
- **Full decision-node AIVAT:** extend the chance-node (all-in) control variate
  with a *pre-committed* value function at decision nodes for the ~10–44×
  sample-size reduction (fewer hands for equivalent significance, REFERENCES.md
  §2; equivalently ~3–7× CI-width narrowing, since CI scales as 1/√n); the
  precondition is to fix the heuristic before seeing the evaluation data (the
  documented footgun).
- **A neural NFSP learner on NLHE:** the tabular version above already reaches
  Nash on Leduc; the next step carries that averaging to the large game via a
  *neural* average-policy network, evaluated with the same exploitability metric.
  (DQN here stays a deliberate baseline.)
- **Test the markets analogy on real data:** the §1 connection stays a
  *motivation* on Kyle / Glosten-Milgrom (not VPIN); the genuine next step is to
  *test* the exploit-predictable-deviations idea on real order-flow data rather
  than assert it. A concrete backtest design exists in local working notes and
  will be committed to the repo before any data is pulled. Until then, treat it
  as a planned design, not a registered one.

  **Microstructure lead (motivation, untested on real order-flow).** A recent
  working paper, Bartlett & O'Hara, *"Adverse Selection in Prediction Markets:
  Evidence from Kalshi"* (SSRN 6615739, April 2026), finds, from 41.6 million
  Kalshi trades, that traders systematically overbet YES in markets that
  predominantly settle NO, generating a "behavioral surplus" that
  cross-subsidizes adverse selection; separately, an adapted VPIN predicts
  maker losses in single-name markets but not in broad-based markets. These are
  abstract-level findings confirmed from the Stanford Law School press page;
  granular figures cited in press coverage are unverified at the primary-source
  level and are not repeated here as fact. The honest framing for this project: the
  YES-overbetting / behavioral-surplus channel maps to poker **tilt** (both are
  predictable behavioral deviations, not information asymmetry), while the
  OFI/VPIN adverse-selection channel maps to the Kyle / Glosten-Milgrom
  informed-opponent framing of §1. Conflating the two is a category error. The
  planned backtest tests both channels separately via Kalshi's official trades
  API, feeding results into the existing `src/stats.py` bootstrap/paired harness
  (`bootstrap_ci`, `paired_t_test`), with a planned pivot rule: if the
  behavioral-surplus channel does not survive transaction costs and out-of-sample
  holdout, pivot to OFI on equity TAQ data. No Kalshi data has been pulled and
  no backtest result exists yet.

---

*Every empirical number above traces to committed data under
[results/](results/) and a figure under [figures/](figures/); every external
claim traces to [REFERENCES.md](REFERENCES.md), where claims that did **not**
survive adversarial verification are kept visible on purpose.*
