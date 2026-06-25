# Pre-Registration: Confirmatory Evaluation Plan

**Repository:** poker-simulation  
**Branch at time of registration:** rl-multihand-episodes  
**Registered:** 2026-06-23  
**Purpose:** Lock the analysis plan and acceptance criteria before re-running evaluation, to prevent garden-of-forking-paths inflation of Type-I error (Gelman & Loken 2014, "The Statistical Crisis in Science," *American Scientist* 102(6):460–465).

## 0. Timing and scope

This document IS the pre-registration. It describes the harness as it exists in the repository and freezes the confirmatory analysis. **Nothing here is claimed to have been pre-registered earlier in git history.** The registration is **concurrent**, not sequenced ahead of the run: this document, the run script (`scripts/measure_confirmatory.py`), and the outcome (§4.6, `results/confirmatory.json`) are committed *together*: there is deliberately no git-provable time gap between freezing the protocol and running it. What the registration buys is therefore **not** a temporal-precedence claim but a **pre-committed reporting rule**: the frozen protocol (§4.3) is run once and its result reported in full regardless of direction (§8). The exploratory pilot (`results/headline_history.json`) informed the *design*; the confirmatory run tests the frozen protocol and is reported here whatever it returned (it returned a smaller, still-resolved edge).

**What this registration does not establish.** Two honest limits follow from the concurrent design. First, the execution trace is not committed, so an external reader cannot verify from the repository that the frozen protocol was run exactly once rather than re-run until it returned a positive result. The safeguard is the pre-committed reporting rule and the single committed outcome, not an auditable run log. Second, the RL checkpoint fixes `torch_seed=0`, but the training-seed sweep that would show this seed was not chosen after seeing favorable results is not committed, so seed-selection independence cannot be externally verified either. Both are flagged here rather than hidden. The second limitation is now closed: the multi-seed training sweep pre-registered in §10 ships in two commits with a git-provable gap — the frozen protocol and verdict rule (§10.3) first, with no results, then the outcome (§10.5, `results/seed_sweep.json`) — so the freeze-before-run ordering, and therefore seed-selection independence for the sweep, is externally verifiable from git history. The first limitation (no committed execution trace for the original §4 confirmatory) remains a goal for a future run.

---

## 1. Background: Why Pre-Registration Matters Here

Poker evaluation is a high-variance sequential process. Even with a genuine edge, single-run point estimates are unreliable at small sample sizes; and post-hoc selection of seeds, opponent configurations, or CI methods inflates false-positive rates in ways that look like principled choices. Gelman & Loken (2014) call this the "garden of forking paths": each undocumented decision (which seeds to use, whether to apply variance reduction, which test statistic to report) is a fork that, collectively, can push p below 0.05 even under a true null. Pre-registration closes the garden by committing all decisions before looking at fresh results.

---

## 2. The Harness: What the Repo Actually Does

### 2.1 Paired-seed design

Each matchup is evaluated over a **list of integer seeds** (`seeds: List[int]`) passed to `evaluate_matchup` in `src/evaluation.py`. For each seed, one seeded heads-up session is run with agent A seated as `player_id=1` and agent B as `player_id=2`. The per-seed chip difference `net_a − net_b` is one paired observation. The deck, all Monte Carlo sampling, and every bot's internal RNG are all driven by a single `random.Random(seed)` (see `_play_match`, `evaluation.py` lines 85–110 and `_wire_rng`, `tournament.py` lines 211–226), so each seed is a fully reproducible, self-contained experiment.

For round-robin evaluation, per-pair seeds are generated deterministically by `_pair_seed(name_a, name_b, seed_index)` (`tournament.py` lines 30–43): a polynomial hash over the lexicographically sorted pair names, so `(A, B)` and `(B, A)` map to the same seed and results are symmetric.

### 2.2 Variance-reduction techniques (opt-in, composable)

Two variance-reduction techniques are implemented in `evaluate_matchup` (`src/evaluation.py` lines 113–167):

**Mirror matching (`mirror=True`):** The same seed is replayed with the two agents in swapped seats; agent A's result is averaged over both orientations. This cancels position-correlated deck luck. In the committed benchmark (`results/variance_reduction.json`, 120 seeds × 100 hands), mirror matching reduced the 95% bootstrap CI width from 712 chips to 464.3 chips, approximately 65% of the raw width (ratio 0.652).

**All-in EV control variate (`luck_adjusted=True`):** All-in pots are scored by their equity-weighted expected value rather than the realised runout, removing board-runout chance variance. This is an AIVAT-family chance-node adjustment (see `REFERENCES.md §2`, Burch et al. 2018). In isolation, the luck adjustment reduced CI width only modestly (ratio 0.988 in the benchmark), because match-outcome variance in this bust-match format is dominated by path-dependent bust events, not single-hand runout luck. Full decision-node AIVAT remains a future step.

**Both combined (`mirror=True, luck_adjusted=True`):** CI width ratio 0.644 in the benchmark (marginally better than mirror alone).

The primary variance-reduction mode for all confirmatory tests is **`mirror=True`**, optionally with `luck_adjusted=True`. The raw arm is always run in parallel as a calibration check.

### 2.3 Chip conservation assertion

Every session asserts `total_after == total_before` at the `_play_match` level (`evaluation.py` lines 104–106). This is also tested at the unit level in:

- `tests/test_phase0.py`: `TestChipConservation`, parametrized over `n_players ∈ {2, 4, 6, 9}`, 50 seeded hands each.
- `tests/test_phase2.py`: `test_chip_conservation_with_mc_bots`.
- `tests/test_tournament.py`: `test_chip_conservation_direct` and `test_chip_conservation_via_tournament`.
- `tests/test_icm_eval.py`: `test_chip_conservation_over_tournament`.
- `tests/test_phase_c_deepen.py`: `test_icm_trainer_chip_conservation`.

A failing chip conservation assertion is a hard error and invalidates the run.

### 2.4 Statistical tests

`src/stats.py` implements three test statistics:

- **Paired t-test** (`paired_t_test`, lines 209–237): one-sample t-test of the per-seed diffs against zero, using scipy's exact t-distribution when available and a normal approximation otherwise.
- **Percentile bootstrap CI** (`bootstrap_ci`, lines 175–206): 10,000 resamples, seeded at 12345, returning `{mean, lo, hi, ci, n}`. Distribution-free; the appropriate uncertainty statement for heavy-tailed poker PnL.
- **Binomial sign test** (`binomial_sign_test`, lines 240–271): exact two-sided binomial test on the win/loss counts (ties dropped). The appropriate test for bankroll-bust matches where each per-seed diff is effectively ±2·stack.

All three statistics are reported. The **primary test statistic is the bootstrap 95% CI** for the mean edge in chips per 100-hand session. A result is declared significant if the bootstrap CI excludes zero. The t-test and sign test serve as cross-checks.

---

## 3. Power Calculation

The high variance of poker outcomes makes power analysis essential. With per-seed chip diffs that are approximately ±2·stack (≈ ±2000 chips per 100-hand bust match), the standard deviation of a single paired observation is on the order of the stack itself. Translating to the canonical bb/100 metric (1 bb = 20 chips, 100-hand session):

**SD ≈ 90 bb/100** is a plausible order-of-magnitude estimate for per-session PnL in a heads-up bust-match format with 100-hand sessions. This is not a confirmed measurement from this repository and is flagged as unverified.

Under the normal approximation, the number of seeds n needed for a 95% confidence interval with half-width δ is approximately:

```
n ≈ (z_{0.975} × SD / δ)² = (1.96 × SD / δ)²
```

For illustrative targets (with SD ≈ 90 bb/100):

| Target CI half-width (δ) | Approximate n needed |
|---|---|
| ±10 bb/100 (rough signal) | ~312 seeds |
| ±5 bb/100 (moderately tight) | ~1,245 seeds |
| ±1.8 bb/100 (tight estimate) | ~9,604 seeds (~100k hands) |
| ±1.0 bb/100 (publication-grade) | ~31,116 seeds (~3M hands) |

At 1 sigma (detecting a 5 bb/100 edge with SNR = 1): n ≈ (90/5)² × 1 = 324 seeds. Reaching SNR = 2 (roughly 95% one-sided power) requires n ≈ 1,296 seeds; SNR = 3 (99% power) requires n ≈ 2,916 seeds. These are all raw-arm figures; the **mirror arm cuts the CI width to ~65% of raw** (ratio 0.652 in the committed benchmark). Because the required n scales as the *square* of the CI width (n ∝ σ² ∝ CI²), that reduces the required n to **~43%** (factor 0.652² ≈ 0.43), not 65%.

**Why this matters:** The current benchmark uses 120 seeds × 100 hands per arm (`results/variance_reduction.json`). That is enough to verify the variance-reduction mechanism and detect large edges (>20 bb/100) but not enough to resolve a 5 bb/100 edge at high confidence. The paired/variance-reduction design is essential: without it, reaching a tight CI estimate would require over 500k hands; with mirror matching, the same CI width is achievable in **~43% as many seeds** (because n ∝ CI², factor 0.652² ≈ 0.43).

**Null and marginal results are a feature.** An adequately powered study that finds no significant edge is informative: it rules out the claimed edge at the tested sample size. A pre-registered null result is more scientifically valuable than an underpowered positive result that may not replicate.

---

## 4. Confirmatory Analysis Plan

The following is locked before any fresh evaluation runs. Deviation from this plan must be documented as exploratory.

### 4.1 Frozen seeds

Confirmatory seeds are the integer range `list(range(N_seeds))` where `N_seeds` is set once at the start of a run and not adjusted after inspecting results. The value of `N_seeds` is committed in the run script before execution. Seeds are not selected, filtered, or extended post-hoc.

For the primary RL-vs-baseline matchup, the confirmatory seed set is **`list(range(500))`** (500 paired seeds × 100 hands = 50,000 hands per arm, mirrored). This targets a CI half-width of approximately ±7–10 bb/100 after mirror variance reduction.

### 4.2 Frozen opponent configuration

The baseline opponent is `BotPlayer` with `tight_threshold=0.2, aggression=0.5` and `MonteCarloEngine(n_simulations=100)`. This matches the "Myopic" arm in `scripts/measure_variance_reduction.py`. The RL agent configuration (checkpoint path, network architecture) is committed before the confirmatory run. No opponent parameters are tuned after inspecting any seed's outcome.

### 4.3 Frozen evaluation call

```python
# Confirmatory call: do not change after registration
result = evaluate_matchup(
    factory_rl, factory_myopic,
    name_a="RL", name_b="Myopic",
    seeds=list(range(500)),
    n_hands=100,
    mirror=True,
    luck_adjusted=False,  # primary arm; luck_adjusted arm is exploratory
)
ci = bootstrap_ci(result.diffs)  # seed=12345, n_resamples=10000
```

### 4.4 Frozen test files (confirmatory analysis)

The following test files constitute the confirmatory analysis and must pass without modification:

| File | What it confirms |
|---|---|
| `tests/test_phase0.py` | Seeded reproducibility; chip conservation at 2/4/6/9 players |
| `tests/test_phase2.py` | Monte Carlo bot integration; chip conservation with MC |
| `tests/test_tournament.py` | Paired-seed determinism; chip conservation in round-robin |
| `tests/test_evaluation.py` | evaluate_matchup structure; zero-sum pairing; bootstrap CI |
| `tests/test_icm_eval.py` | Chip conservation through ICM evaluation |
| `tests/test_phase_c_deepen.py` | Chip conservation through ICM trainer |

These tests are run with `python -m pytest tests/ -v` before and after every confirmatory evaluation. A regression in any of these files invalidates the run.

### 4.5 Primary outcome

**Primary outcome:** The mean edge of the RL agent over the Myopic baseline, in chips per 100-hand session, with its 95% bootstrap CI (seed=12345, 10,000 resamples).

**Confirmatory decision rule:** The edge is considered established if the bootstrap CI excludes zero at the 95% level. The result is reported in full regardless of sign: a CI that brackets zero is reported as a null/marginal result, not suppressed or extended post-hoc. **Executed result: see §4.6.** The edge is resolved at +256 chips, CI [+144, +364].

**Secondary outcomes (reported but not the basis for the confirmatory claim):**
- Paired t-test p-value.
- Binomial sign test p-value on win/loss counts.
- CI width ratio (mirror vs raw) as a realized variance-reduction check.
- Per-seed PnL distribution (histogram and box plot).

### 4.6 Execution status and outcome

This pre-registered call **has now been executed** by
`scripts/measure_confirmatory.py`, exactly as frozen above (`seeds=list(range(500))`,
`n_hands=100`, `mirror=True`, `luck_adjusted=False`; baseline `BotPlayer(tight_threshold=0.2,
aggression=0.5)` with `MonteCarloEngine(100)`). The RL policy is the same
fixed-vs-myopic agent as the exploratory pilot, retrained deterministically
(`torch.manual_seed(0)`, `SelfPlayTrainer(seed=1, opponent_mode="fixed", …)`,
1500 steps) and confirmed identical by reproducing the pilot's
`evaluate_vs_baseline` number **exactly** (125/200, +500 chips, CI [+240, +760])
before the confirmatory call was run; the checkpoint is committed at `models/confirmatory_rl.pt`
(force-added past the `*.pt` gitignore so the exact artifact ships with the repo).

**Confirmatory outcome (`results/confirmatory.json`), reported in full per §8:**
the RL agent beats the Myopic baseline by a mean **+256 chips/match** over the
500 mirrored paired seeds, 95% bootstrap CI **[+144, +364]** (excludes zero →
the edge is **resolved** by the §4.5 decision rule). Of the 500 mirrored seeds RL
finishes ahead in 132 and behind in 68; the other 300 split between the two
seatings (the mirror exposing them as pure seat-luck), so the exact binomial sign
test on the 200 decisive seeds gives **p ≈ 7.1×10⁻⁶** and the paired t-test
**p ≈ 5.0×10⁻⁶**. The raw single-orientation calibration arm agrees in direction
and magnitude: **+400 chips, CI [+232, +576]**. The confirmatory edge is smaller
than the +500 pilot, as expected, since mirror matching removes the
seat-correlated deck luck the single-orientation pilot left in.

**Seed-range transparency.** The confirmatory uses seeds `0..499`; the exploratory
pilot used seeds `0..199`, so 200 of the 500 confirmatory seeds overlap with the
pilot, in the same direction. This is disclosed for completeness: it means the
seed *range* was not chosen independently of having seen a positive pilot signal,
a mild winner's-curse consideration. Three facts bound its impact: (i) training
never saw seeds `0..499` (in-training eval uses seeds `1000+`), so this is not
train/test contamination; (ii) the pilot and confirmatory use *different* protocols
(200 hands single-orientation vs 100 hands mirrored), so a given seed is not the
same experiment; and (iii) the effect is large and far from the decision boundary
(p ≈ 7×10⁻⁶). A future confirmatory should pre-commit a disjoint seed block (e.g.
`500..999`) to remove the consideration entirely.

---

## 5. Tilt / Real-Data Analysis

The real-data tilt analysis is **exploratory** with respect to causal claims and **confirmatory** with respect to the measurement pipeline.

### 5.1 What is confirmed

The pipeline in `src/real_data_tilt.py` and `scripts/measure_tilt_realdata.py` is frozen. Confirmatory claims are:

- The within-player loss-vs-win control (`within_player_loss_vs_win`, lines 331–395 of `src/real_data_tilt.py`), the symmetric, stronger design distinct from the model-free `phenomenon_test` (lines 237–282, which yields the raw n=873 post-loss numbers), uses a within-player paired design with a minimum of 5 hands per group, a loss threshold of 10 bb, and a bootstrap CI with 10,000 resamples.
- The placebo control is a shuffled-label version of the same data under `PLACEBO_SEED = 12345` (`scripts/measure_tilt_realdata.py` line 38).
- The dataset is the 120 `.phhs` files in `data/phh/` from the Kim 2024 PHH dataset (Zenodo DOI 10.5281/zenodo.13997158), parsed by `src/real_data_tilt.py::parse_phhs`.
- Committed measurements (from `results/tilt_realdata.json`): n=685 matched players, 777,687 human hand-rows, within-player aggression shift +3.6 pp (Cohen d=0.25), VPIP shift +2.9 pp (Cohen d=0.14).

### 5.2 What is exploratory

- Any causal interpretation (tilt vs. opponent adaptation, selection effects).
- Any extension to a different dataset or loss threshold.
- The HMM regime-fit result (BIC-based comparison, `fit_regime_hmm`).

### 5.3 Planned confound robustness checks (tilt asymmetry)

The within-player loss-vs-win asymmetry (§5.1) is observational. The following confound checks are pre-committed *before* they are run; if any absorbs the effect, the resulting null is reported in full per §8. None has been executed yet.

- **Session time-of-day / fatigue:** re-run the within-player comparison controlling for hand-number within session and time-of-day quantile.
- **Stake-size:** stratify by stake level and test whether the effect persists within each stratum independently.
- **Survivorship / selection:** report the fraction of eligible loss events where the player left the session immediately, and test robustness after excluding sessions with very few post-event hands.

---

## 6. Exploitability / Leduc Benchmarks

The Leduc exploitability benchmarks are **confirmatory** reproducibility checks, not hypothesis tests.

**Frozen measurement script:** `scripts/measure_exploitability.py`  
**Frozen result file:** `results/exploitability.json`  
**Frozen test file:** `tests/test_phase_d_leduc.py`

Committed reference values (from `results/exploitability.json`):

| Metric | Value |
|---|---|
| CFR average exploitability at 10 iterations | 0.695 |
| CFR average exploitability at 10,000 iterations | 0.009 |
| CFR last-iterate exploitability at 10,000 iterations | 2.221 |
| Tabular Q-learning greedy last-iterate mean (1M+ episodes) | 3.40 (range [1.70, 5.53]) |
| Tabular NFSP average policy exploitability: 50k → 1M episodes | 2.40 → 0.86 |
| Uniform strategy exploitability | 4.747 |

These values are the ground truth for solver correctness checks. Any re-run that differs by more than rounding (last decimal place) must be explained before the result is accepted.

**Lower-bound exploitability (LBR).** For games too large for exact NashConv, `src/leduc_lbr.py` provides a Local Best Response (LBR) estimate that is a strict **lower bound** on exploitability (Lisý & Bowling 2017): the best-responder plays a passive check/call rollout valued against a Bayesian belief over the opponent's card, so its value can only under-state true exploitability. It is **never** reported as an upper bound. `tests/test_leduc_lbr.py` validates the guarantee on Leduc where exact NashConv is available (LBR ≤ exact on uniform, intermediate-CFR, and near-equilibrium policies; LBR captures ~83% of exact for the far-from-Nash uniform policy and sits near 0 at equilibrium). This validation is what licenses LBR on a larger game where exact is infeasible.

---

## 7. Exploratory vs. Confirmatory Split

| Analysis | Status | Rationale |
|---|---|---|
| RL vs Myopic edge, primary mirror arm, 500 seeds | **Confirmatory** | Locked above; **executed.** Result in §4.6 (`results/confirmatory.json`) |
| RL vs Myopic edge, luck_adjusted arm | Exploratory | Variance reduction cross-check only |
| RL vs other opponents (Kelly, Aggro) | Exploratory | Opponent pool not frozen |
| ICM edge in bubble / mild ICM contexts | Exploratory | Research question open |
| Tilt phenomenon measurement (frozen pipeline) | Confirmatory (pipeline) | Pipeline frozen; interpretation exploratory |
| Tilt causal claims | Exploratory | Observational data only |
| Leduc exploitability values | Confirmatory (reproducibility) | Solver correctness check |
| Block-B ablations (bust rate, action grid, self-play) | Exploratory | Ablation design not finalized |
| Pool / round-robin leaderboard | Exploratory | Opponent roster not frozen |

---

## 8. Reporting Commitment

Results are reported in full regardless of direction. Null or marginal results (CI bracketing zero) are reported as informative constraints on the effect size, not as failures. The CI width itself is a reportable outcome: a tight CI that brackets zero rules out large edges and is scientifically useful.

Any deviation from the confirmatory plan after registration is documented explicitly as exploratory and does not replace the pre-registered primary outcome. The pre-registered primary outcome is reported whether or not additional exploratory analyses are run.

---

## 9. File Manifest (Ground Truth)

| Role | Path |
|---|---|
| Evaluation harness | `src/evaluation.py` |
| Confirmatory run script (§4.3/§4.6) | `scripts/measure_confirmatory.py` |
| Committed confirmatory result | `results/confirmatory.json` |
| Committed RL checkpoint | `models/confirmatory_rl.pt` |
| Paired-seed generator | `src/tournament.py` (`_pair_seed`) |
| Statistics (bootstrap CI, t-test, sign test) | `src/stats.py` |
| Real-data tilt analysis | `src/real_data_tilt.py` |
| Tilt measurement script | `scripts/measure_tilt_realdata.py` |
| Variance-reduction measurement script | `scripts/measure_variance_reduction.py` |
| Exploitability measurement script | `scripts/measure_exploitability.py` |
| Committed variance-reduction result | `results/variance_reduction.json` |
| Committed tilt result | `results/tilt_realdata.json` |
| Committed exploitability result | `results/exploitability.json` |
| Confirmatory test suite (chip conservation) | `tests/test_phase0.py`, `tests/test_phase2.py`, `tests/test_tournament.py`, `tests/test_icm_eval.py`, `tests/test_phase_c_deepen.py` |
| Confirmatory test suite (evaluation) | `tests/test_evaluation.py` |
| Confirmatory test suite (Leduc) | `tests/test_phase_d_leduc.py` |
| Citation anchor for variance-reduction literature | `REFERENCES.md §2` |
| Multi-seed robustness run script (§10) | `scripts/measure_seed_sweep.py` |
| Committed multi-seed sweep result (§10) | `results/seed_sweep.json` |
| Neural NFSP method (§11) | `src/leduc_neural_nfsp.py` |
| Neural NFSP run script (§11) | `scripts/measure_neural_nfsp.py` |
| Committed neural NFSP result (§11) | `results/neural_nfsp.json` |
| LBR lower bound (§6) | `src/leduc_lbr.py`, `src/big_leduc_lbr.py` |
| Parameterised R-rank game (§12) | `src/big_leduc.py`, `src/big_leduc_nfsp.py` |
| Scaling run script (§12) | `scripts/measure_scale.py` |
| Committed scaling result (§12) | `results/scale_experiment.json` |

---

## 10. Multi-Seed Training Robustness (v2 Phase 0)

This section closes the second open limitation flagged in §0: the confirmatory
(§4.6) fixes a single training seed, `torch_seed=0`, and the sweep that would show
this seed was not chosen after seeing a favorable result was not committed. It is
registered here and **frozen before the run**.

### 10.0 Timing (a stronger registration than §4)

Unlike the §4 confirmatory, which is concurrent (script and outcome committed
together, §0), this sweep is committed in **two separate commits with a
git-provable gap**: (1) this frozen §10 protocol plus `scripts/measure_seed_sweep.py`
with **no results**, then (2) `results/seed_sweep.json` and the §10.5 outcome after
the run. An external reader can verify from git history that the protocol and the
verdict rule (§10.3) were fixed before any sweep result existed. This removes the
seed-selection-independence concern for the sweep itself; it does not retroactively
add an execution trace to the original §4 confirmatory, which remains concurrent.

### 10.1 What is held fixed, what varies

**Only `torch_seed` varies.** The training recipe is imported verbatim from
`scripts.measure_confirmatory.train_pilot_agent` (`torch.manual_seed(torch_seed)`,
`SelfPlayTrainer(seed=1, opponent_mode="fixed", mc_sims=100, hidden=64,
epsilon_start=1.0, epsilon_end=0.05)`, `train(1500, batch_size=64, refresh_every=10,
hands_per_refresh=12)`). The evaluation call, the eval seed block (`0..499`), the
`n_hands=100`, `mirror=True`, `luck_adjusted=False`, and the Myopic baseline are all
byte-identical to §4.3. Because the eval block is constant across all arms, the
fixed seed-range overlap noted in §4.6 shifts every arm by the same amount and
therefore **cannot explain cross-seed variation** — which is exactly the quantity
of interest here.

### 10.2 Frozen sweep

```python
# Frozen before the run. Do not change after registration.
for torch_seed in range(20):                       # torch_seed in 0..19
    qnet = train_pilot_agent(steps=1500, mc_sims=100, hidden=64,
                             torch_seed=torch_seed)  # confirmatory recipe, verbatim
    primary = evaluate_matchup(factory_rl, factory_myopic, "RL", "Myopic",
                               seeds=list(range(500)), n_hands=100,
                               mirror=True, luck_adjusted=False)   # confirmatory call
    # raw single-orientation calibration arm also run (mirror=False), as in §4
```

`N_train_seeds = 20` is fixed here before execution and is not adjusted after
inspecting any arm's outcome. The seed-0 arm must reproduce the committed
confirmatory edge (+256, `results/confirmatory.json`) exactly; that reproduction is
reported as a correctness gate that the sweep harness is identical to the §4
confirmatory harness.

### 10.3 Primary outcome and pre-committed verdict rule

**Primary outcome:** the per-seed mirror-arm mean edge (chips/match) reported as a
**distribution** across the 20 training seeds — its mean, median, SD, min, max, and
the **across-seed 95% bootstrap CI of the mean per-seed edge** (the uncertainty in
the expected edge of a randomly-initialised training run; distinct from any single
seed's eval CI, which is over eval seeds). Also reported: the count of seeds whose
own eval CI individually resolves positive / negative / null, and **seed 0's
percentile rank** within the 20 edges, which discloses whether the published seed
was favorably located.

**Pre-committed verdict rule:** the +256 headline is **robust to training-seed
choice iff** (a) the median per-seed edge is positive **and** (b) the across-seed
95% bootstrap CI of the mean per-seed edge excludes zero on the positive side;
otherwise the edge is **seed-dependent**. Either way the full distribution is
reported per §8. A seed-dependent result would qualify the public headline and is an
honest finding, not a failure.

### 10.4 Reporting commitment

The full 20-seed distribution is committed to `results/seed_sweep.json` and reported
in §10.5 regardless of direction, per §8. If the edge is seed-dependent, or if seed
0 sits in the top decile of the distribution (i.e. the published number was a
favorable draw), that is stated plainly in §10.5 and propagated to the public
narrative (THESIS / README).

### 10.5 Execution status and outcome

**Status: EXECUTED.** The frozen protocol (§10.2) was run once over `torch_seed`
`0..19` and committed to `results/seed_sweep.json`. The result is reported in full
per §8.

**Correctness gate passed.** The seed-0 arm reproduces the committed confirmatory
edge **+256 chips/match exactly** (`seed0_reproduces_committed: true`), confirming
the sweep harness is byte-identical to the §4 confirmatory harness.

**Outcome — the edge is robust to training-seed choice (verdict: robust).** Over
the 20 seeds the per-seed mirror-arm edge has **mean +351, median +300, SD 264,
range [+24, +984]** chips/match. The across-seed 95% bootstrap CI of the mean
per-seed edge is **[+244.4, +468.0]** (exact stored values; the narrative rounds to
[+244, +468]), which **excludes zero on the positive side**;
combined with a positive median this meets the §10.3 robustness rule. **16 of 20
seeds individually resolve a positive edge** (own eval CI excludes zero), **0
resolve negative**, and **4 are null** (seeds 8, 9, 11, 19; their means are +64,
+72, +24, +28 — directionally positive but not individually resolved at 500 eval
seeds). No seed produced a negative edge.

**The published seed 0 was not a favorable draw — if anything it was conservative.**
Seed 0's +256 sits at the **35th percentile** of the 20-seed distribution, *below*
the median +300 and well below the mean +351. The headline therefore understates,
rather than overstates, the typical edge of a randomly-initialised training run.
The single-seed-selection concern flagged in §0 is resolved in the direction
opposite to the worry: there is no evidence the original seed was chosen for a
favorable result.

**What this does and does not change.** It hardens the §4.6 confirmatory claim: the
+256 edge over the *Myopic* baseline is representative, not a lucky seed. It does
**not** touch the honest negatives reported elsewhere — the edge is still measured
only against the myopic baseline, and the RL agent still loses head-to-head to the
zero-parameter Kelly bot (THESIS §3, `results/pool.json`). Robustness to training
seed is not strength against the field.

**Figure:** `figures/seed_sweep.png` (each seed's edge with its eval CI, seed 0
highlighted, the across-seed CI band, and zero).

---

## 11. Neural Equilibrium Method on Leduc (v2 Phase 2)

The first NEURAL equilibrium method in the repo: neural NFSP (Heinrich & Silver
2016) on Leduc, scored by the SAME exact NashConv metric
(`leduc_eval.exploitability_of`) as the tabular CFR / NFSP / Q learners (§6), so the
comparison is apples-to-apples. Registered here and frozen before the run, in the
same two-commit, git-provable-gap form as §10.

### 11.1 What is fixed

- **Method:** `src.leduc_neural_nfsp.LeducNeuralNFSP` — the NFSP algorithm with the
  per-info-set Q table and count table replaced by two MLPs over a structured 21-d
  info-set feature (private card, board card, round, the two betting histories).
  It is OFF by default and imported nowhere in the engine; the baseline is
  byte-identical and existing tests are untouched.
- **Game and metric:** the repo's own Leduc (`leduc_cfr`), exact NashConv of the
  average policy Pi. Identical, non-tunable metric to §6.
- **Hyperparameters (a-priori, NOT tuned on the metric):** `hidden=64`, two ReLU
  layers; `eta=0.1`; `eps` annealed `0.06 -> 0.0`; `gamma=1.0`; Adam `lr_rl=0.01`,
  `lr_sl=0.01`; `batch=128`; RL replay 200k, SL reservoir 1M; target-net update
  every 1000 steps. These are standard NFSP defaults chosen *before* any
  exploitability was measured. Selecting them after seeing exploitability would be
  the garden-of-forking-paths §1 warns against; that is deliberately avoided.
- **Seeds and checkpoints:** seeds `0..4` (5 training seeds, applying the Phase 0
  multi-seed lesson); exact exploitability of Pi recorded at episode checkpoints
  `{50000, 100000, 200000}` — the same episode counts as the committed tabular NFSP
  curve (`results/exploitability.json` `nfsp_curve`), so the head-to-head is at
  matched budgets.

### 11.2 Frozen run

```python
# Frozen before the run. Do not change after registration.
for seed in range(5):
    m = LeducNeuralNFSP(seed=seed)                 # a-priori defaults above
    for ck in (50_000, 100_000, 200_000):
        m.train(ck - prev)
        record exploitability_of(m.average_strategy_table())   # exact NashConv
```

### 11.3 Primary outcome and pre-committed reading

**Primary outcome:** the exact exploitability of neural NFSP's average policy at
each checkpoint, as a distribution across the 5 seeds (mean and min/max), against
the tabular NFSP baseline at the same episode counts.

**Pre-committed reading of the roadmap gate** ("the neural method must beat the
tabular baseline on the same metric"): the gate is read as **sample efficiency at
matched budget** — neural NFSP beats tabular iff its across-seed mean exploitability
is below the tabular value at the same episode count, for a **majority of the three
checkpoints**. It is stated in advance that **asymptotic** dominance on Leduc is
**not** expected and is **not** the claim: Leduc is small enough to tabulate
exactly, so neural function approximation should plateau above the tabular asymptote
and far above the CFR Nash floor (~0.009). The neural method's value is scaling to
games too large to tabulate; Leduc is the exact-measurable correctness check before
scaling. The full curve is reported regardless of which way the gate falls (§8).

### 11.4 Execution status and outcome

**Status: EXECUTED.** The frozen protocol was run once over seeds `0..4` at
checkpoints `{50k, 100k, 200k}` and committed to `results/neural_nfsp.json`.
Reported in full per §8.

**Neural NFSP converges on the exact metric.** Its average-policy exploitability
falls from the uniform **4.75** to an across-seed mean **1.46** at 200k episodes
(per-seed finals 1.68/1.38/1.44/1.39/1.42 — note the real seed-to-seed variance,
the Phase 0 lesson again). So the method is **validated**: a neural function
approximator, scored by the identical exact NashConv metric as the tabular
learners, does learn a converging Leduc average policy.

**But the matched-budget superiority gate is NOT met.** On the fair 5-seed mean,
neural NFSP beats tabular NFSP at only **1 of 3** checkpoints:

| episodes | neural NFSP (mean [min,max]) | tabular NFSP | neural beats? |
|---|---|---|---|
| 50,000 | 1.94 [1.66, 2.23] | 2.40 | yes |
| 100,000 | 1.86 [1.61, 2.37] | 1.80 | no |
| 200,000 | 1.46 [1.38, 1.68] | 1.34 | no |

By the §11.3 majority rule (beat tabular at ≥2 of 3 checkpoints) the gate **fails**.
Neural NFSP is **competitive** with tabular (within ~0.06 at 100k, ~0.12 at 200k)
and wins at the smallest budget, but tabular edges ahead from 100k on. Both remain far above the
CFR Nash floor (**0.009**).

**Honesty note (the Phase 0 lesson, live).** An earlier single-seed look (seed 0
alone) suggested neural beat tabular at 100k; that impression **did not survive the
5-seed mean**. Reporting the multi-seed distribution, not the favorable single seed,
is exactly the discipline §10 established — and here it converts an apparent win into
an honest null.

**What this means.** This is the expected result on a game small enough to tabulate
exactly: neural function approximation has **no advantage on Leduc**, because the
tabular learners already represent every info-set exactly. The neural method is
validated as correct and convergent, but its *value* — generalising across
info-sets in games too large to tabulate — cannot be demonstrated on Leduc. That is
the motivation for the scale-up step (a larger game where tabular CFR is infeasible
and exploitability is measured by a validated LBR lower bound), where "beats the
tabular baseline" becomes a meaningful and reachable claim.

**Figure:** `figures/neural_nfsp.png` (neural across-seed mean ± min/max vs tabular
NFSP vs the CFR Nash floor, on the exact metric).

---

## 12. Scaling: Does Neural NFSP Help Where Tabular CFR Cannot Converge? (v2 Phase 2, Step 2d)

The §11 honest null (neural NFSP has no edge on the tiny exactly-tabulatable 3-rank
Leduc) motivates the real question: at a scale where tabular CFR cannot practically
**converge**, does a neural method reach a less-exploitable strategy in comparable
compute? Registered here and frozen before the run (two-commit git-provable gap, as
§10/§11). This is posed as an OPEN QUESTION, not a predicted neural win.

### 12.1 What is fixed

- **Game:** `big_leduc` at **R = 20** ranks (40 cards; ~12,120 info-sets; 59,280
  deals per full CFR iteration). Validated isomorphic to standard Leduc at R=3
  (`test_big_leduc.py`).
- **Tabular-convergence infeasibility:** measured from the CFR cost curve
  (`measure_scale.py` step 1). At R=20 a full CFR iteration costs ~6.6 s, so the
  ~10⁴ iterations the 3-rank CFR needed to reach ~0.009 exploitability would take
  **~18 hours**. Tabular CFR convergence is therefore infeasible at this scale; this
  conclusion is independent of the head-to-head below.
- **Matched-wall-clock head-to-head (pre-committed counts, ~3–4 min each on the
  measurement machine; actual wall-clock recorded):** tabular CFR for **30
  iterations** vs neural NFSP for **200,000 episodes** over **3 seeds** (a-priori
  config from §11). Both average policies scored by the **exact NashConv** metric
  (a one-time best response, still feasible at R=20 even though convergence is not)
  and cross-checked by the validated LBR lower bound (§6, `big_leduc_lbr`).

### 12.2 Primary outcome and pre-committed reading

**Primary outcome:** the exact exploitability of (a) neural NFSP's average policy
(across-seed mean and min/max over 3 seeds) and (b) truncated tabular CFR, at the
matched budget; plus the uniform-random ceiling for reference.

**Pre-committed reading — honest and falsifiable both ways.** We report which method
reaches lower exact exploitability at the matched budget, in full, regardless of
direction (§8). The honest a-priori expectation, stated before the run, is that
**tabular CFR will likely win**: CFR converges very fast per iteration (a handful of
iterations already drives exploitability down), whereas neural NFSP needs many
episodes and plateaus (§11). If so, the finding is that **neural NFSP has no
measurable advantage at any scale where tabular CFR is feasible per-iteration** —
its structural advantage is confined to scales where tabular CFR cannot complete
even a few iterations (the cost curve extrapolates this to roughly R≈60, where one
CFR iteration alone costs as much as the entire neural budget), a regime in which
exact exploitability is also infeasible and only a lower bound (LBR) could be
reported. That regime is identified here from the cost curve but not run, and is
flagged as the honest frontier (a scalable sampled-LBR is required, and a lower
bound cannot certify low exploitability — only demonstrate exploitation). A neural
win at R=20 would be the surprising, reportable positive.

### 12.3 Execution status and outcome

**Status: PENDING.** The protocol above is frozen and committed before the run; the
outcome (the R=20 head-to-head, the cost curve, and the convergence extrapolation) is
filled in here in the second commit, alongside `results/scale_experiment.json`,
reported in full regardless of direction.
