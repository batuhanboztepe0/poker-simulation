# Pre-Registration: Confirmatory Evaluation Plan

**Repository:** poker-simulation  
**Branch at time of registration:** rl-multihand-episodes  
**Registered:** 2026-06-23  
**Purpose:** Lock the analysis plan and acceptance criteria before re-running evaluation, to prevent garden-of-forking-paths inflation of Type-I error (Gelman & Loken 2014, "The Statistical Crisis in Science," *American Scientist* 102(6):460–465).

## 0. Timing and scope

This document IS the pre-registration. It describes the harness as it exists in the repository and freezes the confirmatory analysis. **Nothing here is claimed to have been pre-registered earlier in git history.** The registration is **concurrent**, not sequenced ahead of the run: this document, the run script (`scripts/measure_confirmatory.py`), and the outcome (§4.6, `results/confirmatory.json`) are committed *together*: there is deliberately no git-provable time gap between freezing the protocol and running it. What the registration buys is therefore **not** a temporal-precedence claim but a **pre-committed reporting rule**: the frozen protocol (§4.3) is run once and its result reported in full regardless of direction (§8). The exploratory pilot (`results/headline_history.json`) informed the *design*; the confirmatory run tests the frozen protocol and is reported here whatever it returned (it returned a smaller, still-resolved edge).

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
