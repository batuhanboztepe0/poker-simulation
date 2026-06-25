# A guided tour of the results

A five-minute, picture-first walkthrough of what this project found. For the
full narrative and positioning see [THESIS.md](THESIS.md); for the literature
see [REFERENCES.md](REFERENCES.md); for the raw figure index see
[figures/README.md](figures/README.md).

---

## TL;DR (60 seconds)

- **What it is:** a seeded Texas Hold'em engine, a self-play reinforcement-learning
  agent (DQN), and an HMM "tilt" opponent model, evaluated like a **quant backtest**
  (paired seeds, t-tests, bootstrap CIs, variance reduction).
- **The thesis:** poker trains the trader's core skill, *finding edge under
  uncertainty*, where predictable deviations are exploitable and pure randomness
  is not. SIG literally puts traders through ~100 hours of poker to teach it
  (REFERENCES.md §4); the market-microstructure framing (Kyle 1985 /
  Glosten-Milgrom 1985) is **motivation**, untested on real order-flow data.
- **The honest headline:** the lead finding is the exact NashConv result on Leduc
  (figure 4). The CFR time-average converges to Nash (0.695 to 0.009) while the greedy
  last-iterate never does (~2.2), confirmed directly by an independent tabular Q-learner.
  This faithfully reproduces known theory (Freund-Schapire 1999; Mertikopoulos et al.
  2018; Bailey-Piliouras 2018; Daskalakis-Panageas ITCS 2019) with exact Leduc numbers
  and a tabular-NFSP fix. It is a clean pedagogical demonstration built end-to-end, not a
  novel theorem. The project also finds a **within-player post-loss risk-taking asymmetry** on
  777k real human hands (figure 7, d=0.25, well-calibrated to prior literature). The RL
  agent's edge over a myopic baseline **resolves** under a pre-registered confirmatory run
  (500 mirrored seeds: +256 chips, CI [+144, +364]) but is modest, and a 0-parameter Kelly
  bot beats it. Every edge is reported with its CI.

> If you read one figure, read **[`figures/exec_summary.png`](figures/exec_summary.png)** below.

---

## The story in figures

### 1. Are the edges real? Start here

![exec summary](figures/exec_summary.png)

**What you're looking at:** every headline edge as a point with its 95% bootstrap
confidence interval. Gray = the interval straddles 0 (the effect is within
per-seed noise); green/red = it excludes 0.

**Takeaway:** the agent beats the myopic baseline (pre-registered confirmatory:
+256 chips, CI [+144, +364], binomial p≈7×10⁻⁶) and tops the opponent pool (+209). Of the four edges
here, **2 have CIs excluding 0** (this baseline edge, robustly; and the ICM
reward, *negative*, but only suggestive at n=6) and **2 stay within per-seed
noise**. One resolved win plus honest nulls: the whole project in one chart.

### 2. The agent does learn

![learning curve](figures/headline.png)

**What you're looking at:** held-out win rate (left) and mean chip diff with a ±1
SEM ribbon (right) over training.

**Takeaway:** the dip-then-climb is real. The agent first collapses into
over-folding, then recovers to beat the baseline. This is the **exploratory
pilot** (final 125/200 matches, +500, 95% CI [+240, +760], excludes 0). The
50-seed curve on the right is a *noisy training monitor*, not separate evidence:
the same 50 seeds read a significantly **negative** 15/50 (−800 chips) mid-training
(step 500) before recovering to 33/50 at the final checkpoint, so 50 seeds is not
enough to call the edge at an arbitrary checkpoint. The edge is established by the
200-seed pilot above and by the separate **pre-registered confirmatory** at 500
mirrored seeds, which pins it down tightly at +256 [+144, +364] (figure 1).

### 3. It generalizes to a whole opponent pool

![pool leaderboard](figures/pool_leaderboard.png)

**What you're looking at:** a belief and opponent-mix generalist (RL) against a pool
of {myopic, tilt, random} plus an analytic Kelly agent.

**Takeaway:** RL **tops the leaderboard** and beats two of three adaptive
opponents head-to-head (13-3 vs myopic, 12-4 vs random; 9-7 vs tilt is within
noise at n=16). It **loses head-to-head to Kelly (5-11)**.
Reported honestly: the win is the leaderboard and adaptive pool, not Kelly.

![personality sweep](figures/pool_sweep.png)

**What you're looking at:** the same RL agent dropped into a round-robin of static
(tightness × aggression) personalities.

**Takeaway:** RL ranks **#6/17** here (it does not top the static sweep), because
a round-robin rewards farming the weakest static cells, which is not what the
agent was trained for. Another honest, contextualized number.

### 4. The game theory is principled and exact

![exploitability](figures/exploitability.png)

**What you're looking at:** exact exploitability (NashConv; 0 = exact Nash
equilibrium) on Leduc Hold'em, log-log, over training.

**Takeaway:** the **time-average strategy converges to Nash** (0.695 → 0.009),
but the **greedy last-iterate stays exploitable (~2.2) and never converges.** An
independent tabular Q-learning self-play (an actual DQN-regime learner) confirms
it directly: its greedy last-iterate oscillates around 3.40 (range [1.70, 5.53]),
never near Nash. This is the *exact, verifiable* reason
DQN self-play does not reach equilibrium and averaging methods (CFR; NFSP at
scale) do.

(Corrected, independently-verified numbers: an earlier sign error in the CFR
round-transition made the average converge to a degenerate all-call strategy, and
a lock-out in the best-response metric underestimated exploitability, even
returning impossible negative values. Both were caught while baselining NFSP,
confirmed by three independent reimplementations agreeing to machine precision plus
a four-way adversarial check, and fixed. The qualitative result, time-average
reaches Nash while the greedy last-iterate does not, was unchanged; only the magnitudes changed.)

### 5. Measured like a quant: variance reduction

![variance reduction](figures/variance_reduction.png)

**What you're looking at:** the *same* edge measured four ways, with its 95% CI.

**Takeaway:** **duplicate/mirror matching cuts the CI to 65% of raw** (same
conclusion from fewer matches); the all-in EV control variate is ~neutral *in this
bust-match format* (match-outcome variance is dominated by bust path-dependence,
not single-hand runout), stated honestly rather than overclaimed.

### 6. The thesis tested on real human hands

![real-data tilt](figures/tilt_realdata.png)

**What you're looking at:** the exploit-predictable-deviations thesis tested on
**777k hand-rows** of 2009 online play (PHH / Kim 2024, CC-BY-4.0), used for the
**opponent model only, never the policy**. Each effect is shown against a
shuffled-label placebo.

**Takeaway:** after a ≥10bb loss, 873 real players play **looser (VPIP +2.8pp)
and more aggressively (+1.6pp)**. Both 95% CIs exclude 0, while the shuffled
placebo collapses to ~0, so it is the loss, not chance. The project's HMM tilt
detector registers a small but resolved P(tilted) shift, and a separate
Baum-Welch regime HMM corroborates it out-of-sample. Honestly small (1–3pp) in
absolute size but statistically resolved (both CIs exclude 0). This is one of the
project's two real-data headline findings, and the behavioral pattern the
decision-science framing casts as the poker analog of adverse selection (the
markets parallel is motivation, untested on real order flow).

### 7. The confound-controlled result: loss versus an equal win (the stronger test)

![post-loss risk-taking asymmetry](figures/tilt_lossvswin.png)

**What you're looking at:** the post-loss-vs-baseline shift could be confounded:
looser players both lose more and play looser. This panel addresses that by comparing each player's hand after
a ≥10bb **loss** to their hand after an **equal-size ≥10bb win**, matching player,
big-pot arousal, and event size. Only the swing *sign* differs.

**Takeaway:** after a loss players are **+3.6pp more aggressive and +2.9pp looser**
than after an equal win (95% CIs exclude 0, Cohen d=0.25/0.14, n=685 matched
players; shuffled-label placebo ~0). The effect is *larger* than the vs-baseline effect
because players also tighten after a win. This is a clean **within-player post-loss
risk-taking asymmetry**, not generic big-pot arousal. Its relation to the behavioral
literature (the Imas 2016 realization effect, Coval-Shumway loss-chasing) is discussed
in THESIS.md. The within-player control removes the selection confound and is the
stronger of the two tilt tests.

### 8. Honest negatives (a feature, not a bug)

![ICM edge](figures/icm_edge_mild.png)

**What you're looking at:** a concave ICM/Kelly reward vs a risk-neutral chip
reward at a multi-prize table, per seed.

**Takeaway:** the ICM "risk-aversion edge" **does not reproduce** at a properly
seeded scale (mean −146, 1/6 seeds positive). A whole sweep of RL-mechanics
levers, including finer action grid, un-truncated bust clip, tilt-bonus decoupling,
snapshot self-play, and warmed fold-equity (the `blockB_*` and `rollout_fe` panels in
[figures/](figures/)), each works mechanically but **none moves the headline.**
Reported as a clean negative-results sweep, which is itself a result.

---

## How to reproduce (everything is seeded)

```bash
# install (requires Python >= 3.10)
python -m pip install -r requirements.txt

# tests (torch-free parts run anywhere; RL/torch tests skip without torch)
OMP_NUM_THREADS=1 python -m pytest tests/ -q          # 513 green (509 without torch)

# regenerate the committed measurement data under results/ (trains the DQN, so
# it needs torch: pip install "torch>=2.0": commented out in requirements.txt)
OMP_NUM_THREADS=1 bash scripts/run_measurements.sh    # Block B + ICM + rollout + headline + pool
OMP_NUM_THREADS=1 python -m scripts.measure_variance_reduction --out results/variance_reduction.json
OMP_NUM_THREADS=1 python -m scripts.measure_exploitability     --out results/exploitability.json
OMP_NUM_THREADS=1 python -m scripts.measure_confirmatory --raw --out results/confirmatory.json  # pre-registered RL-vs-myopic run (PREREGISTRATION.md §4.3)
OMP_NUM_THREADS=1 python -m scripts.measure_seed_sweep --out results/seed_sweep.json  # multi-seed robustness: retrain torch_seed 0..19, edge as a distribution (PREREGISTRATION.md §10)
OMP_NUM_THREADS=1 python -m scripts.measure_neural_nfsp --out results/neural_nfsp.json  # neural NFSP on Leduc, exact exploitability vs the tabular baseline (PREREGISTRATION.md §11)
OMP_NUM_THREADS=1 python -m scripts.measure_scale --out results/scale_experiment.json  # scaling: neural NFSP vs tabular CFR at R=20 where CFR can't converge (PREREGISTRATION.md §12)

# real-data tilt validation (fetches the PHH subset to data/phh/, gitignored)
python -m scripts.fetch_phh --max-files 120            # the 120 PokerStars 25NL files used
OMP_NUM_THREADS=1 python -m scripts.measure_tilt_realdata  # -> results/tilt_realdata.json

# redraw every figure from results/
python -m scripts.make_figures                        # -> figures/*.png (+ .html)
```

---

## What's honest about this (limitations, stated up front)

- The RL agent's edge over the baseline **resolves** at 200 seeds but is
  **modest** (a 0-parameter Kelly bot beats it head-to-head); the pool edge stays
  within per-seed noise and the ICM reward resolves *negative* (CI excludes 0 but
  n=6, suggestive). All measured, not spun.
- **DQN is a deliberate baseline, not state of the art.** CFR-family methods
  (DeepStack/Libratus/Pluribus/ReBeL) are 2–3 generations ahead (figure 4 shows
  exactly why).
- The **markets thesis is asserted, not yet validated on real market data.** The
  rigorous anchor is Kyle / Glosten-Milgrom, not the (refuted) VPIN claims
  (see REFERENCES.md, where refuted claims are kept visible on purpose).
- The agent's training/eval opponents are **synthetic** (real human logs would
  make a learned policy exploitable, so they feed the opponent model ONLY); the
  tilt opponent-model itself is now **validated on 777k real human hand-rows**
  (PHH / Kim 2024, CC-BY-4.0; doi:10.5281/zenodo.13997158):
  post-loss VPIP +2.8pp and aggression +1.6pp, both 95% CIs exclude 0
  (`tilt_realdata.png`, THESIS.md §6). Real but small, like the rest.

---

## Where to go next

- **[THESIS.md](THESIS.md)**: the full narrative, SOTA placement, and the
  prioritized next steps (full AIVAT, NFSP learner).
- **[REFERENCES.md](REFERENCES.md)**: every claim's source, with honest
  verification flags.
- **[figures/README.md](figures/README.md)**: the complete figure index.
