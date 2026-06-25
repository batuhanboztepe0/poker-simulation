# Poker as a Decision-Science Sandbox

**A poker lab for the skills a trading desk screens for: finding edge under uncertainty, telling a real edge from noise, modelling adversarial counterparties, and sizing risk.** **Built on a seeded Hold'em engine, self-play RL, and an HMM opponent model, and evaluated like a quant backtest: paired seeds, bootstrap CIs, variance reduction.**

![Are the edges real? Every headline edge with its 95% bootstrap CI.](figures/exec_summary.png)

> *The pre-registered confirmatory run resolves the headline edge at +256 chips/match (500 mirrored seeds, CI [+144, +364], binomial p≈7×10⁻⁶). Each headline edge is shown here as a point with its 95% bootstrap CI. Two of four straddle zero, a calibrated result reported in full.*

## What this demonstrates

Prop desks and trading firms screen for a specific way of thinking: expected value under
uncertainty, reading counterparties, and pricing risk. Some (e.g. SIG) literally use poker
to train it. This repo turns that thinking into something measurable:

- **Telling a real edge from noise.** The headline RL-vs-baseline edge resolves at **+256 chips/match**
  under a pre-registered protocol (CI [+144, +364], p≈7×10⁻⁶). Every edge is shown with its 95%
  bootstrap CI, and two of four straddle zero. Showing that, and *not* over-sizing a marginal edge,
  is the trader-maturity signal. (For scale: even an 80,000-hand human-AI match with a margin
  "huge" by professional standards sat at the edge of significance without variance reduction,
  Claudico 2015.)
- **Measuring convergence to Nash.** On Leduc Hold'em with exact NashConv, the CFR time-average
  falls from **0.695 to 0.009** toward Nash, while the greedy last-iterate stays exploitable around
  2.2 and never converges. An independent Q-learner oscillates around 3.40 (range [1.70, 5.53]).
  This is the exact, verifiable reason DQN self-play does not reach equilibrium.
- **Reading exploitable counterparties.** On 777k real human hands, players loosen and turn more
  aggressive after a big loss, the poker analog of adverse selection. A within-player matched
  control (the hand after a loss vs the hand after an *equal-size* win, same player) isolates a
  clean within-player post-loss risk-taking asymmetry (+3.6pp aggression, +2.9pp VPIP; shuffled-label placebo ~0).
- **Respecting principled risk-sizing.** An analytic Kelly bankroll-sizer beats the learned RL
  agent head-to-head, reported plainly, because a learned policy that loses to Kelly is worth
  knowing.

The reinforcement-learning, opponent-modelling, and game-theory machinery underneath (self-play
DQN, HMM tilt detection, exact Leduc-equilibrium analysis) doubles as evidence the ML stack was
built end-to-end, not bolted on.

## What this is, and what it is not

**This is:**
- A seeded, chip-conserving Hold'em engine (2–9 players, side pots) + Monte-Carlo equity.
- A self-play DQN over equity + opponent-belief features, with risk-neutral / Kelly / ICM reward variants and an HMM "tilt" opponent model.
- A quant-style evaluation layer: paired per-seed scenarios, paired t-tests, **bootstrap 95% CIs**, and opt-in duplicate/mirror matching + an all-in EV control variate (the DIVAT/AIVAT variance-reduction lineage).

**This is not:**
- A validated **tradable** signal. The markets parallel (Kyle 1985 / Glosten-Milgrom 1985 informed-vs-noise traders) is a decision-theory hypothesis, untested on real order flow.
- **State of the art.** DQN self-play is a deliberate baseline. The superhuman poker AIs (DeepStack / Libratus / Pluribus / ReBeL) are CFR-family, 2–3 generations ahead. [`figures/exploitability.png`](figures/exploitability.png) and [GUIDE.md](GUIDE.md#4-the-game-theory-is-principled-and-exact) show exactly why, with the exact convergence numbers. (Earlier solver bugs were caught while baselining NFSP, independently verified, and corrected. The full correction history is in GUIDE and THESIS.)
- A claim of **superhuman** or pro-beating play.

## Results at a glance

| Experiment | Result | 95% CI | Statistically resolved? |
|---|---|---|---|
| RL vs myopic, **[pre-registered confirmatory](PREREGISTRATION.md)** (500 mirrored seeds × 100 hands) | **+256** chips/match | [+144, +364], binomial p≈7×10⁻⁶ | Yes, CI excludes 0 (exploratory pilot was +500 at 200 seeds; still loses H2H to Kelly) |
| ↳ **robustness across 20 training seeds** ([pre-registered §10](PREREGISTRATION.md), frozen before the run) | per-seed edge **mean +351, median +300** | across-seed CI [+244, +468]; 16/20 seeds resolve positive, **0 negative** | Yes; seed 0's +256 is at the 35th percentile (*below* median) — not cherry-picked (edge is vs Myopic baseline; RL still loses H2H to Kelly — see rows below) |
| RL vs opponent pool (16 seeds) | **+209** chips, tops leaderboard | [−31, +450] | No, CI includes 0 (loses H2H to Kelly) |
| RL vs **analytic Kelly**, head-to-head (16 seeds) | **5–11** (Kelly wins) | p=0.21, within noise at n=16 | RL **loses** to a 0-parameter closed-form benchmark, reported, not buried |
| Leduc exploitability (exact NashConv) | CFR avg 0.695 → **0.009**; CFR last-iterate ~**2.2**; independent Q-learner oscillates ~**3.40** (range [1.70, 5.53]) | n/a | Exact: averaging → Nash; greedy (DQN-family regime) never converges |
| **Neural NFSP** on Leduc (v2 Phase 2, [pre-registered §11](PREREGISTRATION.md), exact NashConv, 5 seeds) | converges 4.75 → **1.46** (200k eps); beats tabular only at 50k | n/a (exact) | **Honest null on Leduc**: does NOT beat tabular here (small enough to tabulate); neural's value is *scale* — next step measures it on a bigger game via LBR |
| Post-loss tilt, real humans (873 players, 777k hand-rows) | VPIP **+2.8pp**, aggression **+1.6pp** | both exclude 0; placebo ~0 | Yes, real but small |
| Post-loss risk-taking asymmetry (matched: loss vs *equal win*, same player) | aggression **+3.6pp**, VPIP **+2.9pp** | both exclude 0; Cohen d=0.25/0.14; placebo ~0 (n=685) | Yes, clean within-player asymmetry |
| ICM/Kelly vs chip reward (mild ladder, 6 seeds) | **−146** chips | [−249, −51], excludes 0 (n=6) | Directionally negative, n=6, suggestive not robust |

*Every number traces to committed data under [`results/`](results/) and a figure under
[`figures/`](figures/). The real-data hands feed the **opponent model only**, never the policy.*

## Start here

30 seconds: the summary figure and results table above. 5 minutes: GUIDE.md. Full narrative: THESIS.md. Reproduce: notebooks/.

- **[GUIDE.md](GUIDE.md)**: a five-minute, picture-first tour of every result.
- **[THESIS.md](THESIS.md)**: the full narrative, decision-science framing, SOTA placement, and prioritized next steps (full AIVAT, an NFSP learner).
- **[REFERENCES.md](REFERENCES.md)**: every external claim's source, with honest verification flags (refuted claims kept visible on purpose).
- **[notebooks/](notebooks/)**: three executed notebooks that reproduce the headline claims from committed data, no retraining.
- **[figures/README.md](figures/README.md)**: the complete figure index.

## Quickstart

```bash
git clone https://github.com/batuhanboztepe0/poker-simulation.git
cd poker-simulation
python -m pip install -r requirements.txt   # requires Python >= 3.11 (pokerkit, used for the real-data analysis)

python -m pytest tests/ -q                   # 513 tests (509 without torch; RL/torch tests skip)
python -m src.main                           # play a Human vs Bot session
```

Reproducing the measurements/figures (and training the DQN) additionally needs
`pip install "torch>=2.0"`. See [GUIDE.md](GUIDE.md#how-to-reproduce-everything-is-seeded) for
the full seeded reproduction commands.

## Data

The real-data tilt validation uses the **PHH** dataset, *A Dataset of Poker Hand Histories*,
Kim (2024), Zenodo [doi:10.5281/zenodo.13997158](https://doi.org/10.5281/zenodo.13997158),
**[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)**. The NLHE hands originate from a July 2009 HandHQ scrape redistributed by Kim under
CC-BY-4.0. They are used for **opponent-model validation only**, never to train the self-play
policy (human logs in the policy would make the agent exploitable). Raw files are gitignored;
the processed result is committed to [`results/tilt_realdata.json`](results/tilt_realdata.json).

## AI assistance

This project was built with AI assistance (Claude, via Claude Code). The use of AI tools was limited to the following purposes: coding support; multi-agent code review and adversarial auditing of results, figures, and prose; figure generation; literature-search assistance; and writing support. At all stages, the outputs of AI tools were critically reviewed, cross-checked against the committed data and against reliable sources, and revised by me. The responsibility for the final content, analysis, and conclusions rests entirely with me.

## License

Code is released under the [MIT License](LICENSE). The PHH dataset above is separately licensed
CC-BY-4.0 by its author (see Data, above).
