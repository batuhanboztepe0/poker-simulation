# Poker as a Decision-Science Sandbox

A poker lab for the skills a trading desk screens for: finding edge under uncertainty, telling a real edge from noise, modelling adversarial counterparties, and sizing risk. It is built on a seeded Hold'em engine, self-play RL, and an HMM opponent model, and it is evaluated like a quant backtest: paired seeds, bootstrap CIs, and variance reduction.

![Are the edges real? Every headline edge with its 95% bootstrap CI.](figures/exec_summary.png)

> *The pre-registered confirmatory run resolves the headline edge at +256 chips/match (500 mirrored seeds, CI [+144, +364], binomial p≈7×10⁻⁶). Each edge is a point with its 95% bootstrap CI. Two of four straddle zero, a calibrated result reported in full. A fifth pre-registered result, the §13 tilt-exploitation test, resolved negative and is in the table below.*

## Reading these results

The point of this repository is not a single headline number. It is the discipline a quant uses to decide whether an edge is real. Read the scorecard that way:

- **A resolved edge over a weak baseline.** The RL agent beats a myopic baseline by +256 chips/match under a pre-registered protocol, robust across 20 training seeds. It is modest: the same agent loses head-to-head to a zero-parameter Kelly bot.
- **A reliable positive capability, measured exactly.** On Leduc, a Restricted Nash Response earns a strictly positive, exactly-measured EV gain over Nash against every exploitable opponent (up to +2.16 chips per deal versus a maniac), at a quantified exploitability cost (§14). Zero sampling variance, and it reproduces the validated RNR result on this game. When the opponent is no longer handed over but estimated from observed hands, a conservative response still beats Nash for every opponent, while a raw best response to a sparsely-observed opponent can lose (§15, detect-then-exploit).
- **Honest negatives and nulls, reported in full.** Neural NFSP does not beat tabular at scale (§12). A first, heads-up tilt-exploitation knob made things worse, not better (§13), because bust-match variance swamped the edge and it used the wrong counter-archetype. §14 is the corrected, exact answer.
- **The discipline is the deliverable.** Every confirmatory claim is pre-registered with a git-provable freeze-then-result gap, variance-reduced, and reported whichever way it fell. §13 is the proof the discipline works on a negative (the protocol returned the opposite of the hypothesis); §14 and §15 are the proof it works on a positive, including §15, where a pre-registered validity gate failed and the failure is reported as the finding.

## What this demonstrates

Prop desks and trading firms screen for a specific way of thinking: expected value under uncertainty, reading counterparties, and pricing risk. Some firms (for example SIG) use poker to train it. This repository turns that thinking into something measurable.

- **Respecting a principled benchmark over a learned model.** A trained DQN beats a myopic baseline by +256 chips/match (pre-registered, CI [+144, +364], p≈7×10⁻⁶), but a zero-parameter analytic Kelly bankroll-sizer beats that same DQN head-to-head (5-11). A learned policy that loses to a closed-form benchmark is worth knowing, so it is reported first.
- **Telling a real edge from noise.** Every edge is shown with its 95% bootstrap CI, and two of four straddle zero. Sizing a marginal edge correctly, and not over-sizing it, is the trader-maturity signal. For scale, even an 80,000-hand human-AI match with a margin that professionals called large sat at the edge of significance without variance reduction (Claudico, 2015).
- **Measuring convergence to Nash.** On Leduc Hold'em with exact NashConv, the CFR time-average falls from 0.695 to 0.009 toward Nash, while the greedy last-iterate stays exploitable around 2.2 and never converges. An independent Q-learner oscillates around 3.40 (range [1.70, 5.53]). This is the exact, verifiable reason DQN self-play does not reach equilibrium.
- **Reading exploitable counterparties.** On 777k real human hands, players loosen and turn more aggressive after a big loss, the poker analog of adverse selection. A within-player matched control (the hand after a loss against the hand after an equal-size win, same player) isolates a clean post-loss risk-taking asymmetry (+3.6pp aggression, +2.9pp VPIP; shuffled-label placebo near 0).

The reinforcement-learning, opponent-modelling, and game-theory machinery underneath (self-play DQN, HMM tilt detection, exact Leduc-equilibrium analysis) doubles as evidence the ML stack was built end-to-end, not bolted on.

## From poker to a trading desk

Each technique here maps to a standard quant-research practice. That mapping is the transferable skill.

| In this repo | The quant-research practice |
|---|---|
| Paired and mirror (duplicate-deck) seeds | Common-random-numbers / duplicate-scenario backtesting |
| All-in EV control variate (AIVAT family) | Control variates for variance reduction in a backtest |
| Bootstrap CIs plus an exact sign test | Distribution-free significance for heavy-tailed PnL |
| Pre-registration with a freeze-then-result gap | Committing the test before the data, against p-hacking and overfitting |
| Two-state {normal, tilted} HMM tilt detector | Regime-switching (Hamilton) models of a counterparty or market |
| Exploit-vs-exploitability tradeoff (§13) | A strategy tuned to one counterparty is fragile against another |

## What this is, and what it is not

This is:
- A seeded, chip-conserving Hold'em engine (2-9 players, side pots) plus Monte-Carlo equity.
- A self-play DQN over equity and opponent-belief features, with risk-neutral, Kelly, and ICM reward variants and an HMM tilt opponent model.
- A quant-style evaluation layer: paired per-seed scenarios, paired t-tests, bootstrap 95% CIs, and opt-in duplicate/mirror matching plus an all-in EV control variate (the DIVAT/AIVAT variance-reduction lineage).

This is not:
- A validated tradable signal. The markets parallel (Kyle 1985 and Glosten-Milgrom 1985, informed against noise traders) is a decision-theory hypothesis, untested on real order flow.
- State of the art. DQN self-play is a deliberate baseline. The superhuman poker AIs (DeepStack, Libratus, Pluribus, ReBeL) are CFR-family, two to three generations ahead. [`figures/exploitability.png`](figures/exploitability.png) and [GUIDE.md](GUIDE.md#4-the-game-theory-is-principled-and-exact) show exactly why, with the exact convergence numbers. Earlier solver bugs were caught while baselining NFSP, independently verified, and corrected. The full correction history is in GUIDE and THESIS.
- A claim of superhuman or pro-beating play.

## Results at a glance

| Experiment | Result | 95% CI | Statistically resolved? |
|---|---|---|---|
| RL vs myopic, [pre-registered confirmatory](PREREGISTRATION.md) (500 mirrored seeds × 100 hands) | +256 chips/match | [+144, +364], binomial p≈7×10⁻⁶ | Yes, CI excludes 0 (exploratory pilot was +500 at 200 seeds; still loses H2H to Kelly) |
| ↳ robustness across 20 training seeds ([pre-registered §10](PREREGISTRATION.md), frozen before the run) | per-seed edge mean +351, median +300 | across-seed CI [+244, +468]; 16/20 seeds resolve positive, 0 negative | Yes; seed 0's +256 is at the 35th percentile, below median, so not cherry-picked (edge is vs the Myopic baseline; RL still loses H2H to Kelly) |
| RL vs opponent pool (16 seeds) | +209 chips, tops leaderboard | [−31, +450] | No, CI includes 0 (loses H2H to Kelly) |
| RL vs analytic Kelly, head-to-head (16 seeds) | 5-11 (Kelly wins) | p=0.21, within noise at n=16 | RL loses to a 0-parameter closed-form benchmark, reported, not buried |
| Leduc exploitability (exact NashConv) | CFR avg 0.695 → 0.009; CFR last-iterate ~2.2; independent Q-learner oscillates ~3.40 (range [1.70, 5.53]) | n/a | Exact: averaging converges to Nash; greedy (the DQN-family regime) never converges |
| Neural NFSP on Leduc (v2 Phase 2, [pre-registered §11](PREREGISTRATION.md), exact NashConv, 5 seeds) | converges 4.75 → 1.59 (200k); beats tabular at 2/3 checkpoints (50k, 100k) | n/a (exact) | Qualified pass (corrected: §11.4 erratum fixes an epsilon-schedule bug; was a 1/3 null). It meets the pre-committed 2/3 sample-efficiency gate, but tabular still wins asymptotically (200k) and the run is noisy, so not a strong win |
| Scaling: neural NFSP vs tabular CFR at R=20 (v2 Phase 2, [pre-registered §12](PREREGISTRATION.md), exact NashConv, matched episode budget) | tabular CFR 0.25 (30 iters, 198s) vs neural 1.00 (200k eps, ~472s/seed) | n/a (exact) | Honest null at scale: tabular CFR wins (and on less wall-clock) even where its convergence is infeasible (~35h). The wall-clocks were not matched, neural received ~2.4× more. Neural's edge would only appear at unmeasurable extreme scale (~R≈60); the LBR lower bound is validated at scale |
| Exploitation: tilt-exploit knob ([pre-registered §13](PREREGISTRATION.md), 300 mirrored seeds) | −169 chips/match (paired delta) | [−271, −66], sign-test p≈0.04 | Resolved NEGATIVE. The knob hurts: loosening against an aggressive tilter walks into its value, and the disciplined baseline already wins (+533 vs +196). A powered run overturned a +152 small-sample peek |
| **Exact RNR exploitation on Leduc** ([pre-registered §14](PREREGISTRATION.md), exact EV) | RNR gains **+0.96 / +2.16 / +1.50** chips/deal over Nash vs station / maniac / uniform | n/a (exact, zero variance) | **Reliable positive**: a Restricted Nash Response exploits every suboptimal opponent for an exact EV gain over Nash, at a quantified exploitability cost (the tradeoff). p=0 reproduces Nash, p=1 the exact best response; the corrected, exact answer to §13 |
| **Detect-then-exploit on Leduc** ([pre-registered §15](PREREGISTRATION.md), exact EV vs an estimated opponent) | conservative p=0.5 gains **+0.75 / +1.31 / +1.13 / +0.78** over Nash at 40 observed hands (station / maniac / uniform / loose-passive) | 95% CI excludes 0 (all four) | **Reliable positive, opponent estimated not handed**: play Nash, observe N hands, then RNR on the estimate still beats Nash with a conservative p (the meaningful upgrade over §14). Caveat: a raw best response (p=1) to the maniac loses to Nash, the data-biased-response lesson |
| Post-loss tilt, real humans (873 players, 777k hand-rows) | VPIP +2.8pp, aggression +1.6pp | both exclude 0; placebo near 0 | Yes, real but small |
| Post-loss risk-taking asymmetry (matched: loss vs equal win, same player) | aggression +3.6pp, VPIP +2.9pp | both exclude 0; Cohen d=0.25/0.14; placebo near 0 (n=685) | Yes, a clean within-player asymmetry |
| ICM/Kelly vs chip reward (mild ladder, 6 seeds) | −146 chips | [−249, −51], excludes 0 (n=6) | Directionally negative, n=6, suggestive not robust |

*Every number traces to committed data under [`results/`](results/) and a figure under [`figures/`](figures/). The real-data hands feed the opponent model only, never the policy.*

## Start here

30 seconds: the summary figure and the results table above. 5 minutes: GUIDE.md. Full narrative: THESIS.md. Reproduce: notebooks/.

- **[GUIDE.md](GUIDE.md)**: a five-minute, picture-first tour of every result.
- **[THESIS.md](THESIS.md)**: the full narrative, the decision-science framing, the placement against state of the art, and the prioritized next steps.
- **[REFERENCES.md](REFERENCES.md)**: every external claim's source, with honest verification flags. Refuted claims are kept visible on purpose, with confirmed / refuted / qualified tags.
- **[FUTURE_WORK.md](FUTURE_WORK.md)**: the directions the work does not yet take, ordered by value for effort, with the honest limitations stated up front.
- **[notebooks/](notebooks/)**: a one-page visual overview ([`00_overview.ipynb`](notebooks/00_overview.ipynb)) plus five executed chapters that reproduce the headline claims from committed data, no retraining. They read top to bottom on GitHub.
- **[figures/README.md](figures/README.md)**: the complete figure index.

## Quickstart

```bash
git clone https://github.com/batuhanboztepe0/poker-simulation.git
cd poker-simulation
python -m pip install -r requirements.txt   # requires Python >= 3.11 (pokerkit, used for the real-data analysis)

python -m pytest tests/ -q                   # 537 tests (RL/torch tests skip without torch)
python -m src.main                           # play a Human vs Bot session
```

Reproducing the measurements and figures (and training the DQN) additionally needs `pip install "torch>=2.0"`. See [GUIDE.md](GUIDE.md#how-to-reproduce-everything-is-seeded) for the full seeded reproduction commands.

## Data

The real-data tilt validation uses the PHH dataset, *A Dataset of Poker Hand Histories*, Kim (2024), Zenodo [doi:10.5281/zenodo.13997158](https://doi.org/10.5281/zenodo.13997158), [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). The NLHE hands originate from a July 2009 HandHQ scrape redistributed by Kim under CC-BY-4.0. They are used for opponent-model validation only, never to train the self-play policy (human logs in the policy would make the agent exploitable). Raw files are gitignored; the processed result is committed to [`results/tilt_realdata.json`](results/tilt_realdata.json).

## AI assistance

This project was built with AI assistance (Claude, via Claude Code). The use of AI tools was limited to the following purposes: coding support; multi-agent code review and adversarial auditing of results, figures, and prose; figure generation; literature-search assistance; and writing support. At all stages, the outputs of AI tools were critically reviewed, cross-checked against the committed data and against reliable sources, and revised by me. The responsibility for the final content, analysis, and conclusions rests entirely with me.

## License

Code is released under the [MIT License](LICENSE). The PHH dataset above is separately licensed CC-BY-4.0 by its author (see Data, above).
