# poker-simulation

**Texas Hold'em as a market-microstructure sandbox: a seeded engine + self-play DQN + HMM tilt-detection, evaluated like a quant backtest (paired seeds, bootstrap CIs, variance reduction) — with marginal-but-measured edges reported honestly.**

![Are the edges real? Every headline edge with its 95% bootstrap CI.](figures/exec_summary.png)

> *Every headline edge as a point with its 95% bootstrap CI. Three of four straddle zero — directionally positive but within per-seed noise. That is the result, measured and reported, not a failure to hide.*

## The honest headline

The agent is **directionally positive but its edges are marginal — mostly within per-seed
noise.** That is not spun: it is measured, with confidence intervals, and reported as the
finding. In a high-variance game with a single-developer DQN, marginal honestly-measured edges
are the *correct* result — and the rigor + honesty is the deliverable. (For scale: even an
80,000-hand human-AI match with a margin "huge" by professional standards was only at the edge
of significance without variance reduction — Claudico 2015.)

## What this is — and is not

**This is:**
- A seeded, chip-conserving Hold'em engine (2–9 players, side pots) + Monte-Carlo equity.
- A self-play DQN over equity + opponent-belief features, with risk-neutral / Kelly / ICM reward variants and an HMM "tilt" opponent model.
- A quant-style evaluation layer: paired per-seed scenarios, paired t-tests, **bootstrap 95% CIs**, and opt-in duplicate/mirror matching + an all-in EV control variate (the DIVAT/AIVAT variance-reduction lineage).

**This is not:**
- A validated **tradable** signal — the markets parallel (Kyle 1985 / Glosten-Milgrom 1985 informed-vs-noise traders) is a decision-theory hypothesis, untested on real order flow.
- **State of the art** — DQN self-play is a deliberate baseline; the superhuman poker AIs (DeepStack / Libratus / Pluribus / ReBeL) are CFR-family, 2–3 generations ahead. [`figures/exploitability.png`](figures/exploitability.png) shows *exactly* why DQN self-play does not reach Nash.
- A claim of **superhuman** or pro-beating play.

## Results at a glance

| Experiment | Result | 95% CI | Statistically resolved? |
|---|---|---|---|
| RL vs myopic baseline (50 seeds × 200 hands) | **+560** chips/match | [+0, +1040], p=0.047 | Borderline — CI lower bound at 0 |
| RL vs opponent pool (16 seeds) | **+209** chips, tops leaderboard | [−31, +450] | No — CI includes 0 (loses H2H to Kelly) |
| Leduc exploitability (exact NashConv) | time-average 0.43 → **0.0014**; greedy last-iterate ~**0.35** | — | Exact: averaging → Nash, greedy does not converge |
| Post-loss tilt, real humans (873 players, 777k hand-rows) | VPIP **+2.8pp**, aggression **+1.6pp** | both exclude 0; placebo ~0 | Yes — real but small |
| ICM/Kelly vs chip reward (mild ladder, 6 seeds) | **−146** chips | excludes 0, p≈0.049 (n=6) | Borderline-negative — no risk-aversion edge |

*Every number traces to committed data under [`results/`](results/) and a figure under
[`figures/`](figures/). The real-data hands feed the **opponent model only**, never the policy.*

## Start here

- **[GUIDE.md](GUIDE.md)** — a five-minute, picture-first tour of every result.
- **[THESIS.md](THESIS.md)** — the full narrative, market-microstructure thesis, SOTA placement, and prioritized next steps (full AIVAT, an NFSP learner).
- **[references.md](references.md)** — every external claim's source, with honest verification flags (refuted claims kept visible on purpose).
- **[notebooks/](notebooks/)** — three executed notebooks that reproduce the headline claims from committed data, no retraining.
- **[figures/README.md](figures/README.md)** — the complete figure index.

## Quickstart

```bash
git clone https://github.com/batuhanboztepe0/poker-simulation.git
cd poker-simulation
python -m pip install -r requirements.txt   # requires Python >= 3.10

python -m pytest tests/ -q                   # 500 tests (RL/torch tests skip without torch)
python -m src.main                           # play a Human vs Bot session
```

Reproducing the measurements/figures (and training the DQN) additionally needs
`pip install "torch>=2.0"`. See [GUIDE.md](GUIDE.md#how-to-reproduce-everything-is-seeded) for
the full seeded reproduction commands.

## Data

The real-data tilt validation uses the **PHH** dataset — *A Dataset of Poker Hand Histories*,
Kim (2024), Zenodo [doi:10.5281/zenodo.13997158](https://doi.org/10.5281/zenodo.13997158),
**CC-BY-4.0**. The NLHE hands originate from a July 2009 HandHQ scrape redistributed by Kim under
CC-BY-4.0. They are used for **opponent-model validation only** — never to train the self-play
policy (human logs in the policy would make the agent exploitable). Raw files are gitignored;
the processed result is committed to [`results/tilt_realdata.json`](results/tilt_realdata.json).

## License

Code is released under the [MIT License](LICENSE). The PHH dataset above is separately licensed
CC-BY-4.0 by its author (see Data, above).
