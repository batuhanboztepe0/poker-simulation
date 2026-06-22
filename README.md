# poker-simulation

**A poker lab for the skills a trading desk screens for — finding edge under uncertainty, telling a real edge from noise, modelling adversarial counterparties, and sizing risk. Built on a seeded Hold'em engine + self-play RL + an HMM opponent model, and evaluated like a quant backtest: paired seeds, bootstrap CIs, variance reduction.**

![Are the edges real? Every headline edge with its 95% bootstrap CI.](figures/exec_summary.png)

> *Every headline edge as a point with its 95% bootstrap CI. Two of four straddle zero; the headline RL-vs-baseline edge resolves once measured on 200 paired seeds (exact binomial p=0.0005, CI excludes 0). That is the result, measured and reported, not a failure to hide.*

## What this demonstrates

Prop desks and trading firms screen for a specific way of thinking — expected value under
uncertainty, reading counterparties, and pricing risk — and some (e.g. SIG) literally use poker
to train it. This repo turns that thinking into something measurable:

- **Telling a real edge from noise.** Every claimed edge is shown with its 95% bootstrap CI (the
  figure above): 2 of 4 straddle zero. Surfacing that — and *not* over-sizing a marginal edge —
  is the trader-maturity signal, not a result to hide. (For scale: even an 80,000-hand human-AI
  match with a margin "huge" by professional standards sat at the edge of significance without
  variance reduction — Claudico 2015.)
- **Reading exploitable counterparties.** On 777k real human hands, players measurably loosen and
  turn more aggressive after a big loss — the poker analog of adverse selection — shown with a
  shuffled-label placebo and per-player baselines.
- **Respecting principled risk-sizing.** An analytic Kelly bankroll-sizer beats the learned RL
  agent head-to-head; reported plainly, because a learned policy that loses to Kelly is worth
  knowing.

The reinforcement-learning, opponent-modelling, and game-theory machinery underneath (self-play
DQN, HMM tilt detection, exact Leduc-equilibrium analysis) doubles as evidence the ML stack was
built end-to-end, not bolted on.

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
| RL vs myopic baseline (200 seeds × 200 hands) | **+500** chips/match | [+240, +760], exact binomial p=0.0005 | Yes — CI excludes 0 (still loses H2H to Kelly) |
| RL vs opponent pool (16 seeds) | **+209** chips, tops leaderboard | [−31, +450] | No — CI includes 0 (loses H2H to Kelly) |
| Leduc exploitability (exact NashConv) | time-average 0.433 → **0.0014**; greedy last-iterate ~**0.355** | — | Exact: averaging → Nash, greedy does not converge |
| Post-loss tilt, real humans (873 players, 777k hand-rows) | VPIP **+2.8pp**, aggression **+1.6pp** | both exclude 0; placebo ~0 | Yes — real but small |
| ICM/Kelly vs chip reward (mild ladder, 6 seeds) | **−146** chips | [−249, −51], excludes 0 (n=6) | Directionally negative — n=6, suggestive not robust |

*Every number traces to committed data under [`results/`](results/) and a figure under
[`figures/`](figures/). The real-data hands feed the **opponent model only**, never the policy.*

## Start here

- **[GUIDE.md](GUIDE.md)** — a five-minute, picture-first tour of every result.
- **[THESIS.md](THESIS.md)** — the full narrative, decision-science framing, SOTA placement, and prioritized next steps (full AIVAT, an NFSP learner).
- **[references.md](references.md)** — every external claim's source, with honest verification flags (refuted claims kept visible on purpose).
- **[notebooks/](notebooks/)** — three executed notebooks that reproduce the headline claims from committed data, no retraining.
- **[figures/README.md](figures/README.md)** — the complete figure index.

## Quickstart

```bash
git clone https://github.com/batuhanboztepe0/poker-simulation.git
cd poker-simulation
python -m pip install -r requirements.txt   # requires Python >= 3.10

python -m pytest tests/ -q                   # 504 tests (RL/torch tests skip without torch)
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
