# References

The literature this project's design, evaluation methodology, and thesis are
grounded in. Every other document ([THESIS.md](THESIS.md),
[figures/README.md](figures/README.md)) cites into this file.

**Verification status.** The entries below were gathered by a multi-source
literature sweep and the key factual claims were put through 3-vote adversarial
verification. Status reflects that check honestly:
- **✓ confirmed** — claim survived verification (vote shown).
- **⚠ qualified / refuted** — a specific claim attributed to the source did *not*
  survive; cite only the narrow, surviving version. Surfacing these is itself
  part of the project's honest-reporting stance.

---

## 1. Poker AI — state of the art

- **Zinkevich, Johanson, Bowling, Piccione (2007).** "Regret Minimization in
  Games with Incomplete Information." *NeurIPS 2007.* — Introduces Counterfactual
  Regret Minimization (CFR), the game-theoretic backbone of every superhuman
  poker AI below.
- **Moravčík et al. (2017).** "DeepStack: Expert-level artificial intelligence in
  heads-up no-limit poker." *Science 356(6337).* — First expert-level HUNL AI;
  deep counterfactual value networks + continual re-solving.
- **Brown & Sandholm (2018).** "Superhuman AI for heads-up no-limit poker:
  Libratus beats top professionals." *Science 359(6374).* — CFR-family,
  heads-up.
- **Brown & Sandholm (2019).** "Superhuman AI for multiplayer poker (Pluribus)."
  *Science 366(6456).* doi:10.1126/science.aay2400 — **✓ confirmed (3-0):** first
  superhuman AI in 6-player NLHE, winning aggregate **> 30 mbb/g** vs
  professionals (48 mbb/g, SE 25, p=0.028 in 5H+1AI; 32 mbb/g, SE 15, p=0.014 in
  1H+5AI). All prior superhuman poker AIs were heads-up only.
  ⚠ A secondary characterization ("Pluribus lacks strong theoretical guarantees /
  is not CFR-grounded") was **refuted 0-3** — do not repeat it.
- **Brown, Bakhtin, Lerer, Gong (2020).** "Combining Deep Reinforcement Learning
  and Search for Imperfect-Information Games (ReBeL)." *NeurIPS 2020.*
  [link](https://proceedings.neurips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html)
  — **✓ confirmed (3-0):** extends the AlphaZero deep-RL-plus-search paradigm to
  imperfect-information games. ⚠ Claims that ReBeL "achieves superhuman HUNL
  performance" and "provably converges to Nash in imperfect-information games"
  were **refuted 0-3** in this source; cite only the paradigm-extension claim.
- **Heinrich & Silver (2016).** "Deep Reinforcement Learning from Self-Play in
  Imperfect-Information Games (NFSP)." *arXiv:1603.01121.* — The theoretically
  grounded middle ground between plain DQN self-play and full CFR; the
  recommended next-step learner for this project.
- **DQN vs CFR in Leduc Hold'em (2025).** *arXiv:2509.04125.* — Restates that
  off-policy TD (DQN) **assumes a stationary target, which adversarial self-play
  violates** (✓ corroborated 2-1, and independently by the MARL sources below),
  and shows DQN underperforming CFR. ⚠ A specific "46–49% vs 50–54% win rate"
  numeric claim from this source was **refuted 0-3**; use it only for the
  qualitative stationarity point.
- **Foerster et al. (2017).** "Stabilising Experience Replay for Deep
  Multi-Agent RL." *arXiv:1702.08887.* — Independent support for the
  non-stationarity problem in multi-agent self-play.
- **Hu & Wellman (2003).** "Nash Q-Learning for General-Sum Stochastic Games."
  *JMLR.* — Theoretical foundation for why Q-learning's guarantees do not
  transfer to a moving-opponent setting.

## 2. Evaluation rigor — variance reduction for high-variance agents

- **Burch, Schmid, Moravčík, Morrill, Bowling (2018).** "AIVAT: A New Variance
  Reduction Technique for Agent Evaluation in Imperfect Information Games."
  *AAAI 2018.* arXiv:1612.06915 /
  [PDF](https://poker.cs.ualberta.ca/publications/aaai18-burch-aivat.pdf)
  — **✓ confirmed (3-0):** provably unbiased (Theorem 1, zero-expectation
  correction terms); reduces variance from *both* chance and decision nodes;
  **85% SD reduction → 44× fewer hands** for equivalent significance in a real
  man-machine match (conservative bound "more than 10×"). The standard rigorous
  poker-evaluation method.
- **Kim & Sandholm (2026).** AIVAT heuristic pathology. *arXiv:2605.14261*
  (preprint; author Sandholm = PI of Libratus/Pluribus). — **✓ confirmed (3-0):**
  AIVAT's unbiasedness is **void if the heuristic value function is tuned after
  seeing evaluation data** (post-hoc tuning can manufacture > 2,000 mbb/h
  artifacts). **Precondition: fix the value function before the evaluation
  data.** Not yet peer-reviewed.
- **Claudico man-machine match (CMU, 2015).** — **✓ confirmed (3-0, via the AIVAT
  paper's motivating example):** 80,000 HUNL hands with a 9 bb/100 margin
  ("huge" by professional standards) were still only **on the edge of
  statistical significance**. The canonical justification for variance reduction:
  naive chip counting is inadequate at realistic sample sizes.
- **Zinkevich, Bowling, Bard, Kan, Billings (2008).** "Imaginary Observations" /
  importance-sampling estimators for poker.
  [slides](http://johanson.ca/publications/poker/2008-icml-imaginary-observations/2008-icml-imaginary-observations-presentation.pdf)
  — Predecessor variance-reduction lineage (terminal-action / chance corrections).
- **White & Bowling — DIVAT.**
  [PDF](https://webdocs.cs.ualberta.ca/~games/poker/publications/divat-icgaj.pdf)
  — The chance-event variance-reduction predecessor to AIVAT. ⚠ Specific numeric
  claims ("5.50× variance reduction", "Θ(n²) games to separate near-equal bots")
  were **refuted 0-3 / 1-2**; cite DIVAT for the lineage, not those figures.

## 3. Poker ↔ markets / market microstructure

- **Kyle (1985).** "Continuous Auctions and Insider Trading." *Econometrica
  53(6).* [PDF](https://people.duke.edu/~qc2/BA532/1985%20EMA%20Kyle.pdf) — The
  rigorous informed-trader model. **This is the defensible anchor for the
  "predictable deviations are exploitable, randomness is not" thesis** (informed
  trader vs. noise trader), not VPIN.
- **Glosten & Milgrom (1985).** "Bid, ask and transaction prices in a specialist
  market with heterogeneously informed traders." *Journal of Financial Economics
  14(1).* — The adverse-selection model; the rigorous analog for "opponent
  modeling ↔ detecting informed/toxic counterparties."
- **Easley, López de Prado, O'Hara (2012).** "Flow Toxicity and Liquidity in a
  High-Frequency World" (VPIN). *Review of Financial Studies.* SSRN:1695596 —
  **⚠ qualified.** Specific claims that VPIN is parameter-free and an empirically
  validated predictor of toxicity-driven volatility were **refuted 0-3**. Do not
  cite VPIN as a validated direct analog in a portfolio context; the
  informed/uninformed *concept* is better grounded in Kyle / Glosten-Milgrom.
- **Bayesian toxic-flow prediction (PULSE).** SSRN:4265814 — **⚠ refuted (0-3 /
  1-2):** claims about real-time per-trade toxicity prediction did not survive.
- **MacLean, Thorp, Ziemba (eds.).** *The Kelly Capital Growth Investment
  Criterion.* / [CAIA summary](https://www.caia.org/sites/default/files/AIAR_Q3_2016_05_KellyCapital.pdf)
  — Kelly / log-bankroll growth, the shared objective across poker bankroll
  management and capital growth.
- **CFA Institute (2018).** "The Kelly Criterion: You Don't Know the Half of It."
  [link](https://rpc.cfainstitute.org/blogs/enterprising-investor/2018/the-kelly-criterion-you-dont-know-the-half-of-it)
  — Practitioner caveats on full-Kelly's variance (motivates fractional Kelly /
  the project's risk-aversion experiments).

## 4. Quant culture — poker as decision-science pedagogy

- **Susquehanna International Group (SIG).** "Game Theory & Decision Science."
  [sig.com](https://sig.com/who-we-are/game-theory-decision-science/) —
  **✓ confirmed (3-0):** SIG uses poker to teach EV reasoning and risk pricing;
  "our traders go through similar thought processes while evaluating the expected
  value of a given trade." The strongest *institutional* (non-metaphorical)
  version of the poker↔trading thesis.
- **Banerji, G. (WSJ, Sept 2024).** SIG's mandatory ~100-hour poker requirement
  in trader training. — **✓ confirmed (3-0):** independent corroboration of the
  SIG program (co-founder Jeff Yass involved).
- **Chen, B. & Ankenman, J.** *The Mathematics of Poker* (2006); both employed in
  quant roles at SIG. [practitioner profile](https://www.benzinga.com/general/24/03/37852287/from-poker-to-prop-desks-how-bill-chen-leveraged-his-poker-skills-into-a-quant-trading-career)
  — **✓ confirmed (3-0):** a documented poker→quant career path.

## 5. Honest-negative / null results in research & hiring

- **Karl et al. (2024).** Null/negative results in ML. *PMLR v235.*
  [link](https://proceedings.mlr.press/v235/karl24a.html)
- **"Negative results" in ML (2024).** *arXiv:2406.03980.*
- **Reproducibility & reporting (2020).** *arXiv:2011.02832.*
  — Together: rigorous null results signal intellectual honesty and statistical
  maturity; the project's honest-negative framing is positioned as a strength,
  not a liability (see [THESIS.md](THESIS.md) §Positioning). *Caveat: that quant
  firms specifically reward null results is supported more by practitioner
  signal than by documented firm policy — frame as "demonstrates rigor," not
  "firms prefer nulls."*

---

*Compiled from a 26-source sweep (94 candidate claims, 25 adversarially
verified: 12 confirmed, 13 refuted). The refuted items are kept visible above on
purpose — knowing what does **not** hold up is part of the result.*
