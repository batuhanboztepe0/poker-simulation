# Future work

This project measures poker the way a quant measures a strategy: with pre-registration,
exact metrics where they are available, and honest negatives. The exploitation arc
([`PREREGISTRATION.md`](PREREGISTRATION.md) §13 to §15) is the spine. A pre-registered
policy knob resolved negative (§13), the two reasons it failed were fixed to reach an exact
positive (§14), and that positive survived having to estimate the opponent from data (§15).
The arc is complete and stands on its own.

This document is the honest map of what the work does not yet do, and which directions are
worth taking next. It is here because stating a boundary plainly is part of the deliverable.

## What this work does not claim

- The exploitation results are exact on 3-rank Leduc, against opponent classes chosen in
  advance. Transfer to larger games is directional, not measured.
- §15 estimates the opponent with a frequency table under a fixed Dirichlet prior, while the
  hero plays Nash. Other estimators and other observation policies are not tested.
- The opponent detector that runs inside the multiplayer engine (§13) is a two-state tilt
  HMM. The exact-Leduc exploitation of §14 and §15 does not run inside that engine.
- No claim of Nash safety. An exploiter is itself counter-exploitable, by construction. Every
  exploitation result above is measured against a specific opponent, not against a worst case.

## Directions, by value for effort

### 1. Online detect-then-exploit in the full game

The most valuable direction, because it is the only one that adds a capability rather than
more rigor on an existing one.

- **Question.** Does the detect-then-exploit pipeline work in the real multiplayer engine,
  not only on exact Leduc?
- **Approach.** Connect the §13 tilt HMM (online detection) to the §14 and §15 Restricted
  Nash Response (exploit the estimated opponent), and add a pre-fixed AIVAT-style control
  variate to cut the bust-match variance that swamped §13.
- **Risk.** The §13 variance problem returns (per-seed standard deviation near 800 chips),
  the two-state HMM is a coarse opponent model, and the result could be another null.
- **Why it matters.** This is the honest end state of the whole arc: detect and exploit, end
  to end, in the game actually played.

### 2. Opponent-class inference

- **Question.** Instead of estimating a full frequency table, can the bot infer which class
  an opponent belongs to, then exploit the class?
- **Approach.** Bayesian model selection over a small set of classes (calling station,
  maniac, loose-passive, near-Nash), then RNR against the inferred class.
- **Risk.** Class misspecification. A real opponent that fits none of the classes degrades
  the method back toward the §15 frequency estimator, or worse.
- **Why it matters.** Cleaner detection than a raw frequency table, and a direct analogue of
  regime-switching counterparty models.

### 3. Does exploitation survive a bigger game?

- **Question.** Are the §14 and §15 results an artifact of 3-rank Leduc?
- **Approach.** Rerun RNR and detect-then-exploit on the parameterised R-rank game
  ([`src/big_leduc.py`](src/big_leduc.py)). At moderate R the EV stays exact. Beyond the
  point where full enumeration is feasible, only an LBR lower bound applies.
- **Risk.** Past moderate R the exact-EV clarity is lost and the result becomes a lower
  bound, which can demonstrate exploitation but cannot certify it. The clean exact story may
  be better left as the stated limitation than half-answered.
- **Why it matters.** It addresses the most obvious criticism, that the game is a toy.

### 4. §15 robustness

- **Question.** Is the §15 positive knife-edge?
- **Approach.** Sweep the Dirichlet smoothing prior, observe while playing the RNR counter
  (closed loop) rather than Nash (open loop), and add more stochastic opponents.
- **Risk.** Low. The most likely outcome confirms robustness. It could reveal a dependence on
  the prior.
- **Why it matters.** A routine robustness check, useful to preempt a methodological question
  rather than to produce a new finding.

### 5. Training on real hand histories

- **Question.** Can a strategy trained on real human hands reach lower exploitability, instead
  of being hand-built?
- **Approach.** Off-policy fictitious self-play (OFF-FSP, arXiv:2403.00841) on the PHH subset,
  not behaviour cloning. Measure whether exploitability goes down.
- **Risk.** A null is likely. Partial data coverage breaks the Nash convergence guarantee, so
  exploitability may not fall.
- **Why it matters.** The most novel direction, and the closest to a real trading-data
  workflow. A clean null here is still a result under this project's standards.

## The discipline carries over

Any of these follows the same rules as the rest of the repo. Pre-register the protocol and
the verdict in a frozen commit with no results, run once, report whichever way it falls, and
audit before committing. A lower bound is reported only as a lower bound. The point is not to
add wins. It is to keep every claim measured.
