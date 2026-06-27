# Notebooks: the story, end to end

**What this is.** A decision-science portfolio that measures poker the way a quant measures a trading strategy: the question is never *"did I win?"* but *"is the edge real, or is it noise / overfitting?"* These five executed notebooks walk that arc from start to finish. Each reads top-to-bottom on GitHub (outputs are saved, no retraining): it loads the committed `../results/*.json`, recomputes the key statistic with the project's *own* functions, shows the matching figure from [`../figures/`](../figures/), and ends with an honest takeaway.

**The 60-second version.** One RL edge *resolves* over a weak baseline (+256 chips/match) but loses to a 0-parameter Kelly bot; exact game theory on Leduc reproduces Nash convergence; neural methods give honest nulls (and one qualified pass that only appeared after a transparently-reported bug-fix); a real-data behavioral finding holds on 777k human hands; and a pre-registered attempt to *exploit* opponent tilt resolves negative. A tempting, small-sample-supported edge that did not survive a powered test. The deliverable is the measurement discipline, not a single headline number.

**Read in order.** Each notebook is one chapter:

| # | Notebook | The question it answers | Honest result |
|---|---|---|---|
| 1 | [`01_evaluation_rigor.ipynb`](01_evaluation_rigor.ipynb) | Is a high-variance agent's edge *real*? Paired/mirror seeds, bootstrap CIs, the exact binomial sign test, and variance reduction (mirror + all-in-EV control variate). | RL beats the myopic baseline (+256, 95% CI [+144, +364]), but loses to a 0-parameter Kelly bot. |
| 2 | [`02_exploitability_leduc.ipynb`](02_exploitability_leduc.ipynb) | What does ground truth look like? Exact Leduc exploitability (NashConv), recomputed live. | The time-average converges toward Nash (0.009); the greedy last-iterate (the DQN regime) stays exploitable. |
| 3 | [`03_realdata_tilt.ipynb`](03_realdata_tilt.ipynb) | Is "tilt" real in *human* data? 777k PHH hands, within-player, vs a shuffled placebo. | Post-loss players are looser/more aggressive (small but real; placebo ~0). Opponent-model validation only, never the policy. |
| 4 | [`04_neural_scaling.ipynb`](04_neural_scaling.ipynb) | Is the edge a lucky seed, and do *neural* equilibrium methods beat tabular at scale? (§10/§11/§12) | Edge robust across 20 seeds; neural NFSP a qualified 2/3 sample-efficiency pass (after a corrected epsilon bug); at R=20 tabular CFR still beats neural (honest null). |
| 5 | [`05_exploitation.ipynb`](05_exploitation.ipynb) | Can we go *beyond* Nash and exploit detected tilt for EV? A DBR-style `p_tilted` knob, pre-registered (§13). | Resolved negative: the knob *hurts* (−169 chips/match), as loosening walks into an aggressive opponent. A powered pre-registration overturned a +152 small-sample peek. |

The thread that ties them together: pre-registration with a git-provable freeze→result gap (§10–§13), so a result counts the same whether it confirms the hypothesis or refutes it. Chapter 5 is the proof, where the freeze held even though the outcome flipped the opposite way.

Run / refresh:

```bash
python -m pip install -r ../requirements.txt   # adds pokerkit, hmmlearn (analysis-only)
jupyter nbconvert --to notebook --execute --inplace 01_evaluation_rigor.ipynb
# ... likewise 02_..., 03_..., 04_neural_scaling.ipynb, 05_exploitation.ipynb
```

`03` has an optional live-reproduction cell that needs the raw PHH subset
(`python -m scripts.fetch_phh`). Without it the notebook still stands on the
committed JSON. Narrative: [`../THESIS.md`](../THESIS.md). Bibliography:
[`../REFERENCES.md`](../REFERENCES.md). Pre-registration: [`../PREREGISTRATION.md`](../PREREGISTRATION.md).

> `00_origin.ipynb` is a private planning notebook, kept on local disk only. It
> is gitignored and not part of the repository history.
