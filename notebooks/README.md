# Notebooks — reproduce the results

Three executed notebooks that reproduce the project's headline claims **from the
committed `../results/*.json` — no retraining**. Each loads the data, recomputes
the key statistics with the project's own functions, displays the matching figure
from [`../figures/`](../figures/), and ends with an honest takeaway. Outputs are
saved in the notebooks, so they read top-to-bottom on GitHub without running.

| Notebook | What it reproduces |
|---|---|
| [`01_evaluation_rigor.ipynb`](01_evaluation_rigor.ipynb) | The headline RL-vs-myopic edge re-derived from per-seed PnL (bootstrap CI + paired t-test), variance reduction (mirror matching + all-in-EV control variate), and the "are the edges real?" executive summary. |
| [`02_exploitability_leduc.ipynb`](02_exploitability_leduc.ipynb) | Exact Leduc exploitability (NashConv): the time-average strategy → Nash vs the greedy last-iterate (the DQN-self-play regime) staying exploitable, with a live recompute of the metric. |
| [`03_realdata_tilt.ipynb`](03_realdata_tilt.ipynb) | Tilt validation on real human hands (PHH/Kim 2024, CC-BY-4.0): post-loss looseness/aggression vs a shuffled-label placebo, the forward-filter HMM detector separation, and a separate Baum-Welch regime fit. Opponent-model validation only — never the policy. |

Run/refresh them with:

```bash
python -m pip install -r ../requirements.txt   # adds pokerkit, hmmlearn (analysis-only)
jupyter nbconvert --to notebook --execute --inplace 01_evaluation_rigor.ipynb
```

`03` has an optional live-reproduction cell that needs the raw PHH subset
(`python -m scripts.fetch_phh`); without it the notebook still stands on the
committed JSON. Narrative: [`../THESIS.md`](../THESIS.md). Bibliography:
[`../REFERENCES.md`](../REFERENCES.md).

> `00_origin.ipynb` is the private origin notebook and is intentionally
> gitignored.
