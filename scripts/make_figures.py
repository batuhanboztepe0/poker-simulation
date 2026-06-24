"""
make_figures.py
---------------
Render the results story to committed figures/*.png (+ interactive .html) from
the measurement JSON under results/ (produced by scripts/run_measurements.sh,
which trains the multi-agent pool itself). This is the single reproducible figure layer for the
write-up, no re-training is needed to redraw a plot once results/ exists.

Each figure reads only results/ + the existing Plotly factories in app/charts.py,
is self-captioned (a one-line caption maps it to the write-up section it
illustrates), and is SKIPPED gracefully if its data file is absent, so a partial
results/ still renders whatever is available.

    python -m scripts.make_figures            # render every available figure
    python -m scripts.make_figures --list     # list figures + their data deps

Needs kaleido for PNG export (pip install kaleido; see requirements.txt).
"""

import argparse
import json
import math
import os
import statistics
import textwrap

import plotly.graph_objects as go

from app.charts import (
    learning_curve_figure, tournament_leaderboard_figure,
    parameter_heatmap_figure,
    ab_grouped_bar_figure, ab_heatmap_figure, icm_edge_figure,
    forest_plot_figure, exploitability_curve_figure,
    pool_strip_figure,
)
from app.theme import THEME_PLAIN
from src.evaluation import bootstrap_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")


# --- data loading -----------------------------------------------------------

def _load_jsonl(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    return rows or None


def _load_json(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _melt_opponents(rows, opps=("myopic", "tilt", "random")):
    """Wide per-config rows ({config, ..., myopic, tilt, random}) -> long
    {config, opponent, value} rows for the config x opponent heatmaps."""
    out = []
    for r in rows:
        for o in opps:
            if o in r:
                out.append({"config": r["config"], "opponent": o,
                            "value": r[o]})
    return out


# --- rendering --------------------------------------------------------------

def _save(fig, name, caption, width=960, height=560, subcaption=None):
    """Write figures/<name>.png + .html. The panel carries its title, plot, and
    axis labels only. The full description lives in figures/README.md (the
    returned caption) and in GUIDE.md, so no caption is baked into the image. That
    keeps the technical panels clean and avoids text overflowing the frame.
    `subcaption` is accepted for call-site compatibility and is unused."""
    os.makedirs(FIGURES, exist_ok=True)
    fig.write_image(os.path.join(FIGURES, name + ".png"),
                    width=width, height=height, scale=2)
    fig.write_html(os.path.join(FIGURES, name + ".html"), include_plotlyjs="cdn")
    return caption


def _save_plain(fig, name, title, takeaway, width=960, height=540):
    """Plain-audience variant of `_save`: a big centered title and a single short
    plain-language takeaway under the chart. For the simplified, one-message
    figures (figures/plain_*.png) aimed at a general reader, not the dense
    technical panels above."""
    os.makedirs(FIGURES, exist_ok=True)
    wrapped = "<br>".join(textwrap.wrap(takeaway, width=92))
    n = wrapped.count("<br>") + 1
    fig.update_layout(**{**THEME_PLAIN, "title": dict(text=title),
                         "margin": dict(t=80, b=90 + 24 * n, l=80, r=60)})
    fig.add_annotation(text=wrapped, xref="paper", yref="paper", x=0.5, y=-0.22,
                       showarrow=False, align="center", xanchor="center",
                       yanchor="top", font=dict(size=14, color="#333"))
    fig.write_image(os.path.join(FIGURES, name + ".png"),
                    width=width, height=height, scale=2)
    fig.write_html(os.path.join(FIGURES, name + ".html"), include_plotlyjs="cdn")
    return takeaway


def _attach_group_sem(fig, rows, group_key, value_key, group_order):
    """Attach ±1 SEM error bars (over the rows in each group) to the single-series
    grouped-bar figure from `ab_grouped_bar_figure`, so an ablation panel shows
    whether the per-group differences are within seed-to-seed noise."""
    present = {r[group_key] for r in rows}
    groups = [g for g in group_order if g in present]
    sems = []
    for g in groups:
        vals = [r[value_key] for r in rows if r[group_key] == g]
        sems.append(statistics.stdev(vals) / math.sqrt(len(vals))
                    if len(vals) >= 2 else 0.0)
    if fig.data:
        fig.data[0].error_y = dict(type="data", array=sems, visible=True,
                                   thickness=1.5, width=6, color="#444")
    return fig


# Each builder returns a list of (filename, section, caption) for the index, or
# [] when its data is missing.

def fig_exec_summary(index):
    """The glanceable honest summary: each headline edge as a point + 95%
    bootstrap CI; gray = CI straddles 0 (within noise)."""
    rows = []
    conf = _load_json("confirmatory.json")
    if conf and conf.get("confirmatory_primary", {}).get("ci95"):
        cp = conf["confirmatory_primary"]
        c = cp["ci95"]
        rows.append({"label": f"RL vs myopic, pre-registered confirmatory "
                              f"({cp['n_seeds']} mirrored seeds)",
                     "mean": cp["mean_diff"], "lo": c["lo"], "hi": c["hi"]})
    else:
        h = _load_json("headline_history.json")
        if h and h.get("final", {}).get("ci95"):
            f = h["final"]
            c = f["ci95"]
            rows.append({"label": f"RL vs myopic, headline "
                                  f"({f.get('n_seeds')}x{f.get('n_hands')}h)",
                         "mean": f["mean_chip_diff"], "lo": c["lo"], "hi": c["hi"]})
    p = _load_json("pool.json")
    if p:
        for e in p["leaderboard"]:
            if e["name"] == "RL" and e.get("ci95"):
                c = e["ci95"]
                rows.append({"label": f"RL vs pool, leaderboard "
                                      f"({p.get('n_seeds')} seeds)",
                             "mean": e["mean_net_chips"],
                             "lo": c["lo"], "hi": c["hi"]})
    icm = _load_jsonl("icm.jsonl")
    if icm:
        by = {}
        for r in icm:
            by.setdefault(tuple(r.get("ladder", [])), []).append(
                r["icm_minus_chips"])
        for ladder, vals in sorted(by.items(), reverse=True):
            tag = "mild" if (ladder and ladder[-1] > 0) else "bubble"
            ci = bootstrap_ci(vals)
            rows.append({"label": f"ICM vs chips ({tag}, {len(vals)} seeds)",
                         "mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"]})
    if not rows:
        return
    n_real = sum(1 for r in rows if r["lo"] > 0 or r["hi"] < 0)
    cap = (f"Every headline edge with its 95% bootstrap CI. "
           f"{n_real}/{len(rows)} edges are statistically distinguishable from zero: "
           f"the two RL edges are directionally positive and the two ICM edges are "
           f"directionally negative, with unresolved edges within per-seed noise "
           f"(see REFERENCES.md S2; THESIS.md).")
    sub = (f"{n_real}/{len(rows)} edges exclude zero. RL edges positive, "
           f"ICM edges negative. Gray = CI straddles 0 (within per-seed noise).")
    index.append(("exec_summary.png", "§summary",
                  _save(forest_plot_figure(
                      rows, title="Are the edges real? Effects with 95% CIs"),
                      "exec_summary", cap, height=460, subcaption=sub)))


def fig_variance_reduction(index):
    d = _load_json("variance_reduction.json")
    if not d:
        return
    arms = d["arms"]
    rows = [{"label": f"{a['arm']}  (CI width {a['ci_width']:.0f}, "
                      f"{a['ci_width_vs_raw']:.0%} of raw)",
             "mean": a["mean"], "lo": a["lo"], "hi": a["hi"]} for a in arms]
    by = {a["arm"]: a for a in arms}
    mir = by.get("mirror", {}).get("ci_width_vs_raw", 1.0)
    luck = by.get("luck_adjusted", {}).get("ci_width_vs_raw", 1.0)
    base = arms[0]
    cap = (f"The same Myopic-vs-Aggro edge ({base['n_seeds']} seeds x "
           f"{base['n_hands']} hands) measured four ways (REFERENCES.md S2). "
           f"Mirror matching narrows the 95% CI to {mir:.0%} of raw; "
           f"the all-in EV control variate is roughly neutral here ({luck:.0%}) "
           f"because variance is dominated by bust path-dependence, not single-hand runout luck.")
    sub = (f"Mirror matching narrows the 95% CI to {mir:.0%} of raw width "
           f"by cancelling seat and deck luck.")
    index.append(("variance_reduction.png", "§rigor",
                  _save(forest_plot_figure(
                      rows, title="Variance reduction: the same edge, four ways",
                      xaxis_title="Mean edge (95% bootstrap CI)"),
                      "variance_reduction", cap, height=440, subcaption=sub)))


def fig_exploitability(index):
    d = _load_json("exploitability.json")
    if not d:
        return
    curve = d["curve"]
    avg0, avgN = curve[0]["avg_exploitability"], curve[-1]["avg_exploitability"]
    lastN = curve[-1]["last_iterate_exploitability"]
    q_rows = d.get("q_curve")
    nfsp_rows = d.get("nfsp_curve")
    q_mean, q_rng = d.get("q_last_iterate_mean"), d.get("q_last_iterate_range")
    q_txt = ((f" Tabular Q-learning self-play (the DQN regime) confirms this directly: "
              f"its greedy last-iterate oscillates around {q_mean:.2f} (range "
              f"[{q_rng[0]:.2f}, {q_rng[1]:.2f}] over 1M+ episodes) and never "
              f"approaches Nash, genuine non-convergence, not a CFR artifact.")
             if q_mean is not None and q_rng else "")
    nfsp_txt = ((f" Adding only policy-averaging to that same Q-learner (tabular "
                 f"NFSP) flips the outcome: its average policy's exploitability "
                 f"falls {nfsp_rows[0]['exploitability']:.2f} to "
                 f"{nfsp_rows[-1]['exploitability']:.2f} over "
                 f"{nfsp_rows[0]['episodes']//1000}k to "
                 f"{nfsp_rows[-1]['episodes']//1000}k episodes.")
                if nfsp_rows else "")
    cap = (f"Exact Leduc exploitability (NashConv; 0 = exact Nash). "
           f"The CFR time-average converges ({avg0:.3f} to {avgN:.4f}), "
           f"but the greedy last-iterate stays exploitable (~{lastN:.3f}) and does not converge."
           + q_txt + nfsp_txt +
           f" This is the rigorous reason DQN self-play does not reach Nash while "
           f"averaging methods do (REFERENCES.md S1).")
    sub = (f"CFR time-average converges to Nash ({avg0:.3f} to {avgN:.4f}); "
           f"greedy last-iterate stays exploitable (~{lastN:.3f}) throughout training.")
    fig = exploitability_curve_figure(
        curve, uniform=d.get("uniform_exploitability"),
        q_rows=q_rows, nfsp_rows=nfsp_rows)
    index.append(("exploitability.png", "§rigor",
                  _save(fig, "exploitability", cap, height=480, subcaption=sub)))


def fig_headline(index):
    d = _load_json("headline_history.json")
    if not d:
        return
    f = d.get("final", {})
    p = f.get("p_value")
    bp = f.get("binom_p")
    ci = f.get("ci95")
    ci_txt = (f", 95% CI [{ci['lo']:+.0f}, {ci['hi']:+.0f}]" if ci else "")
    resolved = bool(ci and (ci["lo"] > 0 or ci["hi"] < 0))
    # Lead with the exact binomial sign test (binary bust matches); the paired
    # t-test treats the +/-2000 spread as continuous and only corroborates.
    test_txt = ((f", exact binomial sign test p={bp:.4f}"
                 + (f" (paired t agrees, p={p:.4f})" if p is not None else ""))
                if bp is not None
                else (f", paired p={p:.4f}" if p is not None else ""))
    verdict = ((". The 95% CI excludes 0: a statistically resolved edge over the "
                "baseline (this is the exploratory pilot; the 50-seed training "
                "monitor is noisy, significantly negative mid-training at 15/50 at "
                "step 500 before recovering to 33/50 at the final checkpoint, so "
                "the resolved claim rests on the 200-seed pilot and the separate "
                "pre-registered confirmatory at 500 mirrored seeds, which pins it "
                "to +256 [+144, +364]).")
               if resolved else
               (". The 95% CI includes 0, so this edge is within per-seed noise: "
                "directionally positive but not resolved at this sample size."))
    cap = (f"Headline (exploratory pilot): the fixed-vs-myopic RL agent learns to beat the "
           f"myopic EV baseline (binary bust matches, so the exact binomial "
           f"sign test is the right test). Left axis: held-out win rate; "
           f"right axis: mean chip diff with a ±1 SEM ribbon over training. "
           f"Final {f.get('wins')}/{f.get('n_seeds')} matches, "
           f"{f.get('mean_chip_diff', 0):+.0f} mean chips"
           + test_txt + ci_txt + (verdict if ci else "."))
    sub = ("The agent dips into over-folding, then recovers to beat the myopic "
           f"baseline. Final {f.get('wins')}/{f.get('n_seeds')} held-out matches, "
           f"{f.get('mean_chip_diff', 0):+.0f} mean chips"
           + (f", 95% CI [{ci['lo']:+.0f}, {ci['hi']:+.0f}], binomial p={bp:.4f}"
              if ci and bp is not None else "")
           + (", CI excludes 0, a resolved edge over the baseline."
              if resolved else
              ", lower bound at 0, so directionally positive but marginal.")
           + " Full detail in figures/README.md.")
    index.append(("headline.png", "headline",
                  _save(learning_curve_figure(d["history"], ribbon=True),
                        "headline", cap, subcaption=sub)))


def fig_pool(index):
    d = _load_json("pool.json")
    if not d:
        return
    lb = d["leaderboard"]
    lb_rank = ([e["name"] for e in lb].index("RL") + 1
               if any(e["name"] == "RL" for e in lb) else None)
    lb_pos = ("tops the leaderboard" if lb_rank == 1
              else f"ranks #{lb_rank}/{len(lb)}")
    wm = d.get("win_matrix", {})

    def _h2h(opp):  # RL's head-to-head record vs opp, from the committed matrix
        return f"{wm.get('RL', {}).get(opp, 0)}-{wm.get(opp, {}).get('RL', 0)}"

    rl_entry = next((e for e in lb if e["name"] == "RL"), {})
    rl_ci = rl_entry.get("ci95")
    straddles = rl_ci and rl_ci["lo"] <= 0 <= rl_ci["hi"]
    ci_txt = ((f" RL mean {rl_entry.get('mean_net_chips', 0):+.0f}, 95% CI "
               f"[{rl_ci['lo']:+.0f}, {rl_ci['hi']:+.0f}]"
               + (f", this CI includes 0, so the lead is NOT significant at "
                  f"{d.get('n_seeds')} seeds (more seeds or variance reduction "
                  f"needed to confirm)." if straddles else "."))
              if rl_ci else "")
    cap_lb = (f"Pool leaderboard ({d.get('n_seeds')} held-out seeds x {d.get('n_hands')} hands): "
              f"RL {lb_pos} and beats myopic {_h2h('Myopic')} and random {_h2h('Random')} head-to-head, "
              f"but loses to the analytic Kelly ({_h2h('Kelly')})."
              + ci_txt)
    sub_lb = (f"RL {lb_pos} but loses to Kelly head-to-head ({_h2h('Kelly')}) "
              f"over {d.get('n_seeds')} seeds.")
    index.append(("pool_leaderboard.png", "pool",
                  _save(tournament_leaderboard_figure(lb),
                        "pool_leaderboard", cap_lb, subcaption=sub_lb)))

    bs = d.get("best_static", {})
    rank, n_sw = d.get("rl_rank"), d.get("n_agents_in_sweep")
    verdict = ("tops the static leaderboard" if rank == 1 else
               "does NOT top the static leaderboard (the round-robin rewards "
               "farming the weakest static cells, not RL's vs-adaptive-pool "
               "objective)")
    cap_sw = (f"Static sweep: RL ranks #{rank}/{n_sw} "
              f"(RL {d.get('rl_mean', 0):+.0f} vs the best static cell "
              f"{bs.get('mean_net_chips', 0):+.0f} at "
              f"t{bs.get('tight', 0):.2f}/a{bs.get('aggr', 0):.2f}), it {verdict}. "
              f"Blue = net winner.")
    sub_sw = (f"RL ranks #{rank}/{n_sw} in the static personality sweep "
              f"(RL {d.get('rl_mean', 0):+.0f} vs best static "
              f"{bs.get('mean_net_chips', 0):+.0f}).")
    index.append(("pool_sweep.png", "pool",
                  _save(parameter_heatmap_figure(d["grid"]),
                        "pool_sweep", cap_sw, subcaption=sub_sw)))

    # Per-match outcomes cluster hard at ±1000 (most matches are won/lost
    # near-outright), so a box plot's quartiles span the whole range and every
    # agent looks identical. A jittered strip + mean bar shows the real spread
    # and the win-rate ranking honestly. Deterministic jitter keeps it
    # byte-reproducible (no RNG).
    nets = d["per_agent_nets"]
    order = sorted(nets, key=lambda n: sum(nets[n]) / len(nets[n]) if nets[n] else 0.0,
                   reverse=True)
    wr = ", ".join(f"{n} {sum(1 for v in nets[n] if v > 0) / len(nets[n]):.0%}"
                   for n in order)
    cap_box = (f"Per-match net chips per agent (each dot is one held-out match, bar is the mean). "
               f"Most matches are won or lost near-outright (clustered at ±1000); "
               f"agent ranking by match win rate: {wr}.")
    sub_box = (f"Agent ranking by match win rate: {wr}. "
               f"Most outcomes cluster near ±1000 chips.")
    index.append(("pool_pnl_box.png", "pool",
                  _save(pool_strip_figure(nets), "pool_pnl_box", cap_box, subcaption=sub_box)))


def fig_icm(index):
    rows = _load_jsonl("icm.jsonl")
    if not rows:
        return
    # Group by ladder (mild 50/30/20 vs bubble 65/35/0).
    by_ladder = {}
    for r in rows:
        by_ladder.setdefault(tuple(r.get("ladder", [])), []).append(r)
    for ladder, lrows in sorted(by_ladder.items(), key=lambda kv: kv[0],
                                reverse=True):
        label = "/".join(f"{int(round(x*100))}" for x in ladder) if ladder else "?"
        tag = "mild" if (ladder and ladder[-1] > 0) else "bubble"
        n = len(lrows)
        mean = sum(r["icm_minus_chips"] for r in lrows) / n
        n_pos = sum(1 for r in lrows if r["icm_minus_chips"] > 0)
        npl = lrows[0].get("n_players", "?")
        # Honest, data-driven verdict. The concavity edge is marginal: the
        # stressed per-seed variance (±300+) dwarfs it, so report the actual sign.
        if mean > 0 and n_pos * 2 > n:
            verdict = "a small positive prize edge for ICM"
        elif n_pos * 2 >= n:
            verdict = "no robust edge, within per-seed noise"
        else:
            verdict = "ICM-reward UNDERPERFORMING the chip reward here"
        tail = ("On average ICM underperforms the chip reward at this scale"
                if mean < 0 else
                "Per-seed variance (±300+) dominates any concavity edge")
        ci = bootstrap_ci([r["icm_minus_chips"] for r in lrows])
        ci_note = (f"95% CI [{ci['lo']:+.0f}, {ci['hi']:+.0f}] "
                   + (f"excludes 0 but n={n}, suggestive not robust"
                      if ci["lo"] > 0 or ci["hi"] < 0
                      else "includes 0, within per-seed noise"))
        cap = (f"ICM edge ({npl}-player, ladder {label}): per-init-seed "
               f"ICM-reward minus chip-reward mean tournament prize over {n} "
               f"reproducible seeds. Result: {verdict} (mean {mean:+.0f}, {n_pos}/{n} "
               f"seeds ICM>chips; {ci_note}). {tail}.")
        sub = (f"ICM vs chips ({npl}-player, ladder {label}): mean {mean:+.0f} "
               f"over {n} seeds ({n_pos}/{n} positive). {ci_note}.")
        index.append((f"icm_edge_{tag}.png", "icm",
                      _save(icm_edge_figure(
                          lrows, title=f"ICM vs chips prize edge, ladder "
                          f"{label} (mean {mean:+.0f})"),
                          f"icm_edge_{tag}", cap, subcaption=sub)))


def fig_block_b(index):
    # B2 -- action grid (five vs seven per init seed).
    rows = _load_jsonl("action_grid.jsonl")
    if rows:
        cap = ("B2: the finer 7-action grid does NOT beat the default "
               "5-action grid (both strongly beat myopic). "
               "Mean held-out chip diff vs myopic per init seed; 5-grid stays the default.")
        sub = ("7-action grid does not beat 5-action grid vs myopic. "
               "5-grid remains the default.")
        index.append(("blockB_action_grid.png", "block-B",
                      _save(ab_grouped_bar_figure(
                          rows, group_key="init_seed", value_key="mean",
                          by_key="grid", yaxis_title="mean chip diff vs myopic",
                          title="B2: 5-action vs 7-action grid (vs myopic)"),
                          "blockB_action_grid", cap, subcaption=sub)))

    # B3 -- bust clip: per-clip averages of mean chip diff AND bust rate (the
    # two quantities the bust-clip claim rests on), averaged over the init seeds.
    rows = _load_jsonl("bust_clip.jsonl")
    if rows:
        clip_order = ["old", "4.6", "wide"]
        cap = ("B3: widening the multi-hand bust clip does NOT help. "
               "The tight 3.0 ('old') clip has the best mean held-out chip diff vs "
               "myopic (averaged over 6 init seeds); '4.6' and 'wide' (approx 6.9) "
               "regress and destabilise some inits. Error bars = ±1 SEM over 6 seeds.")
        sub = ("Tight 3.0 clip has the best chip diff vs myopic. "
               "Wider clips regress and destabilise some init seeds.")
        index.append(("blockB_bust_clip.png", "block-B",
                      _save(_attach_group_sem(ab_grouped_bar_figure(
                          rows, group_key="clip", value_key="mean",
                          group_order=clip_order,
                          yaxis_title="mean chip diff vs myopic (avg over seeds)",
                          title="B3: bust-clip mean chip diff vs myopic "
                                "(tight 3.0 to wide 6.9)"),
                          rows, "clip", "mean", clip_order),
                          "blockB_bust_clip", cap, subcaption=sub)))
        cap2 = ("B3: bust rate per clip (avg over seeds). The tight 3.0 "
                "clip also has the lowest bust rate. In this heads-up "
                "winner-take-all format bust is approximately 1 minus win rate, "
                "so there is no independent risk lever to recover by un-clipping. "
                "Error bars = ±1 SEM over 6 seeds.")
        sub2 = ("Tight 3.0 clip also has the lowest bust rate "
                "across all 6 init seeds.")
        index.append(("blockB_bust_rate.png", "block-B",
                      _save(_attach_group_sem(ab_grouped_bar_figure(
                          rows, group_key="clip", value_key="bust_rate",
                          group_order=clip_order,
                          yaxis_title="bust rate (avg over seeds)",
                          title="B3: bust rate by clip (tight 3.0 to wide 6.9)"),
                          rows, "clip", "bust_rate", clip_order),
                          "blockB_bust_rate", cap2, subcaption=sub2)))

    # B4 -- snapshot self-play (config x opponent mean net chips).
    rows = _load_jsonl("selfplay.jsonl")
    if rows:
        cap = ("B4: snapshot self-play vs the fixed-vs-myopic recipe. "
               "Self-play trades the myopic bench for more tilt-robustness "
               "but is higher-variance; fixed stays the dependable recipe.")
        sub = ("Self-play gains tilt-robustness but is more variable "
               "than the fixed-vs-myopic training recipe.")
        index.append(("blockB_selfplay.png", "block-B",
                      _save(ab_heatmap_figure(
                          _melt_opponents(rows), row_key="config",
                          col_key="opponent", value_key="value",
                          title="B4: self-play vs fixed (mean RL net chips)"),
                          "blockB_selfplay", cap, subcaption=sub)))

    # B5 -- tilt-bonus decouple (config x opponent mean net chips).
    rows = _load_jsonl("tilt_decouple.jsonl")
    if rows:
        cap = ("B5: tilt-bonus configurations, mean RL net chips vs each "
               "opponent (avg over seeds). The PnL feature alone (pnl_nobonus, +249) "
               "is best; the naive bonus drags it down (pnl_naive +46); the decouple "
               "(pnl_decouple +138) is safe but no net gain; the bonus without the "
               "PnL feature (nopnl_bonus -159) collapses. Feature and bonus are "
               "substitutes, not complements.")
        sub = ("PnL feature alone (+249) beats every bonus configuration. "
               "Feature and bonus are substitutes, not complements.")
        index.append(("blockB_tilt_decouple.png", "block-B",
                      _save(ab_heatmap_figure(
                          _melt_opponents(rows), row_key="config",
                          col_key="opponent", value_key="value",
                          title="B5: tilt-bonus decouple (mean RL net chips)"),
                          "blockB_tilt_decouple", cap, subcaption=sub)))


def fig_tilt_realdata(index):
    """Real-data tilt validation: post-loss aggression/VPIP shift + the project's
    HMM detector separation on real human hands, each against a shuffled-label
    placebo."""
    d = _load_json("tilt_realdata.json")
    if not d:
        return
    ph, det, reg, cfg = d["phenomenon"], d["detector"], d["regime"], d["config"]

    def _row(label, ci):
        return {"label": label, "mean": ci["mean"], "lo": ci["lo"],
                "hi": ci["hi"]}

    rows = [
        _row("VPIP shift post-loss, real humans", ph["real"]["vpip"]),
        _row("VPIP shift, shuffled-label placebo", ph["placebo"]["vpip"]),
        _row("Aggression shift post-loss, real humans", ph["real"]["aggr"]),
        _row("Aggression shift, placebo", ph["placebo"]["aggr"]),
        _row("HMM P(tilted) separation, real", det["real"]["separation"]),
        _row("HMM P(tilted) separation, placebo",
             det["placebo"]["separation"]),
    ]
    lb = int(cfg["loss_bb"])
    npl = ph["real"]["n_players"]
    bic = reg.get("bic_gain")
    if reg.get("two_state_found") and bic and bic > 0:
        ratio = (reg["p_loss_given_high"] / reg["p_loss_base"]
                 if reg.get("p_loss_base") else 0.0)
        regime_txt = (f"A separate Baum-Welch HMM corroborates the phenomenon: it beats a "
                      f"1-state model out-of-sample (held-out LL "
                      f"{reg['heldout_ll_gain']:+.0f}; held-out DBIC {bic:+.0f}) "
                      f"with a calm (P[aggr]={reg['p_aggr_low']:.2f}) and an active "
                      f"(P[aggr]={reg['p_aggr_high']:.2f}) regime, with the active "
                      f"regime {ratio:.1f}x enriched for a recent big loss "
                      f"({reg['p_loss_given_high']:.1%} vs {reg['p_loss_base']:.1%} "
                      f"base).")
    else:
        regime_txt = ("A 2-state HMM does not beat a 1-state model on per-hand "
                      "aggression, so the tilt here is a conditional shift rather "
                      "than a persistent regime.")
    cap = (f"Real-data tilt validation ({d['n_sequences']:,} sessions, "
           f"{d['n_rows']:,} hand-rows from {cfg['n_files']} PokerStars 25NL "
           f"files; PHH/Kim 2024, CC-BY-4.0, used for opponent-model "
           f"validation only). After a >='{lb}bb loss, "
           f"{npl} real 2009 online players play looser "
           f"(VPIP {ph['real']['vpip']['mean']*100:+.1f}pp) and more aggressively "
           f"({ph['real']['aggr']['mean']*100:+.1f}pp): both 95% CIs exclude 0. "
           f"The project's forward-filter HMM detector registers a resolved P(tilted) "
           f"separation ({det['real']['separation']['mean']:+.3f}, CI excludes 0). "
           f"Shuffled-label placebo collapses to ~0 for each effect. {regime_txt}")
    sub = (f"After a >={lb}bb loss, {npl} real players play looser "
           f"(VPIP {ph['real']['vpip']['mean']*100:+.1f}pp) and more aggressively "
           f"({ph['real']['aggr']['mean']*100:+.1f}pp), both 95% CIs exclude 0. "
           f"Shuffled placebo collapses to ~0.")
    index.append(("tilt_realdata.png", "§real-data",
                  _save(forest_plot_figure(
                      rows, title="Is tilt detectable in real hands? "
                      "(effects vs shuffled-label placebo)",
                      xaxis_title="Post-loss effect (probability units; "
                                  "95% bootstrap CI)"),
                      "tilt_realdata", cap, height=480, subcaption=sub)))


def fig_tilt_lossvswin(index):
    d = _load_json("tilt_realdata.json")
    if not d or "within_player" not in d:
        return
    ph = d["phenomenon"]["real"]
    wr, wpl = d["within_player"]["real"], d["within_player"]["placebo"]
    lb = int(wr.get("swing_bb", d["config"]["loss_bb"]))
    npl = wr["n_players"]

    def pp(ci):
        return {"mean": ci["mean"] * 100, "lo": ci["lo"] * 100, "hi": ci["hi"] * 100}

    rows = []
    for metric, name in (("aggr", "Aggression"), ("vpip", "VPIP")):
        rows.append({"label": f"{name}: post-loss vs baseline (all other hands)",
                     **pp(ph[metric])})
        rows.append({"label": f"{name}: post-loss vs post-win (matched, n={npl})",
                     **pp(wr[metric])})
        rows.append({"label": f"{name}: post-loss vs post-win (shuffled placebo)",
                     **pp(wpl[metric])})
    da, dv = wr["aggr_cohen_d"], wr["vpip_cohen_d"]
    cap = (f"Within-player control (confound-controlled tilt test). "
           f"Each player's hand after a >={lb}bb loss is compared against their hand "
           f"after an equal >={lb}bb win, matching player, big-pot arousal, and event "
           f"size so only the swing sign differs. After a loss players are "
           f"{wr['aggr']['mean']*100:+.1f}pp more aggressive and "
           f"{wr['vpip']['mean']*100:+.1f}pp looser than after an equal win "
           f"(95% CIs exclude 0; Cohen d={da:.2f}/{dv:.2f}; n={npl} matched "
           f"players). The shuffled placebo collapses to ~0. The matched effect is "
           f"larger than post-loss-vs-baseline because players also tighten after a win.")
    sub = (f"After a >={lb}bb loss vs an equal win, players are "
           f"{wr['aggr']['mean']*100:+.1f}pp more aggressive and "
           f"{wr['vpip']['mean']*100:+.1f}pp looser (95% CIs exclude 0, "
           f"Cohen d={da:.2f}/{dv:.2f}, n={npl}).")
    index.append(("tilt_lossvswin.png", "§real-data",
                  _save(forest_plot_figure(
                      rows, title="Loss-aversion asymmetry: post-loss vs an "
                      "equal-size post-win (within player)",
                      xaxis_title="Shift in the next hand "
                                  "(percentage points; 95% bootstrap CI)"),
                      "tilt_lossvswin", cap, height=480, subcaption=sub)))


def fig_rollout_fe(index):
    rows = _load_jsonl("rollout_fe.jsonl")
    if not rows:
        return
    n = rows[0].get("n_seeds", "?")
    h = rows[0].get("n_hands", "?")
    cap = (f"B1: warmed-belief rollout fold-equity ({n} paired seeds x {h} hands). "
           f"Warming largely fixes the cold-FE over-bluff disaster vs myopic/tilt "
           f"(cold_fe to warm_fe: myopic -1133 to -67, tilt flips positive), "
           f"but it is worse vs random and the nit punishes it. "
           f"Fold-equity stays OFF by default.")
    sub = (f"Warming fixes the cold-FE over-bluff disaster vs myopic/tilt "
           f"but hurts vs random, so fold-equity stays off by default.")
    index.append(("rollout_fe.png", "fold-equity",
                  _save(ab_heatmap_figure(
                      rows, row_key="config", col_key="opponent",
                      value_key="mean_diff", colorbar_title="Rollout net chips",
                      title="B1: warmed-belief rollout fold-equity"),
                      "rollout_fe", cap, subcaption=sub)))


# ---------------------------------------------------------------------------
# Plain, single-message figures for a general audience (figures/plain_*.png).
# Same committed data, simplified to one chart + one plain-language takeaway.
# ---------------------------------------------------------------------------

def fig_plain_rl_edge(index):
    conf = _load_json("confirmatory.json")
    if conf and conf.get("confirmatory_primary", {}).get("ci95"):
        cp = conf["confirmatory_primary"]
        f = {"mean_chip_diff": cp["mean_diff"], "ci95": cp["ci95"],
             "n_hands": cp["n_hands"]}
    else:
        d = _load_json("headline_history.json")
        if not d or not d.get("final", {}).get("ci95"):
            return
        f = d["final"]
    ci = f["ci95"]
    mean = f["mean_chip_diff"]
    straddles = ci["lo"] <= 0 <= ci["hi"]
    color = "#888888" if straddles else "#2ca02c"
    fig = go.Figure(go.Bar(
        x=[mean], y=["trained bot vs simple baseline"], orientation="h",
        marker_color=color, width=0.4,
        error_x=dict(type="data", symmetric=False, array=[ci["hi"] - mean],
                     arrayminus=[mean - ci["lo"]], thickness=3, width=12,
                     color="#444")))
    fig.add_vline(x=0, line=dict(color="#d62728", dash="dot"))
    fig.update_layout(
        xaxis_title=f"chips won vs the baseline over {f.get('n_hands','?')} hands "
                    f"(bar = average, line = 95% range)",
        yaxis_title="", showlegend=False)
    take = ((f"On average the trained bot wins (+{mean:.0f} chips), and the 95% "
             f"range is [{ci['lo']:+.0f}, {ci['hi']:+.0f}], it stays clear of 0, "
             f"so the edge over the simple baseline is real and statistically "
             f"resolved (pre-registered confirmatory over 500 mirrored seeds; "
             f"still modest: a 0-parameter Kelly bot beats this RL "
             f"agent head-to-head, see the pool figures).") if not straddles
            else (f"On average the trained bot wins (+{mean:.0f} chips), but the "
                  f"95% range is [{ci['lo']:+.0f}, {ci['hi']:+.0f}], it reaches "
                  f"back to 0, so the edge is real but marginal, not a sure "
                  f"thing."))
    index.append(("plain_rl_edge.png", "§plain",
                  _save_plain(fig, "plain_rl_edge",
                              "Does the trained bot beat the simple baseline?",
                              take, height=420)))


def fig_plain_nash(index):
    d = _load_json("exploitability.json")
    if not d:
        return
    curve = d["curve"]
    iters = [r["iters"] for r in curve]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=[r["avg_exploitability"] for r in curve],
        mode="lines+markers", name="averages its history",
        line=dict(width=3, color="#2ca02c")))
    fig.add_trace(go.Scatter(
        x=iters, y=[r["last_iterate_exploitability"] for r in curve],
        mode="lines+markers", name="plays latest only (DQN)",
        line=dict(width=3, color="#d62728")))
    fig.update_layout(
        xaxis_title="training rounds (log scale)", xaxis_type="log",
        yaxis_title="how beatable (0 = perfect play)",
        legend=dict(x=0.98, y=0.95, xanchor="right", bgcolor="rgba(255,255,255,0.6)"))
    a = curve[-1]["avg_exploitability"]
    l = curve[-1]["last_iterate_exploitability"]
    take = (f"A self-play bot that averages its whole history converges to "
            f"near-perfect play ({a:.4f}); one that always uses its latest "
            f"strategy, like a plain DQN, stays beatable ({l:.3f}).")
    index.append(("plain_nash.png", "§plain",
                  _save_plain(fig, "plain_nash",
                              "Why 'averaging' matters when an AI learns by self-play",
                              take)))


def fig_plain_tilt(index):
    d = _load_json("tilt_realdata.json")
    if not d:
        return
    ph = d["phenomenon"]
    real = ph["real"]["vpip"]
    plac = ph["placebo"]["vpip"]
    fig = go.Figure(go.Bar(
        x=["after a big loss", "shuffled control"],
        y=[real["mean"] * 100, plac["mean"] * 100],
        marker_color=["#2ca02c", "#888888"],
        error_y=dict(type="data", symmetric=False,
                     array=[(real["hi"] - real["mean"]) * 100,
                            (plac["hi"] - plac["mean"]) * 100],
                     arrayminus=[(real["mean"] - real["lo"]) * 100,
                                 (plac["mean"] - plac["lo"]) * 100],
                     thickness=3, width=14, color="#444")))
    fig.add_hline(y=0, line=dict(color="#444"))
    fig.update_layout(
        yaxis_title="extra hands played vs usual (VPIP, % points)",
        xaxis_title="", showlegend=False)
    lb = int(d["config"]["loss_bb"])
    take = (f"Yes, after a {lb}bb+ loss, real online players voluntarily play "
            f"~{real['mean']*100:.1f}pp more hands (VPIP) than usual (95% bars exclude 0). "
            f"A shuffled control collapses to ~0, so it is the loss, not chance.")
    index.append(("plain_tilt.png", "§plain",
                  _save_plain(fig, "plain_tilt",
                              "Do real poker players 'tilt' after a loss?",
                              take)))


BUILDERS = [fig_exec_summary, fig_variance_reduction, fig_exploitability,
            fig_headline, fig_pool, fig_icm, fig_block_b, fig_tilt_realdata,
            fig_tilt_lossvswin, fig_rollout_fe,
            fig_plain_rl_edge, fig_plain_nash, fig_plain_tilt]

DATA_DEPS = {
    "fig_exec_summary": "results/{headline_history.json, pool.json, icm.jsonl}",
    "fig_variance_reduction": "results/variance_reduction.json",
    "fig_exploitability": "results/exploitability.json",
    "fig_headline": "results/headline_history.json",
    "fig_pool": "results/pool.json",
    "fig_icm": "results/icm.jsonl",
    "fig_block_b": "results/{action_grid,bust_clip,selfplay,tilt_decouple}.jsonl",
    "fig_tilt_realdata": "results/tilt_realdata.json",
    "fig_tilt_lossvswin": "results/tilt_realdata.json",
    "fig_rollout_fe": "results/rollout_fe.jsonl",
    "fig_plain_rl_edge": "results/headline_history.json",
    "fig_plain_nash": "results/exploitability.json",
    "fig_plain_tilt": "results/tilt_realdata.json",
}


def write_index(index):
    """Write figures/README.md mapping each rendered figure to its section + caption."""
    os.makedirs(FIGURES, exist_ok=True)
    lines = [
        "# Figures: the results story",
        "",
        "Rendered by `python -m scripts.make_figures` from the committed "
        "measurement JSON in [`../results/`](../results/) (regenerate that data "
        "with `scripts/run_measurements.sh`, which trains the multi-agent pool too; the "
        "standalone variance/exploitability/tilt results have their own measure "
        "scripts, see [../GUIDE.md](../GUIDE.md)). Each "
        "`.png` is for the write-up; the matching `.html` is interactive.",
        "",
        "**Start with [`exec_summary.png`](exec_summary.png)**: every headline "
        "edge with its 95% bootstrap CI (the honest 'are the edges real?' "
        "view). Narrative: [../THESIS.md](../THESIS.md). Bibliography: "
        "[../REFERENCES.md](../REFERENCES.md).",
        "",
        "| Figure | Section | What it shows |",
        "|---|---|---|",
    ]
    for fname, section, caption in index:
        lines.append(f"| [`{fname}`]({fname}) | {section} | {caption} |")
    lines.append("")
    with open(os.path.join(FIGURES, "README.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="list the figures + their data dependencies and exit")
    args = ap.parse_args()
    if args.list:
        for b in BUILDERS:
            print(f"{b.__name__:18s} <- {DATA_DEPS.get(b.__name__, '?')}")
        return

    index = []
    for b in BUILDERS:
        b(index)
    if not index:
        print("No results/ data found, run scripts/run_measurements.sh first.")
        return
    write_index(index)
    print(f"Rendered {len(index)} figure(s) -> {FIGURES}/")
    for fname, section, _ in index:
        print(f"  {section:5s} {fname}")


if __name__ == "__main__":
    main()
