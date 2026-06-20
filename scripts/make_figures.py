"""
make_figures.py
---------------
Render the results story to committed figures/*.png (+ interactive .html) from
the measurement JSON under results/ (produced by scripts/run_measurements.sh and
scripts/measure_pool.py). This is the single reproducible figure layer for the
write-up — no re-training is needed to redraw a plot once results/ exists.

Each figure reads only results/ + the existing Plotly factories in app/charts.py,
is self-captioned (a one-line caption maps it to the RL_HANDOFF section it
illustrates), and is SKIPPED gracefully if its data file is absent — so a partial
results/ still renders whatever is available.

    python -m scripts.make_figures            # render every available figure
    python -m scripts.make_figures --list     # list figures + their data deps

Needs kaleido for PNG export (pip install kaleido; see requirements.txt).
"""

import argparse
import json
import os
import textwrap

from app.charts import (
    learning_curve_figure, tournament_leaderboard_figure,
    parameter_heatmap_figure, pnl_box_figure,
    ab_grouped_bar_figure, ab_heatmap_figure, icm_edge_figure,
    forest_plot_figure,
)
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
    {config, opponent, value} rows for the config×opponent heatmaps."""
    out = []
    for r in rows:
        for o in opps:
            if o in r:
                out.append({"config": r["config"], "opponent": o,
                            "value": r[o]})
    return out


# --- rendering --------------------------------------------------------------

def _save(fig, name, caption, width=960, height=560):
    """Write figures/<name>.png + .html with a wrapped bottom caption."""
    os.makedirs(FIGURES, exist_ok=True)
    wrapped = "<br>".join(textwrap.wrap(caption, width=108))
    n_lines = wrapped.count("<br>") + 1
    fig.update_layout(margin=dict(b=70 + 24 * n_lines, t=70))
    # Anchored top-left in the bottom margin (below the x-axis title), grows down.
    fig.add_annotation(text=wrapped, xref="paper", yref="paper", x=0, y=-0.22,
                       showarrow=False, align="left", xanchor="left",
                       yanchor="top", font=dict(size=11, color="#555"))
    fig.write_image(os.path.join(FIGURES, name + ".png"),
                    width=width, height=height, scale=2)
    fig.write_html(os.path.join(FIGURES, name + ".html"), include_plotlyjs="cdn")
    return caption


# Each builder returns a list of (filename, section, caption) for the index, or
# [] when its data is missing.

def fig_exec_summary(index):
    """The glanceable honest summary: each headline edge as a point + 95%
    bootstrap CI; gray = CI straddles 0 (within noise)."""
    rows = []
    h = _load_json("headline_history.json")
    if h and h.get("final", {}).get("ci95"):
        f = h["final"]
        c = f["ci95"]
        rows.append({"label": f"RL vs myopic — headline "
                              f"({f.get('n_seeds')}×{f.get('n_hands')}h)",
                     "mean": f["mean_chip_diff"], "lo": c["lo"], "hi": c["hi"]})
    p = _load_json("pool.json")
    if p:
        for e in p["leaderboard"]:
            if e["name"] == "RL" and e.get("ci95"):
                c = e["ci95"]
                rows.append({"label": f"RL vs pool — leaderboard "
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
            rows.append({"label": f"ICM − chips — {tag} ({len(vals)} seeds)",
                         "mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"]})
    if not rows:
        return
    n_real = sum(1 for r in rows if r["lo"] > 0 or r["hi"] < 0)
    cap = ("EXECUTIVE SUMMARY — every headline edge as a point with its 95% "
           "bootstrap CI. Gray = the CI straddles 0 (effect is within per-seed "
           f"noise); green/red = CI excludes 0. {n_real}/{len(rows)} edges are "
           "statistically distinguishable from zero at these sample sizes — the "
           "honest takeaway: the agent is directionally positive but the edges "
           "are marginal, exactly what rigorous variance accounting should show "
           "(see references.md §2; THESIS.md).")
    index.append(("exec_summary.png", "§summary",
                  _save(forest_plot_figure(
                      rows, title="Are the edges real? Effects with 95% CIs"),
                      "exec_summary", cap, height=460)))


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
    cap = (f"Variance reduction (references.md §2): the SAME Myopic-vs-Aggro edge "
           f"({base['n_seeds']} seeds × {base['n_hands']} hands) measured four "
           f"ways. Duplicate/mirror matching narrows the 95% CI to {mir:.0%} of "
           f"raw (cancels seat/deck luck); the all-in EV control variate is "
           f"~neutral here ({luck:.0%}) because in a multi-hand BUST match the "
           f"variance is dominated by bust path-dependence, not single-hand "
           f"runout luck (the big AIVAT gains are for per-hand win-rate "
           f"estimation). Narrower CI = same conclusion from fewer matches.")
    index.append(("variance_reduction.png", "§rigor",
                  _save(forest_plot_figure(
                      rows, title="Variance reduction: the same edge, four ways",
                      xaxis_title="Mean edge (95% bootstrap CI)"),
                      "variance_reduction", cap, height=440)))


def fig_headline(index):
    d = _load_json("headline_history.json")
    if not d:
        return
    f = d.get("final", {})
    p = f.get("p_value")
    ci = f.get("ci95")
    ci_txt = (f", 95% CI [{ci['lo']:+.0f}, {ci['hi']:+.0f}]" if ci else "")
    cap = (f"Headline (§8): the fixed-vs-myopic RL agent learns to beat the "
           f"myopic EV baseline. Left axis = held-out win rate; right axis = "
           f"mean chip diff with a ±1 SEM ribbon over training. "
           f"Final {f.get('wins')}/{f.get('n_seeds')} matches, "
           f"{f.get('mean_chip_diff', 0):+.0f} mean chips"
           + (f", paired p={p:.4f}" if p is not None else "") + ci_txt
           + (". The CI's lower bound near 0 shows the edge is real but marginal "
              "— variance accounting matters (see exec_summary)." if ci else "."))
    index.append(("headline.png", "§8",
                  _save(learning_curve_figure(d["history"], ribbon=True),
                        "headline", cap)))


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
               + (f" — this CI includes 0, so the lead is NOT significant at "
                  f"{d.get('n_seeds')} seeds (more seeds / variance reduction "
                  f"needed to confirm)." if straddles else "."))
              if rl_ci else "")
    cap_lb = (f"§10 generalist (RL = belief + sharp HMM + PnL feed + "
              f"opponent-mix, multi-hand chips, {d.get('steps')} steps): "
              f"cross-agent leaderboard over {d.get('n_seeds')} held-out seeds × "
              f"{d.get('n_hands')} hands. RL {lb_pos} and beats each of the "
              f"{{myopic, tilt, random}} adaptive opponents head-to-head "
              f"({_h2h('Myopic')} / {_h2h('Tilt')} / {_h2h('Random')}); it "
              f"loses H2H to the analytic Kelly ({_h2h('Kelly')}). The §10 win "
              f"is the leaderboard + adaptive-pool result, not Kelly H2H." + ci_txt)
    index.append(("pool_leaderboard.png", "§10",
                  _save(tournament_leaderboard_figure(lb),
                        "pool_leaderboard", cap_lb)))

    bs = d.get("best_static", {})
    rank, n_sw = d.get("rl_rank"), d.get("n_agents_in_sweep")
    verdict = ("and tops it" if rank == 1 else
               f"and does NOT top it (the round-robin rewards farming the "
               f"weakest static cells, not RL's vs-adaptive-pool objective)")
    cap_sw = (f"§10: the same RL generalist dropped into a round-robin of static "
              f"(tight × aggression) personalities. RL ranks #{rank}/{n_sw} "
              f"(RL {d.get('rl_mean', 0):+.0f} vs the best static cell "
              f"{bs.get('mean_net_chips', 0):+.0f} at "
              f"t{bs.get('tight', 0):.2f}/a{bs.get('aggr', 0):.2f}) — it beats "
              f"the adaptive pool (left figure) {verdict}. Blue = net winner.")
    index.append(("pool_sweep.png", "§10",
                  _save(parameter_heatmap_figure(d["grid"]),
                        "pool_sweep", cap_sw)))

    cap_box = ("§10: per-seed net-chip distribution per agent (the spread behind "
               "the leaderboard means — poker variance is wide).")
    index.append(("pool_pnl_box.png", "§10",
                  _save(pnl_box_figure(d["per_agent_nets"]),
                        "pool_pnl_box", cap_box)))


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
        # Honest, data-driven verdict. The concavity edge is marginal and §13
        # stressed per-seed variance (±300+) dwarfs it, so report the actual sign.
        if mean > 0 and n_pos * 2 > n:
            verdict = "a small positive prize edge for ICM"
        elif n_pos * 2 >= n:
            verdict = "no robust edge — within per-seed noise"
        else:
            verdict = "ICM-reward UNDERPERFORMING the chip reward here"
        tail = ("so ICM UNDERPERFORMS the chip reward on average at this scale"
                if mean < 0 else
                "so per-seed variance (±300+) dominates any concavity edge")
        cap = (f"§13 ICM edge ({npl}-player, ladder {label}): per-init-seed "
               f"ICM-reward minus chip-reward mean tournament prize over {n} "
               f"reproducible seeds — {verdict} (mean {mean:+.0f}, {n_pos}/{n} "
               f"seeds ICM>chips). The earlier §13 small-positive estimate came "
               f"from 3 noisier unseeded inits; {tail}.")
        index.append((f"icm_edge_{tag}.png", "§13",
                      _save(icm_edge_figure(
                          lrows, title=f"ICM − chips prize edge — ladder "
                          f"{label} (mean {mean:+.0f})"),
                          f"icm_edge_{tag}", cap)))


def fig_block_b(index):
    # B2 — action grid (five vs seven per init seed).
    rows = _load_jsonl("action_grid.jsonl")
    if rows:
        cap = ("B2 (§16): the finer 7-action grid does NOT beat the default "
               "5-action grid (both strongly beat myopic) — extra sizings just "
               "spread the budget over more actions. Mean held-out chip diff vs "
               "myopic, per init seed. 5-grid stays the default.")
        index.append(("blockB_action_grid.png", "§16",
                      _save(ab_grouped_bar_figure(
                          rows, group_key="init_seed", value_key="mean",
                          by_key="grid", yaxis_title="mean chip diff vs myopic",
                          title="B2: 5-action vs 7-action grid (vs myopic)"),
                          "blockB_action_grid", cap)))

    # B3 — bust clip: per-clip averages of mean chip diff AND bust rate (the
    # two quantities the §15 claim rests on), averaged over the init seeds.
    rows = _load_jsonl("bust_clip.jsonl")
    if rows:
        cap = ("B3 (§15): widening the multi-hand bust clip does NOT help — the "
               "tight 3.0 ('old') clip has the best mean held-out chip diff vs "
               "myopic (averaged over 6 init seeds); '4.6' and 'wide' (≈6.9) "
               "regress and destabilise some inits (high weight-init variance).")
        index.append(("blockB_bust_clip.png", "§15",
                      _save(ab_grouped_bar_figure(
                          rows, group_key="clip", value_key="mean",
                          group_order=["old", "4.6", "wide"],
                          yaxis_title="mean chip diff vs myopic (avg over seeds)",
                          title="B3: bust-clip mean chip diff vs myopic "
                                "(tight 3.0 → wide 6.9)"),
                          "blockB_bust_clip", cap)))
        cap2 = ("B3 (§15): bust rate per clip (avg over seeds). The tight 3.0 "
                "clip also has the lowest bust rate — in this heads-up "
                "winner-take-all format bust ≈ 1 − win rate, so there is no "
                "independent risk lever to recover by un-clipping the ruin signal.")
        index.append(("blockB_bust_rate.png", "§15",
                      _save(ab_grouped_bar_figure(
                          rows, group_key="clip", value_key="bust_rate",
                          group_order=["old", "4.6", "wide"],
                          yaxis_title="bust rate (avg over seeds)",
                          title="B3: bust rate by clip (tight 3.0 → wide 6.9)"),
                          "blockB_bust_rate", cap2)))

    # B4 — snapshot self-play (config × opponent mean net chips).
    rows = _load_jsonl("selfplay.jsonl")
    if rows:
        cap = ("B4 (§18): snapshot self-play vs the fixed-vs-myopic recipe, mean "
               "RL net chips vs each opponent (averaged over init seeds). "
               "Self-play trades the myopic bench for more tilt-robustness but "
               "is higher-variance — fixed stays the dependable recipe.")
        index.append(("blockB_selfplay.png", "§18",
                      _save(ab_heatmap_figure(
                          _melt_opponents(rows), row_key="config",
                          col_key="opponent", value_key="value",
                          title="B4: self-play vs fixed (mean RL net chips)"),
                          "blockB_selfplay", cap)))

    # B5 — tilt-bonus decouple (config × opponent mean net chips).
    rows = _load_jsonl("tilt_decouple.jsonl")
    if rows:
        cap = ("B5 (§17): tilt-bonus configurations, mean RL net chips vs each "
               "opponent (avg over seeds). The PnL feature ALONE (pnl_nobonus, "
               "+249) is best; adding the naive bonus drags it down (pnl_naive "
               "+46 — the footgun, worse than the feature alone); the decouple "
               "(pnl_decouple +138) is safe but no net gain; the bonus WITHOUT "
               "the PnL feature (nopnl_bonus −159) collapses. Feature and bonus "
               "are substitutes, not complements.")
        index.append(("blockB_tilt_decouple.png", "§17",
                      _save(ab_heatmap_figure(
                          _melt_opponents(rows), row_key="config",
                          col_key="opponent", value_key="value",
                          title="B5: tilt-bonus decouple (mean RL net chips)"),
                          "blockB_tilt_decouple", cap)))


def fig_rollout_fe(index):
    rows = _load_jsonl("rollout_fe.jsonl")
    if not rows:
        return
    n = rows[0].get("n_seeds", "?")
    h = rows[0].get("n_hands", "?")
    cap = (f"B1 (§14): warmed-belief rollout fold-equity, net chips of the "
           f"rollout vs each opponent ({n} paired seeds × {h} hands). Warming "
           f"LARGELY fixes the cold-FE over-bluff disaster vs myopic/tilt "
           f"(cold_fe → warm_fe rows: myopic −1133→−67, tilt flips positive), "
           f"but it is NOT a uniform win — it is worse vs random and the nit "
           f"punishes it — so fold-equity stays OFF by default. Blue = rollout "
           f"wins chips.")
    index.append(("rollout_fe.png", "§14",
                  _save(ab_heatmap_figure(
                      rows, row_key="config", col_key="opponent",
                      value_key="mean_diff", colorbar_title="Rollout net chips",
                      title="B1: warmed-belief rollout fold-equity"),
                      "rollout_fe", cap)))


BUILDERS = [fig_exec_summary, fig_variance_reduction, fig_headline, fig_pool,
            fig_icm, fig_block_b, fig_rollout_fe]

DATA_DEPS = {
    "fig_exec_summary": "results/{headline_history.json, pool.json, icm.jsonl}",
    "fig_variance_reduction": "results/variance_reduction.json",
    "fig_headline": "results/headline_history.json",
    "fig_pool": "results/pool.json",
    "fig_icm": "results/icm.jsonl",
    "fig_block_b": "results/{action_grid,bust_clip,selfplay,tilt_decouple}.jsonl",
    "fig_rollout_fe": "results/rollout_fe.jsonl",
}


def write_index(index):
    """Write figures/README.md mapping each rendered figure to its § + caption."""
    os.makedirs(FIGURES, exist_ok=True)
    lines = [
        "# Figures — the results story",
        "",
        "Rendered by `python -m scripts.make_figures` from the committed "
        "measurement JSON in [`../results/`](../results/) (regenerate that data "
        "with `scripts/run_measurements.sh` + `scripts/measure_pool.py`). Each "
        "`.png` is for the write-up; the matching `.html` is interactive.",
        "",
        "**Start with [`exec_summary.png`](exec_summary.png)** — every headline "
        "edge with its 95% bootstrap CI (the honest \"are the edges real?\" "
        "view). Narrative: [../THESIS.md](../THESIS.md). Bibliography: "
        "[../references.md](../references.md).",
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
        print("No results/ data found — run scripts/run_measurements.sh first.")
        return
    write_index(index)
    print(f"Rendered {len(index)} figure(s) -> {FIGURES}/")
    for fname, section, _ in index:
        print(f"  {section:5s} {fname}")


if __name__ == "__main__":
    main()
