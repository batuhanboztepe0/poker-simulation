"""
charts.py
---------
Pure Plotly figure factories for the dashboard and write-up figures.

Every function takes plain data (DataFrames / dicts produced by src.analytics)
and returns a plotly.graph_objects.Figure. No Streamlit, no I/O, no global
state — so each factory is unit-testable on a fixture.

Style contract: all factories call apply_theme(fig, ...) so that
changing app/theme.py is the only thing needed to restyle every figure.
"""

import math
import statistics

import plotly.graph_objects as go

from app.theme import COLORS, THEME, THEME_PLAIN, apply_theme, wrap_label


# ---------------------------------------------------------------------------
# Dashboard figures (Phase 2 — equity / belief / EV panels)
# ---------------------------------------------------------------------------

def equity_curve_figure(equity_df, name_by_id=None):
    """
    Line chart of cumulative stack per player over hands.

    Args:
        equity_df (DataFrame): columns [hand_number, player_id, stack].
        name_by_id (dict | None): optional {player_id: display_name}.

    Returns:
        go.Figure
    """
    name_by_id = name_by_id or {}
    fig = go.Figure()
    for pid in sorted(equity_df["player_id"].unique()):
        sub = equity_df[equity_df["player_id"] == pid].sort_values("hand_number")
        fig.add_trace(go.Scatter(
            x=sub["hand_number"], y=sub["stack"],
            mode="lines", name=str(name_by_id.get(pid, f"Player {pid}")),
        ))
    apply_theme(fig,
        title="Stack trajectory (equity curve)",
        xaxis_title="Hand",
        yaxis_title="Chips",
    )
    return fig


def win_rate_figure(win_rates, name_by_id=None):
    """
    Bar chart of per-player hand win rate.

    Args:
        win_rates (dict): {player_id: win_rate in [0, 1]}.
        name_by_id (dict | None): optional {player_id: display_name}.

    Returns:
        go.Figure
    """
    name_by_id = name_by_id or {}
    pids = sorted(win_rates)
    fig = go.Figure(go.Bar(
        x=[str(name_by_id.get(p, f"Player {p}")) for p in pids],
        y=[win_rates[p] for p in pids],
        marker_color=COLORS["curve_d"],
    ))
    apply_theme(fig,
        title="Hand win rate",
        xaxis_title="Player",
        yaxis_title="Win rate",
        yaxis_range=[0, 1],
    )
    return fig


def hand_label_figure(label_counts):
    """
    Bar chart of made-hand label distribution at showdown.

    Args:
        label_counts (dict): {hand_label: count}.

    Returns:
        go.Figure
    """
    labels = list(label_counts.keys())
    values = [label_counts[k] for k in labels]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=COLORS["curve_d"]))
    apply_theme(fig,
        title="Showdown hand-label distribution",
        xaxis_title="Hand",
        yaxis_title="Count",
    )
    return fig


def ev_accuracy_figure(ev_df):
    """
    Scatter of predicted equity vs realised showdown outcome, with y=x line.

    Args:
        ev_df (DataFrame): columns [predicted_equity, realised, ...].

    Returns:
        go.Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ev_df["predicted_equity"], y=ev_df["realised"],
        mode="markers", name="showdowns",
        marker=dict(opacity=0.5, color=COLORS["curve_d"]),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="calibrated (y=x)",
        line=dict(dash="dash", color=COLORS["neutral"]),
    ))
    apply_theme(fig,
        title="EV accuracy: predicted equity vs realised outcome",
        xaxis_title="Predicted equity",
        yaxis_title="Realised (1=win)",
    )
    return fig


def belief_trace_figure(trace_df, value="posterior_mean", name_by_id=None):
    """
    Line chart of a belief quantity (posterior_mean or p_tilted) over hands.

    Args:
        trace_df (DataFrame): columns [hand_number, player_id, posterior_mean,
            p_tilted].
        value (str): which column to plot.
        name_by_id (dict | None): optional {player_id: display_name}.

    Returns:
        go.Figure
    """
    name_by_id = name_by_id or {}
    fig = go.Figure()
    for pid in sorted(trace_df["player_id"].unique()):
        sub = trace_df[trace_df["player_id"] == pid].sort_values("hand_number")
        fig.add_trace(go.Scatter(
            x=sub["hand_number"], y=sub[value],
            mode="lines", name=str(name_by_id.get(pid, f"Player {pid}")),
        ))
    apply_theme(fig,
        title=f"Opponent belief: {value} over hands",
        xaxis_title="Hand",
        yaxis_title=value,
        yaxis_range=[0, 1],
    )
    return fig


def roi_leaderboard_figure(summary, name_by_id=None):
    """
    Bar chart leaderboard of chip EV per hand.

    Args:
        summary (dict): session_summary output {player_id: {...}}.
        name_by_id (dict | None): optional {player_id: display_name}.

    Returns:
        go.Figure
    """
    name_by_id = name_by_id or {}
    ordered = sorted(summary.items(),
                     key=lambda kv: kv[1]["chip_ev_per_hand"], reverse=True)
    vals = [row["chip_ev_per_hand"] for _pid, row in ordered]
    colors = [COLORS["pos"] if v >= 0 else COLORS["neg"] for v in vals]
    fig = go.Figure(go.Bar(
        x=[str(name_by_id.get(pid, row.get("name", f"Player {pid}")))
           for pid, row in ordered],
        y=vals,
        marker_color=colors,
    ))
    apply_theme(fig,
        title="ROI leaderboard (chip EV per hand)",
        xaxis_title="Player",
        yaxis_title="Chips / hand",
    )
    return fig


# ---------------------------------------------------------------------------
# Tournament / pool figures
# ---------------------------------------------------------------------------

def tournament_leaderboard_figure(leaderboard):
    """
    Horizontal bar chart of mean net chips per agent with 95% CI error bars
    when available. Best agent at top. Bars are green (positive) or red (negative).

    Args:
        leaderboard (list[dict]): each entry has at minimum {name,
            mean_net_chips}; optionally {ci95: {lo, hi}}.

    Returns:
        go.Figure
    """
    ordered = sorted(leaderboard, key=lambda d: d["mean_net_chips"])
    means = [d["mean_net_chips"] for d in ordered]
    names = [d["name"] for d in ordered]
    colors = [COLORS["pos"] if m >= 0 else COLORS["neg"] for m in means]

    # Build error bars from ci95 when present; fall back to None arrays.
    has_ci = any(d.get("ci95") for d in ordered)
    error_plus = [d["ci95"]["hi"] - d["mean_net_chips"]
                  if d.get("ci95") else 0 for d in ordered]
    error_minus = [d["mean_net_chips"] - d["ci95"]["lo"]
                   if d.get("ci95") else 0 for d in ordered]

    fig = go.Figure(go.Bar(
        x=means,
        y=names,
        orientation="h",
        marker_color=colors,
        error_x=dict(
            type="data",
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            thickness=2,
            width=7,
            color="#555",
            visible=has_ci,
        ),
    ))
    fig.add_vline(x=0, line=dict(color=COLORS["zero_line"], dash="dot", width=1))
    apply_theme(fig,
        title="Agent tournament leaderboard (mean net chips, 95% CI)",
        xaxis_title="Mean net chips",
        yaxis_title="",
        showlegend=False,
    )
    return fig


def tournament_matrix_figure(win_matrix, n_seeds):
    """
    Annotated heatmap of head-to-head win rates (row agent beats column agent).

    Args:
        win_matrix (dict[str, dict[str, int]]): win_matrix[A][B] = seeds A beat B.
        n_seeds (int): total seeds per pair (denominator for win-rate).

    Returns:
        go.Figure
    """
    names = sorted(win_matrix.keys())
    denom = max(1, n_seeds)
    z, text = [], []
    for row_name in names:
        row_z, row_text = [], []
        for col_name in names:
            if row_name == col_name:
                row_z.append(None)
                row_text.append("")
            else:
                wins = win_matrix[row_name].get(col_name, 0)
                rate = wins / denom
                row_z.append(rate)
                row_text.append(f"{rate:.2f}")
        z.append(row_z)
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z, x=names, y=names,
        text=text, texttemplate="%{text}",
        colorscale=COLORS["heatmap_scale"],
        zmid=0.5,
        colorbar=dict(title="Win rate", thickness=14),
    ))
    apply_theme(fig,
        title="Head-to-head win rates (row beats column)",
        xaxis_title="Opponent (column)",
        yaxis_title="Agent (row)",
    )
    return fig


def pool_strip_figure(per_agent_nets):
    """
    Jittered strip plot + mean bar of per-match net chips per agent.

    Each dot is one held-out match outcome. Deterministic jitter (no RNG) keeps
    the figure byte-reproducible. Replaces the inline go.Figure() block that was
    previously inside fig_pool in make_figures.py.

    Rationale for a strip over a box plot: per-match outcomes cluster hard at
    ±1000 (most matches are won/lost near-outright), so box quartiles span the
    full range and every agent looks identical. The strip shows real spread and
    the win-rate ranking honestly.

    Args:
        per_agent_nets (dict): {agent_name: [net chips per match]}.

    Returns:
        go.Figure
    """
    JITTER_WIDTH = 0.45
    JITTER_MODULUS = 11
    JITTER_STEPS = 5
    JITTER_OFFSET = 2

    order = sorted(
        per_agent_nets,
        key=lambda n: (sum(per_agent_nets[n]) / len(per_agent_nets[n])
                       if per_agent_nets[n] else 0.0),
        reverse=True,
    )

    # One color per agent position so dots and mean bars share the same hue.
    palette = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c", "#e74c3c"]

    fig = go.Figure()
    for i, name in enumerate(order):
        ys = per_agent_nets[name]
        color = palette[i % len(palette)]

        # Deterministic jitter: spread dots along x using a fixed arithmetic
        # sequence mod JITTER_MODULUS so the layout is the same on every render.
        xs = [
            i + (((j * JITTER_STEPS + JITTER_OFFSET) % JITTER_MODULUS)
                 / (JITTER_MODULUS - 1) - 0.5) * JITTER_WIDTH
            for j in range(len(ys))
        ]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=name,
            marker=dict(size=5, opacity=0.50, color=color),
            showlegend=False,
        ))

        # Mean bar: solid horizontal line spanning ±0.28 around the agent index.
        mean = sum(ys) / len(ys)
        win_rate = sum(1 for v in ys if v > 0) / len(ys)
        fig.add_trace(go.Scatter(
            x=[i - 0.28, i + 0.28], y=[mean, mean], mode="lines",
            line=dict(color=color, width=3),
            name=f"{name} mean",
            showlegend=False,
        ))

        # Invisible scatter just for the legend entry with win-rate annotation.
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=9, color=color),
            name=f"{name} ({win_rate:.0%} won)",
        ))

    fig.add_hline(y=0, line=dict(color=COLORS["zero_line"], dash="dot", width=1))
    apply_theme(fig,
        title="Per-match net chips by agent (dot = match, bar = mean)",
        yaxis_title="Net chips (per match)",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(order))),
            ticktext=order,
            gridcolor="rgba(0,0,0,0)",  # no vertical gridlines on categorical axis
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#dde1e4",
            borderwidth=1,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Evaluation / PnL panels
# ---------------------------------------------------------------------------

def pnl_distribution_figure(diffs, name_a="A", name_b="B"):
    """
    Histogram of per-seed PnL diffs (net_chips A − net_chips B), mean marked.

    Args:
        diffs (list[float]): per-seed chip differences.
        name_a (str): label for agent A.
        name_b (str): label for agent B.

    Returns:
        go.Figure
    """
    mean = (sum(diffs) / len(diffs)) if diffs else 0.0
    fig = go.Figure(go.Histogram(
        x=diffs, nbinsx=20,
        marker_color=COLORS["curve_d"],
        marker_line=dict(width=0.5, color="#fff"),
    ))
    fig.add_vline(x=0, line=dict(color=COLORS["zero_line"], dash="dot"))
    fig.add_vline(
        x=mean,
        line=dict(color=COLORS["pos"] if mean >= 0 else COLORS["neg"]),
        annotation_text=f"mean {mean:+.0f}",
        annotation_font_size=11,
    )
    apply_theme(fig,
        title=f"PnL distribution: {name_a} − {name_b} (per seed)",
        xaxis_title=f"Chip PnL ({name_a} − {name_b})",
        yaxis_title="Seeds",
        showlegend=False,
    )
    return fig


def paired_diff_figure(diffs, name_a="A", name_b="B", seeds=None):
    """
    Per-seed PnL diff as a green (win) / red (loss) bar chart (paired view).

    Args:
        diffs (list[float]): per-seed chip differences.
        name_a (str): label for agent A.
        name_b (str): label for agent B.
        seeds (list | None): seed labels for x-axis.

    Returns:
        go.Figure
    """
    x = [str(s) for s in (seeds if seeds is not None else range(len(diffs)))]
    colors = [
        COLORS["pos"] if d > 0 else COLORS["neg"] if d < 0 else COLORS["neutral"]
        for d in diffs
    ]
    fig = go.Figure(go.Bar(x=x, y=diffs, marker_color=colors))
    fig.add_hline(y=0, line=dict(color=COLORS["zero_line"]))
    apply_theme(fig,
        title=f"Per-seed PnL: {name_a} − {name_b}",
        xaxis_title="Seed",
        yaxis_title=f"Chip PnL ({name_a} − {name_b})",
    )
    return fig


def learning_curve_figure(history, ribbon=True):
    """
    Dual-axis RL learning curve: win rate (left) and mean chip diff vs
    baseline (right) over training steps.

    When `ribbon` is True and snapshots carry `per_seed_diffs`, a ±1 SEM
    band is shaded around the mean chip-diff line.

    Args:
        history (list[dict]): snapshots with keys {step, wins, n_seeds,
            mean_chip_diff} and optionally {per_seed_diffs}.
        ribbon (bool): whether to draw the SEM ribbon.

    Returns:
        go.Figure
    """
    steps = [h["step"] for h in history]
    win_rate = [h["wins"] / h["n_seeds"] if h.get("n_seeds") else 0.0
                for h in history]
    mean_diff = [h["mean_chip_diff"] for h in history]

    fig = go.Figure()

    if ribbon and any(h.get("per_seed_diffs") for h in history):
        xs, upper, lower = [], [], []
        for h in history:
            diffs = h.get("per_seed_diffs")
            if not diffs or len(diffs) < 2:
                continue
            n = len(diffs)
            m = sum(diffs) / n
            var = sum((d - m) ** 2 for d in diffs) / (n - 1)
            sem = (var ** 0.5) / (n ** 0.5)
            xs.append(h["step"])
            upper.append(m + sem)
            lower.append(m - sem)
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1],
            y=upper + lower[::-1],
            yaxis="y2",
            fill="toself",
            fillcolor=COLORS["ribbon"],
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=steps, y=win_rate, mode="lines+markers",
        name="win rate vs baseline",
        line=dict(color=COLORS["curve_d"], width=2),
        marker=dict(size=5),
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=mean_diff, mode="lines+markers",
        name="mean chip diff",
        line=dict(color=COLORS["curve_a"], width=2),
        marker=dict(size=5),
        yaxis="y2",
    ))

    apply_theme(fig,
        title="RL learning curve (held-out eval during training)",
        xaxis_title="Training step",
        yaxis=dict(title="Win rate", range=[0, 1], side="left"),
        yaxis2=dict(
            title="Mean chip diff",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=72, b=110, l=90, r=90),  # extra right for y2 title
    )
    return fig


def equity_drawdown_figure(dd_df, name=None):
    """
    Bankroll trajectory + underwater drawdown for one player.

    Args:
        dd_df (DataFrame): analytics.drawdown_curve output
            [hand_number, stack, peak, drawdown].
        name (str | None): player display name.

    Returns:
        go.Figure
    """
    label = name or "Player"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd_df["hand_number"], y=dd_df["stack"],
        mode="lines", name=f"{label} stack",
        line=dict(color=COLORS["curve_d"]),
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=dd_df["hand_number"], y=[-d for d in dd_df["drawdown"]],
        mode="lines", name="drawdown",
        line=dict(color=COLORS["neg"]),
        fill="tozeroy",
        fillcolor=f"rgba(231,76,60,0.15)",
        yaxis="y2",
    ))
    apply_theme(fig,
        title=f"Bankroll & drawdown: {label}",
        xaxis_title="Hand",
        yaxis=dict(title="Chips"),
        yaxis2=dict(
            title="Drawdown",
            overlaying="y",
            side="right",
            rangemode="tozero",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(t=72, b=110, l=90, r=90),
    )
    return fig


def pnl_box_figure(per_agent_nets):
    """
    Box plot of per-seed net chips per agent (cross-agent PnL distribution).

    Args:
        per_agent_nets (dict): {name: [net chips per (pair, seed)]}.

    Returns:
        go.Figure
    """
    order = sorted(
        per_agent_nets,
        key=lambda n: (sum(per_agent_nets[n]) / len(per_agent_nets[n])
                       if per_agent_nets[n] else 0.0),
        reverse=True,
    )
    palette = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c", "#e74c3c"]
    fig = go.Figure()
    for i, name in enumerate(order):
        fig.add_trace(go.Box(
            y=per_agent_nets[name],
            name=name,
            boxmean=True,
            marker_color=palette[i % len(palette)],
        ))
    fig.add_hline(y=0, line=dict(color=COLORS["zero_line"], dash="dot"))
    apply_theme(fig,
        title="Per-seed net-chip distribution by agent",
        yaxis_title="Net chips (per match)",
        showlegend=False,
    )
    return fig


def parameter_heatmap_figure(grid, value="mean_net_chips"):
    """
    Heatmap of a swept metric over the (tight_threshold × aggression) grid.
    Blue = net winner, red = net loser, white ≈ break-even.

    Args:
        grid (list[dict]): cells with keys 'tight', 'aggr', and `value`.
        value (str): metric key to colour by.

    Returns:
        go.Figure
    """
    tights = sorted({g["tight"] for g in grid})
    aggrs = sorted({g["aggr"] for g in grid})
    lookup = {(g["tight"], g["aggr"]): g[value] for g in grid}
    z = [[lookup.get((t, a)) for a in aggrs] for t in tights]
    text = [[(f"{lookup[(t, a)]:+.0f}" if (t, a) in lookup else "")
             for a in aggrs] for t in tights]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"{a:.2f}" for a in aggrs],
        y=[f"{t:.2f}" for t in tights],
        text=text,
        texttemplate="%{text}",
        colorscale=COLORS["heatmap_scale"],
        zmid=0,
        colorbar=dict(title="Mean net chips", thickness=14),
    ))
    apply_theme(fig,
        title="Personality fitness landscape (mean net chips per match)",
        xaxis_title="aggression",
        yaxis_title="tight_threshold",
    )
    return fig


# ---------------------------------------------------------------------------
# A/B measurement panels (Block B / A5 results JSONs under results/)
# ---------------------------------------------------------------------------

def ab_grouped_bar_figure(rows, group_key, value_key, by_key=None,
                          title=None, yaxis_title=None, group_order=None):
    """
    Grouped bar chart of an A/B metric. x = distinct group_key values;
    one bar series per distinct by_key value (single series if by_key is None).
    Rows sharing a (group, series) cell are averaged.

    Args:
        rows (list[dict]): measurement rows.
        group_key (str): x-axis category key.
        value_key (str): metric to plot (averaged per cell).
        by_key (str | None): series key, or None for a single series.
        group_order (list | None): explicit x-axis order.
        title (str | None): figure title.
        yaxis_title (str | None): y-axis label.

    Returns:
        go.Figure
    """
    present = {r[group_key] for r in rows}
    groups = ([g for g in group_order if g in present] if group_order
              else sorted(present, key=str))
    series = (sorted({r[by_key] for r in rows}, key=str) if by_key else [None])
    palette = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    fig = go.Figure()
    for idx, s in enumerate(series):
        ys = []
        for g in groups:
            vals = [r[value_key] for r in rows
                    if r[group_key] == g and (by_key is None or r[by_key] == s)]
            ys.append(sum(vals) / len(vals) if vals else None)
        fig.add_trace(go.Bar(
            name=str(s) if s is not None else value_key,
            x=[str(g) for g in groups],
            y=ys,
            marker_color=palette[idx % len(palette)],
        ))
    apply_theme(fig,
        title=title or f"{value_key} by {group_key}",
        xaxis_title=group_key,
        yaxis_title=yaxis_title or value_key,
        barmode="group",
    )
    if by_key is None:
        fig.update_layout(showlegend=False)
    return fig


def ab_heatmap_figure(rows, row_key, col_key, value_key, title=None,
                      colorbar_title="Mean net chips", zmid=0):
    """
    Annotated heatmap of an A/B metric over (row_key × col_key). Cells sharing
    a (row, col) are averaged. Blue = positive, red = negative, white ≈ zmid.

    Args:
        rows (list[dict]): rows with row_key, col_key, value_key fields.
        row_key (str): key for heatmap rows.
        col_key (str): key for heatmap columns.
        value_key (str): metric to aggregate.
        title (str | None): figure title.
        colorbar_title (str): colorbar label.
        zmid (float): center value for the diverging colorscale.

    Returns:
        go.Figure
    """
    row_vals = sorted({r[row_key] for r in rows}, key=str)
    col_vals = sorted({r[col_key] for r in rows}, key=str)
    cell = {}
    for rv in row_vals:
        for cv in col_vals:
            vals = [r[value_key] for r in rows
                    if r[row_key] == rv and r[col_key] == cv]
            cell[(rv, cv)] = (sum(vals) / len(vals)) if vals else None
    z = [[cell[(rv, cv)] for cv in col_vals] for rv in row_vals]
    text = [[(f"{cell[(rv, cv)]:+.0f}" if cell[(rv, cv)] is not None else "")
             for cv in col_vals] for rv in row_vals]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(c) for c in col_vals],
        y=[str(r) for r in row_vals],
        text=text,
        texttemplate="%{text}",
        colorscale=COLORS["heatmap_scale"],
        zmid=zmid,
        colorbar=dict(title=colorbar_title, thickness=14),
    ))
    apply_theme(fig,
        title=title or f"{value_key} by {row_key} × {col_key}",
        xaxis_title=col_key,
        yaxis_title=row_key,
    )
    return fig


def forest_plot_figure(rows, title=None,
                       xaxis_title="Mean net chips (95% bootstrap CI)"):
    """
    Forest plot of estimated effects with 95% confidence intervals.

    Each row is a point at the mean with whiskers to [lo, hi]; a dotted zero
    line marks the null. Green = CI excludes zero above (real positive edge),
    red = excludes zero below, gray = CI straddles zero (within noise).

    Rendered as a SINGLE trace using arrays rather than one trace per row so
    that hover and legend stay clean regardless of row count. Long y-axis
    labels are wrapped at word boundaries with <br>.

    Args:
        rows (list[dict]): each has keys 'label', 'mean', 'lo', 'hi'.
        title (str | None): figure title.
        xaxis_title (str): x-axis label.

    Returns:
        go.Figure
    """
    def _color(r):
        if r["lo"] > 0:
            return COLORS["pos"]
        if r["hi"] < 0:
            return COLORS["neg"]
        return COLORS["neutral"]

    # One trace per row so each gets its own error_x color (Plotly does not
    # support per-point error_x.color on a single trace). hovertemplate and
    # showlegend=False keep hover and legend clean regardless of row count.
    fig = go.Figure()
    # Reverse so the first row in the list appears at the top of the y-axis.
    for r in reversed(rows):
        c = _color(r)
        label = wrap_label(r["label"])
        fig.add_trace(go.Scatter(
            x=[r["mean"]],
            y=[label],
            mode="markers",
            marker=dict(color=c, size=10, symbol="circle"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[r["hi"] - r["mean"]],
                arrayminus=[r["mean"] - r["lo"]],
                color=c,
                thickness=2,
                width=7,
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "mean: %{x:+.1f}<br>"
                "<extra></extra>"
            ),
        ))
    fig.add_vline(x=0, line=dict(color=COLORS["zero_line"], dash="dot", width=1))
    # THEME contains a top-level `yaxis` key; override it explicitly after
    # spreading THEME so the forest plot's autorange=True takes precedence.
    theme_no_yaxis = {k: v for k, v in THEME.items() if k != "yaxis"}
    apply_theme(fig,
        title=title or "Effect sizes with 95% bootstrap CIs",
        xaxis_title=xaxis_title,
        yaxis=dict(
            title="",
            autorange=True,
            gridcolor=THEME["yaxis"]["gridcolor"],
            linecolor=THEME["yaxis"]["linecolor"],
            tickfont=THEME["yaxis"]["tickfont"],
        ),
        margin=dict(t=72, b=80, l=280, r=60),  # wide left for wrapped labels
    )
    return fig


def exploitability_curve_figure(rows, uniform=None, title=None, q_rows=None,
                                nfsp_rows=None):
    """
    Exploitability (NashConv) over self-play training, log-log scale.

    Traces:
        - CFR time-average   → converges to Nash (green)
        - CFR last-iterate   → stays exploitable (orange)
        - Q-learning last-iterate (if q_rows) → non-convergence (red)
        - NFSP average-policy (if nfsp_rows) → learned averaging fix (blue)

    Args:
        rows (list[dict]): each {iters, avg_exploitability,
            last_iterate_exploitability}.
        uniform (float | None): uniform-random baseline to mark.
        title (str | None): figure title.
        q_rows (list[dict] | None): each {episodes, exploitability}.
        nfsp_rows (list[dict] | None): each {episodes, exploitability}.

    Returns:
        go.Figure
    """
    iters = [r["iters"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters,
        y=[r["avg_exploitability"] for r in rows],
        mode="lines+markers",
        name="CFR time-average (→ Nash)",
        line=dict(color=COLORS["curve_a"], width=2),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=iters,
        y=[r["last_iterate_exploitability"] for r in rows],
        mode="lines+markers",
        name="CFR last-iterate (greedy, stays exploitable)",
        line=dict(color=COLORS["curve_b"], width=2),
        marker=dict(size=5),
    ))
    if q_rows:
        fig.add_trace(go.Scatter(
            x=[r["episodes"] for r in q_rows],
            y=[r["exploitability"] for r in q_rows],
            mode="lines+markers",
            name="Q-learning self-play last-iterate (DQN regime)",
            line=dict(color=COLORS["curve_c"], width=2, dash="dot"),
            marker=dict(size=5),
        ))
    if nfsp_rows:
        fig.add_trace(go.Scatter(
            x=[r["episodes"] for r in nfsp_rows],
            y=[r["exploitability"] for r in nfsp_rows],
            mode="lines+markers",
            name="NFSP average-policy (→ Nash)",
            line=dict(color=COLORS["curve_d"], width=2),
            marker=dict(size=5),
        ))
    if uniform is not None:
        fig.add_hline(
            y=uniform,
            line=dict(color=COLORS["zero_line"], dash="dot"),
            annotation_text=f"uniform random ({uniform:.2f})",
            annotation_font_size=11,
        )
    apply_theme(fig,
        title=title or "Leduc exploitability: averaging converges, greedy doesn't",
        xaxis_title="self-play training (CFR iterations / Q-learning episodes)",
        xaxis_type="log",
        yaxis_title="Exploitability (NashConv; 0 = exact Nash)",
        yaxis_type="log",
        legend=dict(
            yanchor="top", y=0.98,
            xanchor="left", x=0.02,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#dde1e4",
            borderwidth=1,
        ),
    )
    return fig


def icm_edge_figure(rows, title=None):
    """
    Per-init-seed ICM-minus-chips prize edge bar chart.

    Green = ICM reward earned more tournament prize than chip reward.
    Red = less. Text labels switch inside/outside automatically based on bar
    direction to avoid clipping on negative bars.

    Args:
        rows (list[dict]): each has {init_seed, icm_minus_chips}.
        title (str | None): figure title.

    Returns:
        go.Figure
    """
    ordered = sorted(rows, key=lambda r: r["init_seed"])
    y = [r["icm_minus_chips"] for r in ordered]
    colors = [
        COLORS["pos"] if v > 0 else COLORS["neg"] if v < 0 else COLORS["neutral"]
        for v in y
    ]
    # 'auto' lets Plotly choose inside vs outside per bar based on bar direction,
    # preventing negative-bar labels from disappearing below the axis.
    text_positions = ["outside" if v >= 0 else "inside" for v in y]
    mean = sum(y) / len(y) if y else 0.0

    fig = go.Figure()
    for i, (r, color, pos) in enumerate(zip(ordered, colors, text_positions)):
        v = r["icm_minus_chips"]
        fig.add_trace(go.Bar(
            x=[f"seed {r['init_seed']}"],
            y=[v],
            marker_color=color,
            text=[f"{v:+.0f}"],
            textposition=pos,
            showlegend=False,
            textfont=dict(size=11),
        ))
    fig.add_hline(y=0, line=dict(color=COLORS["zero_line"], width=1))
    apply_theme(fig,
        title=title or f"ICM − chips prize edge (mean {mean:+.0f})",
        xaxis_title="init seed",
        yaxis_title="ICM − chips mean prize",
        showlegend=False,
        barmode="relative",
    )
    return fig
