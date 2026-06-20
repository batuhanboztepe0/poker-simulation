"""
charts.py
---------
Pure Plotly figure factories for the dashboard (Phase 2).

Every function takes plain data (DataFrames / dicts produced by
src.analytics) and returns a plotly.graph_objects.Figure. No Streamlit, no
I/O, no global state — so each factory is unit-testable on a fixture.
"""

import plotly.graph_objects as go


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
    fig.update_layout(
        title="Stack trajectory (equity curve)",
        xaxis_title="Hand", yaxis_title="Chips",
    )
    return fig


def win_rate_figure(win_rates, name_by_id=None):
    """
    Bar chart of per-player hand win rate.

    Args:
        win_rates (dict): {player_id: win_rate in [0, 1]}.
        name_by_id (dict | None): optional {player_id: display_name}.
    """
    name_by_id = name_by_id or {}
    pids = sorted(win_rates)
    fig = go.Figure(go.Bar(
        x=[str(name_by_id.get(p, f"Player {p}")) for p in pids],
        y=[win_rates[p] for p in pids],
    ))
    fig.update_layout(
        title="Hand win rate", xaxis_title="Player",
        yaxis_title="Win rate", yaxis=dict(range=[0, 1]),
    )
    return fig


def hand_label_figure(label_counts):
    """
    Bar chart of made-hand label distribution at showdown.

    Args:
        label_counts (dict): {hand_label: count}.
    """
    labels = list(label_counts.keys())
    values = [label_counts[k] for k in labels]
    fig = go.Figure(go.Bar(x=labels, y=values))
    fig.update_layout(
        title="Showdown hand-label distribution",
        xaxis_title="Hand", yaxis_title="Count",
    )
    return fig


def ev_accuracy_figure(ev_df):
    """
    Scatter of predicted equity vs realised showdown outcome, with y=x.

    Args:
        ev_df (DataFrame): columns [predicted_equity, realised, ...].
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ev_df["predicted_equity"], y=ev_df["realised"],
        mode="markers", name="showdowns",
        marker=dict(opacity=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="calibrated (y=x)",
        line=dict(dash="dash"),
    ))
    fig.update_layout(
        title="EV accuracy: predicted equity vs realised outcome",
        xaxis_title="Predicted equity", yaxis_title="Realised (1=win)",
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
    """
    name_by_id = name_by_id or {}
    fig = go.Figure()
    for pid in sorted(trace_df["player_id"].unique()):
        sub = trace_df[trace_df["player_id"] == pid].sort_values("hand_number")
        fig.add_trace(go.Scatter(
            x=sub["hand_number"], y=sub[value],
            mode="lines", name=str(name_by_id.get(pid, f"Player {pid}")),
        ))
    fig.update_layout(
        title=f"Opponent belief: {value} over hands",
        xaxis_title="Hand", yaxis_title=value, yaxis=dict(range=[0, 1]),
    )
    return fig


def roi_leaderboard_figure(summary, name_by_id=None):
    """
    Bar chart leaderboard of chip EV per hand.

    Args:
        summary (dict): session_summary output {player_id: {...}}.
        name_by_id (dict | None): optional {player_id: display_name}.
    """
    name_by_id = name_by_id or {}
    ordered = sorted(summary.items(),
                     key=lambda kv: kv[1]["chip_ev_per_hand"], reverse=True)
    fig = go.Figure(go.Bar(
        x=[str(name_by_id.get(pid, row.get("name", f"Player {pid}")))
           for pid, row in ordered],
        y=[row["chip_ev_per_hand"] for _pid, row in ordered],
    ))
    fig.update_layout(
        title="ROI leaderboard (chip EV per hand)",
        xaxis_title="Player", yaxis_title="Chips / hand",
    )
    return fig


def tournament_leaderboard_figure(leaderboard):
    """
    Horizontal bar chart of mean net chips per agent, best agent at top.

    Args:
        leaderboard (list[dict]): each entry has {name, mean_net_chips, ...}.

    Returns:
        go.Figure
    """
    # Sort ascending by value so the best agent ends up at the top of the
    # horizontal bar chart (plotly renders bars bottom-to-top).
    ordered = sorted(leaderboard, key=lambda d: d["mean_net_chips"])
    fig = go.Figure(go.Bar(
        x=[d["mean_net_chips"] for d in ordered],
        y=[d["name"] for d in ordered],
        orientation="h",
    ))
    fig.update_layout(
        title="Agent tournament leaderboard (mean net chips)",
        xaxis_title="Mean net chips",
        yaxis_title="Agent",
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
    z = []
    text = []
    for row_name in names:
        row_z = []
        row_text = []
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
        z=z,
        x=names,
        y=names,
        text=text,
        texttemplate="%{text}",
        colorscale="RdBu",
        zmid=0.5,
        colorbar=dict(title="Win rate"),
    ))
    fig.update_layout(
        title="Head-to-head win rates (row beats column)",
        xaxis_title="Opponent (column)",
        yaxis_title="Agent (row)",
    )
    return fig


# ---------------------------------------------------------------------------
# Evaluation / PnL panels (src.evaluation, src.analytics drawdown)
# ---------------------------------------------------------------------------

def pnl_distribution_figure(diffs, name_a="A", name_b="B"):
    """
    Histogram of per-seed PnL diffs (net_chips A − net_chips B), mean marked.
    Each seed is one MC-driven match, so this is the realised PnL distribution
    of A vs B across scenarios.
    """
    mean = (sum(diffs) / len(diffs)) if diffs else 0.0
    fig = go.Figure(go.Histogram(x=diffs, nbinsx=20))
    fig.add_vline(x=0, line=dict(color="gray", dash="dot"))
    fig.add_vline(x=mean, line=dict(color="green" if mean >= 0 else "red"),
                  annotation_text=f"mean {mean:+.0f}")
    fig.update_layout(
        title=f"PnL distribution: {name_a} − {name_b} (per seed)",
        xaxis_title=f"Chip PnL ({name_a} − {name_b})", yaxis_title="Seeds",
        showlegend=False,
    )
    return fig


def paired_diff_figure(diffs, name_a="A", name_b="B", seeds=None):
    """Per-seed PnL diff as a green(win)/red(loss) bar chart (paired view)."""
    x = [str(s) for s in (seeds if seeds is not None else range(len(diffs)))]
    colors = ["#2ca02c" if d > 0 else "#d62728" if d < 0 else "#888"
              for d in diffs]
    fig = go.Figure(go.Bar(x=x, y=diffs, marker_color=colors))
    fig.add_hline(y=0, line=dict(color="gray"))
    fig.update_layout(
        title=f"Per-seed PnL: {name_a} − {name_b}",
        xaxis_title="Seed", yaxis_title=f"Chip PnL ({name_a} − {name_b})",
    )
    return fig


def learning_curve_figure(history, ribbon=True):
    """
    Dual-axis RL learning curve from `SelfPlayTrainer.history` snapshots (each
    {step, wins, n_seeds, mean_chip_diff[, per_seed_diffs]}): win rate (left) and
    mean chip diff vs baseline (right) over training steps.

    When `ribbon` and the snapshots carry `per_seed_diffs`, a ±1 SEM band of the
    per-seed chip diff is shaded around the mean-chip-diff line (the eval-seed
    spread at each step) — the confidence ribbon for the headline figure.
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
            if not diffs:
                continue
            n = len(diffs)
            if n < 2:
                continue
            mean = sum(diffs) / n
            var = sum((d - mean) ** 2 for d in diffs) / (n - 1)  # sample variance
            sem = (var ** 0.5) / (n ** 0.5)
            xs.append(h["step"])
            upper.append(mean + sem)
            lower.append(mean - sem)
        # Closed band (forward upper, reversed lower) on the chip-diff axis.
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1], y=upper + lower[::-1], yaxis="y2",
            fill="toself", fillcolor="rgba(44,160,44,0.15)",
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(x=steps, y=win_rate, mode="lines+markers",
                             name="win rate vs baseline", yaxis="y"))
    fig.add_trace(go.Scatter(x=steps, y=mean_diff, mode="lines+markers",
                             name="mean chip diff", yaxis="y2"))
    fig.update_layout(
        title="RL learning curve (held-out eval during training)",
        xaxis_title="Training step",
        yaxis=dict(title="Win rate", range=[0, 1]),
        yaxis2=dict(title="Mean chip diff", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    return fig


def equity_drawdown_figure(dd_df, name=None):
    """
    Bankroll trajectory + underwater drawdown for one player.

    Args:
        dd_df (DataFrame): `analytics.drawdown_curve` output
            [hand_number, stack, peak, drawdown].
        name (str | None): player display name.
    """
    label = name or "Player"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd_df["hand_number"], y=dd_df["stack"],
        mode="lines", name=f"{label} stack", yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=dd_df["hand_number"], y=[-d for d in dd_df["drawdown"]],
        mode="lines", name="drawdown", yaxis="y2",
        fill="tozeroy", line=dict(color="#d62728"),
    ))
    fig.update_layout(
        title=f"Bankroll & drawdown: {label}",
        xaxis_title="Hand",
        yaxis=dict(title="Chips"),
        yaxis2=dict(title="Drawdown", overlaying="y", side="right",
                    rangemode="tozero"),
        legend=dict(orientation="h"),
    )
    return fig


def pnl_box_figure(per_agent_nets):
    """
    Box plot of per-seed net chips per agent (cross-agent PnL distribution).

    Args:
        per_agent_nets (dict): {name: [net chips per (pair, seed)]}.
    """
    order = sorted(
        per_agent_nets,
        key=lambda n: (sum(per_agent_nets[n]) / len(per_agent_nets[n])
                       if per_agent_nets[n] else 0.0),
        reverse=True,
    )
    fig = go.Figure()
    for name in order:
        fig.add_trace(go.Box(y=per_agent_nets[name], name=name, boxmean=True))
    fig.add_hline(y=0, line=dict(color="gray", dash="dot"))
    fig.update_layout(
        title="Per-seed net-chip distribution by agent",
        yaxis_title="Net chips (per match)", showlegend=False,
    )
    return fig


def parameter_heatmap_figure(grid, value="mean_net_chips"):
    """
    Heatmap of a swept metric over the (tight_threshold × aggression) grid — the
    personality fitness landscape from `evaluation.parameter_sweep`. Blue = net
    winner, red = net loser, white ≈ break-even.

    Args:
        grid (list[dict]): cells with keys 'tight', 'aggr', and `value`.
        value (str): metric key to colour by (default mean net chips).
    """
    tights = sorted({g["tight"] for g in grid})
    aggrs = sorted({g["aggr"] for g in grid})
    lookup = {(g["tight"], g["aggr"]): g[value] for g in grid}
    z = [[lookup.get((t, a)) for a in aggrs] for t in tights]
    text = [[(f"{lookup[(t, a)]:+.0f}" if (t, a) in lookup else "")
             for a in aggrs] for t in tights]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{a:.2f}" for a in aggrs], y=[f"{t:.2f}" for t in tights],
        text=text, texttemplate="%{text}",
        colorscale="RdBu", zmid=0, colorbar=dict(title="Mean net chips"),
    ))
    fig.update_layout(
        title="Personality fitness landscape (mean net chips per match)",
        xaxis_title="aggression", yaxis_title="tight_threshold",
    )
    return fig


# ---------------------------------------------------------------------------
# A/B measurement panels (the Block B / A5 results JSONs under results/)
# ---------------------------------------------------------------------------

def ab_grouped_bar_figure(rows, group_key, value_key, by_key=None,
                          title=None, yaxis_title=None, group_order=None):
    """
    Grouped bar chart of an A/B metric. x = distinct `group_key` values; one bar
    series per distinct `by_key` value (a single series if `by_key` is None).
    Rows sharing a (group, series) cell are averaged. Drives the Block B A/B
    JSONs (measure_action_grid / bust_clip / selfplay / tilt_decouple) — e.g.
    group_key='init_seed', by_key='grid', value_key='mean'.

    Args:
        rows (list[dict]): measurement rows.
        group_key (str): x-axis category key.
        value_key (str): metric to plot (averaged per cell).
        by_key (str | None): series key, or None for a single series.
        group_order (list | None): explicit x-axis order for the groups (e.g. a
            monotone clip order 'old'/'4.6'/'wide'); defaults to a string sort.
    """
    present = {r[group_key] for r in rows}
    groups = ([g for g in group_order if g in present] if group_order
              else sorted(present, key=str))
    series = (sorted({r[by_key] for r in rows}, key=str) if by_key else [None])
    fig = go.Figure()
    for s in series:
        ys = []
        for g in groups:
            vals = [r[value_key] for r in rows
                    if r[group_key] == g and (by_key is None or r[by_key] == s)]
            ys.append(sum(vals) / len(vals) if vals else None)
        fig.add_trace(go.Bar(
            name=str(s) if s is not None else value_key,
            x=[str(g) for g in groups], y=ys))
    fig.update_layout(
        title=title or f"{value_key} by {group_key}",
        xaxis_title=group_key, yaxis_title=yaxis_title or value_key,
        barmode="group",
    )
    if by_key is None:
        fig.update_layout(showlegend=False)
    return fig


def ab_heatmap_figure(rows, row_key, col_key, value_key, title=None,
                      colorbar_title="Mean net chips", zmid=0):
    """
    Annotated heatmap of an A/B metric over (row_key × col_key); cells sharing a
    (row, col) are averaged. For the config×opponent panels, melt the
    per-config rows ({config, init_seed, myopic, tilt, random}) into
    {config, opponent, value} rows first. Blue = positive, red = negative,
    white ≈ zmid.
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
        z=z, x=[str(c) for c in col_vals], y=[str(r) for r in row_vals],
        text=text, texttemplate="%{text}",
        colorscale="RdBu", zmid=zmid, colorbar=dict(title=colorbar_title)))
    fig.update_layout(
        title=title or f"{value_key} by {row_key} × {col_key}",
        xaxis_title=col_key, yaxis_title=row_key)
    return fig


def icm_edge_figure(rows, title=None):
    """
    Per-init-seed ICM-minus-chips prize edge (measure_icm rows). Green = the
    concave ICM reward earned MORE tournament prize than the risk-neutral chip
    reward at this multi-prize table; red = less. The RL_HANDOFF §13 story is the
    SIGN consistency across seeds, not any single significant p-value.
    """
    ordered = sorted(rows, key=lambda r: r["init_seed"])
    y = [r["icm_minus_chips"] for r in ordered]
    colors = ["#2ca02c" if v > 0 else "#d62728" if v < 0 else "#888" for v in y]
    mean = sum(y) / len(y) if y else 0.0
    fig = go.Figure(go.Bar(
        x=[f"seed {r['init_seed']}" for r in ordered], y=y,
        marker_color=colors, text=[f"{v:+.0f}" for v in y],
        textposition="outside"))
    fig.add_hline(y=0, line=dict(color="gray"))
    fig.update_layout(
        title=title or f"ICM − chips prize edge (mean {mean:+.0f})",
        xaxis_title="init seed", yaxis_title="ICM − chips mean prize",
        showlegend=False,
    )
    return fig
