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


def learning_curve_figure(history):
    """
    Dual-axis RL learning curve from `SelfPlayTrainer.history` snapshots (each
    {step, wins, n_seeds, mean_chip_diff}): win rate (left) and mean chip diff
    vs baseline (right) over training steps.
    """
    steps = [h["step"] for h in history]
    win_rate = [h["wins"] / h["n_seeds"] if h.get("n_seeds") else 0.0
                for h in history]
    mean_diff = [h["mean_chip_diff"] for h in history]
    fig = go.Figure()
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
