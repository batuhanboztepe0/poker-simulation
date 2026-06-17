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
