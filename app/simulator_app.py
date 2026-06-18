"""
simulator_app.py
----------------
Streamlit dashboard for the poker simulator (Phase 2).

Run with:
    streamlit run app/simulator_app.py

Configure the table (hands, bots, per-bot tight/aggression, MC sims, seed) in
the sidebar, run a seeded simulation, and inspect every analytics panel. Runs
persist to data/session_<seed>.parquet and can be replayed from disk.

A fresh GameEngine is created per run (no engine sharing across runs).
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation_runner import run_session
from src.stats import session_summary
from src.analytics import (
    equity_curve, win_rate_by_player, hand_label_distribution, ev_accuracy,
    belief_trace_dataframe, load_parquet,
)
from app.charts import (
    equity_curve_figure, win_rate_figure, hand_label_figure,
    ev_accuracy_figure, roi_leaderboard_figure, belief_trace_figure,
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


def _sidebar_config():
    """Render sidebar controls and return the run configuration."""
    st.sidebar.header("Simulation settings")
    n_bots = st.sidebar.slider("Number of bots", 2, 9, 4)
    n_hands = st.sidebar.slider("Hands", 10, 5000, 500, step=10)
    seed = st.sidebar.number_input("Seed", value=42, step=1)
    fast_mode = st.sidebar.checkbox("Fast mode (no Monte Carlo)", value=False)
    mc_sims = st.sidebar.slider("MC simulations", 100, 2000, 200, step=100,
                                disabled=fast_mode)
    small_blind = st.sidebar.number_input("Small blind", value=10, step=5)
    big_blind = st.sidebar.number_input("Big blind", value=20, step=5)

    st.sidebar.header("Per-bot personality")
    configs = []
    for i in range(1, n_bots + 1):
        with st.sidebar.expander(f"Bot {i}", expanded=(i <= 2)):
            tight = st.slider(f"tight_threshold (Bot {i})", 0.0, 1.0, 0.4,
                              key=f"tight_{i}")
            aggr = st.slider(f"aggression (Bot {i})", 0.0, 1.0, 0.5,
                             key=f"aggr_{i}")
            configs.append({
                "name": f"Bot{i}", "tight_threshold": tight, "aggression": aggr,
            })
    return {
        "configs": configs, "n_hands": n_hands, "seed": int(seed),
        "fast_mode": fast_mode, "mc_simulations": mc_sims,
        "small_blind": int(small_blind), "big_blind": int(big_blind),
    }


def _render_panels(result, collector):
    """Render every analytics panel for a finished session."""
    name_by_id = {s.player_id: s.name for s in result.players}
    summary = session_summary(result)

    st.subheader("Equity curves")
    st.plotly_chart(
        equity_curve_figure(equity_curve(result), name_by_id),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Win rate")
        st.plotly_chart(
            win_rate_figure(win_rate_by_player(result), name_by_id),
            use_container_width=True,
        )
    with col2:
        st.subheader("ROI leaderboard")
        st.plotly_chart(
            roi_leaderboard_figure(summary, name_by_id),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Hand-label distribution")
        labels = hand_label_distribution(result)
        if labels:
            st.plotly_chart(hand_label_figure(labels), use_container_width=True)
        else:
            st.info("No showdowns in this run.")
    with col4:
        st.subheader("EV accuracy")
        ev_df = ev_accuracy(result)
        if not ev_df.empty:
            st.plotly_chart(ev_accuracy_figure(ev_df), use_container_width=True)
        else:
            st.info("No predicted-equity data (enable Monte Carlo).")

    # Phase B: opponent belief posterior over hands (only when bots model).
    belief_df = belief_trace_dataframe(result)
    if not belief_df.empty:
        st.subheader("Opponent belief (posterior mean over hands)")
        st.plotly_chart(
            belief_trace_figure(belief_df, value="posterior_mean", name_by_id=name_by_id),
            use_container_width=True,
        )

    st.subheader("Player summary")
    st.dataframe(summary)

    st.subheader("Event table")
    st.dataframe(collector.to_dataframe())


def main():
    st.set_page_config(page_title="Poker Simulator", layout="wide")
    st.title("Poker-Quant Simulator")

    page = st.sidebar.radio("Page", ["Run", "Replay", "Tournament"])

    if page == "Run":
        cfg = _sidebar_config()
        if st.sidebar.button("Run simulation", type="primary"):
            with st.spinner("Simulating..."):
                result, collector = run_session(
                    cfg["configs"], cfg["n_hands"], seed=cfg["seed"],
                    small_blind=cfg["small_blind"], big_blind=cfg["big_blind"],
                    mc_simulations=cfg["mc_simulations"],
                    fast_mode=cfg["fast_mode"],
                )
                os.makedirs(DATA_DIR, exist_ok=True)
                path = os.path.join(DATA_DIR, f"session_{cfg['seed']}.parquet")
                collector.save_parquet(path)
                st.success(f"Ran {result.n_hands} hands. Saved to {path}")
                _render_panels(result, collector)
        else:
            st.info("Configure the table in the sidebar and click "
                    "**Run simulation**.")

    elif page == "Replay":
        st.header("Replay a saved run")
        if not os.path.isdir(DATA_DIR):
            st.info("No saved runs yet.")
            return
        files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
        if not files:
            st.info("No saved runs yet.")
            return
        choice = st.selectbox("Saved session", files)
        df = load_parquet(os.path.join(DATA_DIR, choice))
        st.subheader("Event log")
        st.dataframe(df)

    elif page == "Tournament":
        st.header("Agent Tournament")
        st.sidebar.header("Tournament settings")
        t_n_hands = st.sidebar.slider("Hands per session", 10, 500, 100, step=10)
        t_seeds_raw = st.sidebar.text_input("Seeds (comma-separated)", value="0,1,2,3,4")
        t_fast_mode = st.sidebar.checkbox("Fast mode (no Monte Carlo)", value=True)
        if st.button("Run tournament", type="primary"):
            # Lazy imports: keep module top-level import-safe for headless tests.
            from src.tournament import run_tournament
            from app.charts import (
                tournament_leaderboard_figure,
                tournament_matrix_figure,
            )
            from src.player import BotPlayer
            from src.kelly_agent import KellyBotPlayer

            try:
                t_seeds = [int(s.strip()) for s in t_seeds_raw.split(",")
                           if s.strip()]
            except ValueError:
                st.error("Seeds must be comma-separated integers.")
                return

            roster = {
                "Myopic": lambda pid, stack: BotPlayer(pid, "Myopic", stack),
                "Kelly": lambda pid, stack: KellyBotPlayer(pid, "Kelly", stack),
            }
            try:
                import torch  # noqa: F401
                from src.rl_agent import RLBotPlayer, QNetwork

                def _rl_factory(pid, stack):
                    return RLBotPlayer(pid, "RL", stack)

                roster["RL"] = _rl_factory
            except ImportError:
                pass

            with st.spinner("Running tournament..."):
                t_result = run_tournament(
                    roster=roster,
                    seeds=t_seeds,
                    n_hands=t_n_hands,
                    fast_mode=t_fast_mode,
                )

            st.subheader("Leaderboard")
            st.plotly_chart(
                tournament_leaderboard_figure(t_result.leaderboard),
                use_container_width=True,
            )

            st.subheader("Head-to-head matrix")
            st.plotly_chart(
                tournament_matrix_figure(t_result.win_matrix, len(t_seeds)),
                use_container_width=True,
            )

            st.subheader("Notes")
            st.info(t_result.kuhn_cfr_note)
            for note in t_result.notes:
                st.warning(note)
        else:
            st.info("Configure the tournament in the sidebar and click "
                    "**Run tournament**.")


if __name__ == "__main__":
    main()
