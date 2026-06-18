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


def _evaluation_page():
    """Monte-Carlo PnL evaluation page: matchup, cross-agent, RL curve, drawdown."""
    st.header("Evaluation / PnL — Monte-Carlo")
    st.caption("Each seed is one MC-driven heads-up match; the per-seed chip "
               "difference is a realised PnL draw. Pick agents and a mode.")

    # Lazy imports keep the module import-safe for headless tests.
    from src.evaluation import evaluate_matchup, evaluate_roster, parameter_sweep
    from src.analytics import drawdown_curve, max_drawdown
    from src.simulation import simulate_session
    from src.player import BotPlayer
    from src.kelly_agent import KellyBotPlayer
    from src.stochastic_control import RolloutPolicy, RolloutBotPlayer
    from src.adaptive_agent import AdaptiveBotPlayer
    from src.monte_carlo import MonteCarloEngine
    from app.charts import (
        pnl_distribution_figure, paired_diff_figure, learning_curve_figure,
        equity_drawdown_figure, pnl_box_figure, tournament_leaderboard_figure,
        tournament_matrix_figure, parameter_heatmap_figure,
    )

    st.sidebar.header("Evaluation settings")
    mode = st.sidebar.radio("Mode", [
        "Matchup (PnL + t-test)", "Cross-agent leaderboard",
        "Parameter sweep", "RL learning curve", "Equity + drawdown",
    ])
    n_seeds = st.sidebar.slider("Seeds", 2, 60, 20)
    n_hands = st.sidebar.slider("Hands per match", 20, 500, 150, step=10)
    mc_sims = st.sidebar.slider("MC simulations", 100, 1000, 200, step=100)

    def mc():
        return MonteCarloEngine(n_simulations=mc_sims)

    factories = {
        "Myopic": lambda pid, stack: BotPlayer(
            pid, "Myopic", stack, tight_threshold=0.2, aggression=0.5,
            mc_engine=mc()),
        "Kelly": lambda pid, stack: KellyBotPlayer(
            pid, "Kelly", stack, kelly_scalar=0.5, mc_engine=mc()),
        "Rollout": lambda pid, stack: RolloutBotPlayer(
            pid, "Rollout", stack, rollout_policy=RolloutPolicy(mc())),
        "Adaptive(tilt)": lambda pid, stack: AdaptiveBotPlayer(
            pid, "Adaptive(tilt)", stack, mode="tilt", mc_engine=mc()),
        "Adaptive(random)": lambda pid, stack: AdaptiveBotPlayer(
            pid, "Adaptive(random)", stack, mode="random", mc_engine=mc()),
    }

    # Optional RL agent from a saved checkpoint in models/.
    rl_history = []
    models_dir = os.path.join(os.path.dirname(DATA_DIR), "models")
    try:
        import torch  # noqa: F401
        have_torch = True
    except ImportError:
        have_torch = False
    ckpts = (sorted(f for f in os.listdir(models_dir)
                    if f.endswith((".pt", ".pth")))
             if os.path.isdir(models_dir) else [])
    if not have_torch:
        st.sidebar.caption("torch not installed → RL agent unavailable.")
    elif not ckpts:
        st.sidebar.caption(
            "No RL checkpoints in models/. Train one with "
            "`python -m scripts.train_rl --save models/rl.pt --eval-every 200`.")
    else:
        sel = st.sidebar.selectbox("RL checkpoint (models/)", ["(none)"] + ckpts)
        if sel != "(none)":
            from src.rl_agent import load_checkpoint, RLBotPlayer
            qnet, ckpt = load_checkpoint(os.path.join(models_dir, sel))
            rl_history = ckpt.get("history", [])
            fm = ckpt.get("feature_mode", "base")
            if fm == "base":
                def _rl(pid, stack, _q=qnet):
                    return RLBotPlayer(pid, "RL", stack, qnet=_q, epsilon=0.0,
                                       training=False, mc_engine=mc())
                factories["RL"] = _rl
            else:
                st.sidebar.warning(
                    f"Checkpoint uses feature_mode='{fm}', which needs episodic "
                    "horizon/belief context — its learning curve is shown, but it "
                    "can't play the heads-up PnL evals here.")

    names = list(factories.keys())
    seeds = list(range(n_seeds))

    if mode == "Matchup (PnL + t-test)":
        c1, c2 = st.columns(2)
        a = c1.selectbox("Agent A (seat 1)", names, index=0)
        b = c2.selectbox("Agent B (seat 2)", names,
                         index=min(1, len(names) - 1))
        if a == b:
            st.warning("Pick two different agents.")
        elif st.button("Run matchup", type="primary"):
            with st.spinner(f"Playing {n_seeds} seeded matches..."):
                mr = evaluate_matchup(factories[a], factories[b], a, b, seeds,
                                      n_hands=n_hands)
            tt = mr.t_test
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{a} wins", f"{mr.wins_a}/{len(seeds)}")
            m2.metric("Mean PnL diff", f"{mr.mean_diff:+.0f}")
            m3.metric("t-stat",
                      f"{tt['t']:.2f}" if tt['t'] is not None else "—")
            m4.metric("p-value",
                      f"{tt['p_value']:.4f}" if tt['p_value'] is not None else "—")
            st.plotly_chart(pnl_distribution_figure(mr.diffs, a, b),
                            use_container_width=True)
            st.plotly_chart(paired_diff_figure(mr.diffs, a, b, mr.seeds),
                            use_container_width=True)

    elif mode == "Cross-agent leaderboard":
        chosen = st.multiselect("Agents", names, default=names)
        if len(chosen) < 2:
            st.warning("Pick at least two agents.")
        elif st.button("Run round-robin", type="primary"):
            with st.spinner("Running round-robin..."):
                rr = evaluate_roster({n: factories[n] for n in chosen}, seeds,
                                     n_hands=n_hands)
            st.subheader("Leaderboard (mean net chips)")
            st.plotly_chart(tournament_leaderboard_figure(rr.leaderboard),
                            use_container_width=True)
            st.subheader("Per-seed net-chip distribution")
            st.plotly_chart(pnl_box_figure(rr.per_agent_nets),
                            use_container_width=True)
            st.subheader("Head-to-head win matrix")
            st.plotly_chart(tournament_matrix_figure(rr.win_matrix, rr.n_seeds),
                            use_container_width=True)

    elif mode == "RL learning curve":
        if rl_history:
            st.plotly_chart(learning_curve_figure(rl_history),
                            use_container_width=True)
            st.caption(f"{len(rl_history)} held-out eval snapshots from the "
                       "loaded checkpoint.")
        else:
            st.info("Load an RL checkpoint (sidebar) trained with `--eval-every` "
                    "so it carries a learning-curve history, e.g. "
                    "`python -m scripts.train_rl --save models/rl.pt --eval-every 200`.")

    elif mode == "Equity + drawdown":
        c1, c2 = st.columns(2)
        a = c1.selectbox("Agent A (seat 1)", names, index=0)
        b = c2.selectbox("Agent B (seat 2)", names,
                         index=min(1, len(names) - 1))
        seed = st.number_input("Seed", value=0, step=1)
        if st.button("Simulate one match", type="primary"):
            pa = factories[a](1, 1000)
            pb = factories[b](2, 1000)
            with st.spinner("Simulating..."):
                res = simulate_session([pa, pb], n_hands=n_hands, seed=int(seed))
            for pid, nm in ((1, a), (2, b)):
                st.metric(f"{nm} (seat {pid}) max drawdown",
                          f"{max_drawdown(res, pid):.0f} chips")
                st.plotly_chart(equity_drawdown_figure(drawdown_curve(res, pid), nm),
                                use_container_width=True)

    elif mode == "Parameter sweep":
        st.caption("Round-robin a grid of (tight × aggression) personalities — "
                   "the static-skill fitness landscape an RL agent must beat.")
        c1, c2 = st.columns(2)
        tight_raw = c1.text_input("tight_threshold values", "0.2,0.4,0.6,0.8")
        aggr_raw = c2.text_input("aggression values", "0.3,0.5,0.7,0.9")
        fast = st.checkbox("Fast mode (no Monte Carlo — much quicker)", value=True)
        add_extra = st.checkbox("Add Kelly + adaptive agents to the grid",
                                value=False)
        if st.button("Run sweep", type="primary"):
            try:
                tights = [float(x) for x in tight_raw.split(",") if x.strip()]
                aggrs = [float(x) for x in aggr_raw.split(",") if x.strip()]
            except ValueError:
                st.error("Grid values must be comma-separated numbers.")
                return
            extra = None
            if add_extra:
                extra = {n: factories[n] for n in
                         ("Kelly", "Adaptive(tilt)", "Adaptive(random)")
                         if n in factories}
            n_cells = len(tights) * len(aggrs) + (len(extra) if extra else 0)
            with st.spinner(f"Round-robin over {n_cells} agents "
                            f"× {n_seeds} seeds..."):
                rr, grid = parameter_sweep(
                    tights, aggrs, seeds, n_hands=n_hands,
                    mc_sims=(None if fast else mc_sims), fast_mode=fast,
                    extra_agents=extra)
            st.subheader("Fitness landscape (mean net chips)")
            st.plotly_chart(parameter_heatmap_figure(grid),
                            use_container_width=True)
            st.subheader("Leaderboard")
            st.plotly_chart(tournament_leaderboard_figure(rr.leaderboard),
                            use_container_width=True)
            st.subheader("Per-seed net-chip distribution")
            st.plotly_chart(pnl_box_figure(rr.per_agent_nets),
                            use_container_width=True)


def main():
    st.set_page_config(page_title="Poker Simulator", layout="wide")
    st.title("Poker-Quant Simulator")

    page = st.sidebar.radio("Page",
                            ["Run", "Replay", "Tournament", "Evaluation"])

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

    elif page == "Evaluation":
        _evaluation_page()


if __name__ == "__main__":
    main()
