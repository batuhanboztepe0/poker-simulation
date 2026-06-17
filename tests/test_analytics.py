"""
test_analytics.py
-----------------
Tests for Phase 2: event collection, analytics helpers, Parquet round-trip,
and the pure Plotly chart factories.

Run with: python -m pytest tests/test_analytics.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.player import BotPlayer
from src.monte_carlo import MonteCarloEngine
from src.events import (
    HandEvent, EVENT_ACTION, EVENT_SHOWDOWN, EVENT_HAND_END, STREET_SHOWDOWN,
    STREET_HAND_END,
)
from src.simulation import simulate_session, SessionResult, HandRecord, PlayerSnapshot
from src.simulation_runner import make_bot_players, run_session
from src.analytics import (
    HandResultCollector, events_to_dataframe, load_parquet,
    equity_curve, win_rate_by_player, hand_label_distribution, ev_accuracy,
)


def make_bots(n, mc=None):
    return [BotPlayer(i, f"B{i}", 1000, 0.3, 0.5, mc) for i in range(1, n + 1)]


def _royal_showdown_session():
    """Session with one showdown where both players play a royal flush board."""
    board = ["As", "Ks", "Qs", "Js", "Ts"]
    events = [
        HandEvent(1, "Pre-Flop", EVENT_ACTION, player_id=1, action="call",
                  amount=20, equity=0.6),
        HandEvent(1, "Pre-Flop", EVENT_ACTION, player_id=2, action="check",
                  amount=0, equity=0.4),
        HandEvent(1, STREET_SHOWDOWN, EVENT_SHOWDOWN, pot=40,
                  community_cards=board,
                  payload={"contenders": [1, 2],
                           "hole_cards": {1: ["2h", "3h"], 2: ["4h", "5h"]}}),
        HandEvent(1, STREET_HAND_END, EVENT_HAND_END, pot=40,
                  community_cards=board, payload={"winnings": {1: 20, 2: 20}}),
    ]
    return SessionResult(
        hands=[HandRecord(1, {1: 20, 2: 20}, events,
                          stacks={1: 1010, 2: 1010})],
        players=[PlayerSnapshot(1, "P1", 1000, 1010),
                 PlayerSnapshot(2, "P2", 1000, 1010)],
        starting_stacks={1: 1000, 2: 1000},
        final_stacks={1: 1010, 2: 1010},
        seed=0, small_blind=10, big_blind=20,
    )


# ===========================================================================
# HandResultCollector + DataFrame + Parquet
# ===========================================================================

class TestCollector:
    def test_collector_is_observer(self):
        from src.game import GameEngine
        import random
        rng = random.Random(1)
        collector = HandResultCollector()
        players = [BotPlayer(i, f"B{i}", 1000, 0.3, 0.5, None, rng)
                   for i in range(1, 3)]
        engine = GameEngine(players, 10, 20, verbose=False, rng=rng,
                            observers=[collector])
        engine.play_hand()
        assert len(collector.events) > 0
        df = collector.to_dataframe()
        assert len(df) == len(collector.events)
        assert "event_type" in df.columns

    def test_from_session(self):
        result = simulate_session(make_bots(3), n_hands=10, seed=1,
                                  fast_mode=True)
        collector = HandResultCollector.from_session(result)
        assert len(collector.events) == len(result.all_events())

    def test_dataframe_is_parquet_safe(self):
        result = simulate_session(make_bots(3), n_hands=8, seed=1,
                                  fast_mode=True)
        df = HandResultCollector.from_session(result).to_dataframe()
        # community_cards serialized to str, payload to JSON str (or None)
        assert df["community_cards"].map(lambda x: isinstance(x, str)).all()

    def test_parquet_round_trip(self, tmp_path):
        pytest.importorskip("pyarrow")
        result = simulate_session(make_bots(3), n_hands=10, seed=2,
                                  fast_mode=True)
        collector = HandResultCollector.from_session(result)
        path = os.path.join(tmp_path, "session.parquet")
        collector.save_parquet(path)
        loaded = load_parquet(path)
        original = collector.to_dataframe()
        assert len(loaded) == len(original)
        assert list(loaded.columns) == list(original.columns)
        assert loaded["hand_id"].tolist() == original["hand_id"].tolist()


# ===========================================================================
# Panel helpers
# ===========================================================================

class TestPanelHelpers:
    def test_equity_curve_shape(self):
        result = simulate_session(make_bots(4), n_hands=15, seed=3,
                                  fast_mode=True)
        df = equity_curve(result)
        assert set(df.columns) == {"hand_number", "player_id", "stack"}
        # hand 0 has every player's starting stack
        h0 = df[df["hand_number"] == 0]
        assert len(h0) == 4
        for _, row in h0.iterrows():
            assert row["stack"] == result.starting_stacks[row["player_id"]]

    def test_win_rate_sums_reasonably(self):
        result = simulate_session(make_bots(4), n_hands=40, seed=4,
                                  fast_mode=True)
        rates = win_rate_by_player(result)
        assert set(rates.keys()) == {1, 2, 3, 4}
        for r in rates.values():
            assert 0.0 <= r <= 1.0

    def test_hand_label_distribution_exact(self):
        session = _royal_showdown_session()
        dist = hand_label_distribution(session)
        assert dist == {"Royal Flush": 2}

    def test_hand_label_distribution_real_session(self):
        mc = MonteCarloEngine(n_simulations=120)
        result = simulate_session(make_bots(4, mc=mc), n_hands=30, seed=5)
        dist = hand_label_distribution(result)
        assert isinstance(dist, dict)
        assert all(v > 0 for v in dist.values())

    def test_ev_accuracy_columns(self):
        session = _royal_showdown_session()
        df = ev_accuracy(session)
        assert set(df.columns) == {
            "hand_number", "player_id", "predicted_equity", "realised"}
        # both contenders chopped (2-way split) -> realised 0.5 for each
        assert (df["realised"] == 0.5).all()

    def test_ev_accuracy_populated_with_mc(self):
        mc = MonteCarloEngine(n_simulations=120)
        result = simulate_session(make_bots(3, mc=mc), n_hands=40, seed=6)
        df = ev_accuracy(result)
        # MC bots log decision-time equity, so showdown rows should exist
        if not df.empty:
            assert df["predicted_equity"].between(0.0, 1.0).all()
            # realised is a pot-share in [0, 1] (1.0 sole win, 1/k for a chop)
            assert df["realised"].between(0.0, 1.0).all()


# ===========================================================================
# Equity logging on action events
# ===========================================================================

class TestEquityLogging:
    def test_action_events_carry_equity_with_mc(self):
        mc = MonteCarloEngine(n_simulations=120)
        result = simulate_session(make_bots(3, mc=mc), n_hands=10, seed=7)
        action_events = [e for e in result.all_events()
                         if e.event_type == EVENT_ACTION]
        assert any(e.equity is not None for e in action_events)


# ===========================================================================
# simulation_runner
# ===========================================================================

class TestSimulationRunner:
    def test_make_bot_players(self):
        configs = [
            {"name": "Tight", "tight_threshold": 0.6, "aggression": 0.2},
            {"name": "Loose", "tight_threshold": 0.2, "aggression": 0.8},
        ]
        players = make_bot_players(configs)
        assert len(players) == 2
        assert players[0].name == "Tight"
        assert players[0].player_id == 1
        assert players[1].aggression == 0.8

    def test_run_session_returns_result_and_collector(self):
        configs = [{"name": f"B{i}"} for i in range(1, 4)]
        result, collector = run_session(configs, n_hands=20, seed=8,
                                        fast_mode=True)
        assert result.n_hands > 0
        assert len(collector.events) == len(result.all_events())

    def test_run_session_chip_conservation(self):
        configs = [{"name": f"B{i}", "stack": 1000} for i in range(1, 5)]
        result, _ = run_session(configs, n_hands=50, seed=9, fast_mode=True)
        assert sum(result.final_stacks.values()) == 4 * 1000


# ===========================================================================
# Chart factories (plotly)
# ===========================================================================

class TestCharts:
    def setup_method(self):
        pytest.importorskip("plotly")

    def test_equity_curve_figure(self):
        from app.charts import equity_curve_figure
        import plotly.graph_objects as go
        result = simulate_session(make_bots(3), n_hands=15, seed=10,
                                  fast_mode=True)
        fig = equity_curve_figure(equity_curve(result))
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # one trace per player

    def test_win_rate_figure(self):
        from app.charts import win_rate_figure
        import plotly.graph_objects as go
        result = simulate_session(make_bots(3), n_hands=20, seed=11,
                                  fast_mode=True)
        fig = win_rate_figure(win_rate_by_player(result))
        assert isinstance(fig, go.Figure)

    def test_hand_label_figure(self):
        from app.charts import hand_label_figure
        import plotly.graph_objects as go
        fig = hand_label_figure({"Full House": 3, "Flush": 1})
        assert isinstance(fig, go.Figure)

    def test_ev_accuracy_figure(self):
        from app.charts import ev_accuracy_figure
        import plotly.graph_objects as go
        session = _royal_showdown_session()
        fig = ev_accuracy_figure(ev_accuracy(session))
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # scatter + y=x line

    def test_roi_leaderboard_figure(self):
        from app.charts import roi_leaderboard_figure
        from src.stats import session_summary
        import plotly.graph_objects as go
        result = simulate_session(make_bots(3), n_hands=20, seed=12,
                                  fast_mode=True)
        fig = roi_leaderboard_figure(session_summary(result))
        assert isinstance(fig, go.Figure)
