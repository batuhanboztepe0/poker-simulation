"""
simulation_runner.py
--------------------
Thin wrapper over `simulate_session` for the dashboard (Phase 2).

`make_bot_players` builds a roster from lightweight config dicts; `run_session`
runs a seeded session and bundles the result with an event collector so the
Streamlit app (and tests) can render every panel from one call.
"""

from src.player import BotPlayer
from src.monte_carlo import MonteCarloEngine
from src.simulation import simulate_session
from src.analytics import HandResultCollector

DEFAULT_STARTING_STACK = 1000


def make_bot_players(configs, starting_stack=DEFAULT_STARTING_STACK,
                     mc_engine=None):
    """
    Build BotPlayers from a list of config dicts.

    Each config may set: name, stack, tight_threshold, aggression.
    Missing keys fall back to sensible defaults. player_id is the 1-based
    index in the list.

    Args:
        configs (list[dict]): Per-bot configuration.
        starting_stack (int): Default stack when a config omits "stack".
        mc_engine (MonteCarloEngine | None): Shared equity engine for all bots.

    Returns:
        list[BotPlayer]
    """
    players = []
    for i, cfg in enumerate(configs, start=1):
        players.append(BotPlayer(
            player_id=i,
            name=cfg.get("name", f"Bot{i}"),
            stack=cfg.get("stack", starting_stack),
            tight_threshold=cfg.get("tight_threshold", 0.4),
            aggression=cfg.get("aggression", 0.5),
            mc_engine=mc_engine,
        ))
    return players


def run_session(configs, n_hands, seed=None, small_blind=10, big_blind=20,
                mc_simulations=200, fast_mode=False,
                starting_stack=DEFAULT_STARTING_STACK):
    """
    Run one seeded session and return (SessionResult, HandResultCollector).

    A fresh GameEngine is created per call inside simulate_session (no engine
    sharing across runs). When fast_mode is False, a single shared MC engine
    is built for all bots.

    Args:
        configs (list[dict]): Per-bot configuration.
        n_hands (int): Hands to play.
        seed (int | None): Top-level seed.
        small_blind, big_blind (int): Blind structure.
        mc_simulations (int): MC sims per equity call (ignored in fast_mode).
        fast_mode (bool): Disable MC for speed.
        starting_stack (int): Default per-bot stack.

    Returns:
        tuple: (SessionResult, HandResultCollector)
    """
    mc_engine = None
    if not fast_mode:
        mc_engine = MonteCarloEngine(n_simulations=mc_simulations)

    players = make_bot_players(configs, starting_stack=starting_stack,
                               mc_engine=mc_engine)
    result = simulate_session(
        players, n_hands,
        small_blind=small_blind, big_blind=big_blind,
        seed=seed, fast_mode=fast_mode,
    )
    collector = HandResultCollector.from_session(result)
    return result, collector
