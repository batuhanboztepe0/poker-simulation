"""
featurizer.py
-------------
18-dimensional state featurization phi(s) for the RL agent (Phase C).

Pure, torch-free: maps a (game_state, hero) pair to a fixed-length float
vector for the Q-network. Imports nothing heavy so the non-torch path stays
clean.

Cross-sectional-alpha parallel: phi(s) is the feature row; the Q-net + TD
update is the model trained on realised PnL.
"""

from src.ev_calculator import pot_odds

FEATURE_DIM = 18

# Extended-feature dimension constants (for callers that opt in to the
# horizon and/or belief appends via featurize_extended).
FEATURE_DIM_HORIZON = 19   # base + horizon_frac
FEATURE_DIM_BELIEF = 20    # base + 2 belief floats (posterior_mean, p_tilted)
FEATURE_DIM_FULL = 21      # base + horizon_frac + 2 belief floats

# Streets in canonical order for the one-hot block.
_STREETS = ["Pre-Flop", "Flop", "Turn", "River"]


def featurize(game_state, hero, equity=None):
    """
    Build the 18-dim feature vector phi(s).

    Args:
        game_state (dict): Engine game-state snapshot.
        hero (Player): The acting player (for stack / position).
        equity (float | None): Precomputed equity; if None, uses
            game_state.get("equity") or 0.5.

    Returns:
        list[float]: A length-FEATURE_DIM feature vector.
    """
    pot = max(1, game_state.get("pot", 1))
    call_amount = game_state.get("call_amount", 0)
    current_bet = game_state.get("current_bet", 0)
    n_players = max(1, game_state.get("n_players", 2))
    active = game_state.get("active_player_count", 2)
    opponent_ids = game_state.get("opponent_ids") or []
    n_opp = len(opponent_ids) if opponent_ids else max(1, active - 1)

    stack = getattr(hero, "stack", 0)
    if equity is None:
        equity = game_state.get("equity")
        if equity is None:
            equity = 0.5

    all_stacks = game_state.get("all_stacks") or {}
    total_chips = sum(all_stacks.values()) if all_stacks else max(1, stack)
    total_chips = max(1, total_chips)

    odds = pot_odds(call_amount, pot) if call_amount > 0 else 0.0
    street = game_state.get("round_name", "Pre-Flop")
    street_onehot = [1.0 if street == s else 0.0 for s in _STREETS]

    spr = stack / pot  # stack-to-pot ratio
    call_frac = call_amount / max(1, stack)  # fraction of stack to call

    features = [
        float(equity),                       # 0  hand equity
        float(odds),                         # 1  pot odds faced
        float(equity - odds),                # 2  edge over pot odds
        min(5.0, pot / max(1, stack)),       # 3  pot / stack
        min(5.0, spr),                       # 4  stack / pot (SPR)
        min(1.0, call_frac),                 # 5  call / stack
        stack / total_chips,                 # 6  share of all chips
        min(1.0, current_bet / pot),         # 7  current bet / pot
        n_opp / 8.0,                         # 8  opponents (normalized)
        active / max(1, n_players),          # 9  fraction still active
        1.0 if call_amount == 0 else 0.0,    # 10 can check
        1.0 if call_amount >= stack else 0.0,  # 11 facing all-in
        street_onehot[0],                    # 12 pre-flop
        street_onehot[1],                    # 13 flop
        street_onehot[2],                    # 14 turn
        street_onehot[3],                    # 15 river
        min(1.0, len(game_state.get("community_cards", [])) / 5.0),  # 16 board
        min(1.0, pot / total_chips),         # 17 pot / all chips
    ]
    return features


def featurize_extended(game_state, hero, equity=None, horizon=None,
                       belief=None):
    """
    Extended feature vector phi_ext(s), optionally appending horizon and/or
    belief floats to the canonical 18-dim featurize() output.

    The first 18 positions are byte-identical to featurize(game_state, hero,
    equity).  Additional floats are appended in a fixed order: horizon first
    (if supplied), then belief (if supplied), so callers can use the
    FEATURE_DIM_* constants to know the resulting length.

    Args:
        game_state (dict): Engine game-state snapshot.
        hero (Player): The acting player.
        equity (float | None): Precomputed equity (passed through to featurize).
        horizon (tuple | None): (hands_remaining, hands_per_episode). Appends
            a single float `hands_remaining / max(1, hands_per_episode)` clamped
            to [0.0, 1.0].
        belief (BeliefState | None): Opponent belief state.  Appends two floats:
            belief.posterior_mean() and belief.p_tilted().  Any NaN/Inf is
            clamped to [0.0, 1.0] to avoid corrupting the Q-network.

    Returns:
        list[float]: Length 18, 19, 20, or 21 depending on which extras are set.
    """
    features = featurize(game_state, hero, equity=equity)

    if horizon is not None:
        remaining, total = horizon
        frac = float(remaining) / max(1, total)
        frac = max(0.0, min(1.0, frac))
        features = features + [frac]

    if belief is not None:
        pm = float(belief.posterior_mean())
        pt = float(belief.p_tilted())
        # Guard against degenerate belief states (NaN / Inf from particle filter).
        pm = pm if (pm == pm and abs(pm) != float("inf")) else 0.5
        pt = pt if (pt == pt and abs(pt) != float("inf")) else 0.0
        features = features + [pm, pt]

    return features
