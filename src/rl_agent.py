"""
rl_agent.py
-----------
Reinforcement-learning agent scaffold (Phase C).

A small DQN-style value model over the 18-dim featurization, trained by
self-play with Monte-Carlo (terminal-reward) returns. `torch` is an OPTIONAL
dependency: the module imports cleanly without it (so `ReplayBuffer` and the
action helpers are always available), and only `QNetwork`, `SelfPlayTrainer`
and `RLBotPlayer` require torch — they raise a clear error if it is missing.

Cross-sectional-alpha parallel: phi(s) features + Q-net + TD/MC target is an
alpha model fit on realised PnL.
"""

import copy
import math
import os
from collections import deque

from src.player import (
    BotPlayer,
    ACTION_FOLD, ACTION_CHECK, ACTION_CALL, ACTION_RAISE, ACTION_ALL_IN,
)
from src.featurizer import (
    featurize, featurize_extended, FEATURE_DIM,
    FEATURE_DIM_HORIZON, FEATURE_DIM_BELIEF, FEATURE_DIM_FULL,
)
from src.icm import icm_equity
from src.monte_carlo import MonteCarloEngine

try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except ImportError:  # pragma: no cover - exercised only without torch
    _HAVE_TORCH = False


# Discrete action set indexed by the Q-network output.
ACTION_NAMES = ["fold", "passive", "raise_half", "raise_pot", "all_in"]
N_ACTIONS = len(ACTION_NAMES)

# Extended 7-action grid.  Indices 2-5 are raise fractions of the pot:
#   2=quarter, 3=half, 4=two-thirds, 5=pot, 6=all-in.
# WARNING: these indices are INCOMPATIBLE with the 5-action grid — do not
# mix checkpoints trained with different grids.
EXTENDED_ACTION_NAMES = [
    "fold", "passive",
    "raise_quarter", "raise_half", "raise_two_thirds", "raise_pot",
    "all_in",
]
N_EXTENDED_ACTIONS = len(EXTENDED_ACTION_NAMES)


# ---------------------------------------------------------------------------
# Action mapping (torch-free)
# ---------------------------------------------------------------------------

def legal_action_indices(game_state, stack):
    """Indices into ACTION_NAMES that are legal in this state."""
    call_amount = game_state.get("call_amount", 0)
    min_raise = game_state.get("min_raise", max(call_amount * 2, 1))
    legal = [0, 1]  # fold + passive (check/call) always available
    can_raise = stack > call_amount and stack >= min_raise
    if can_raise:
        legal += [2, 3]
    if stack > 0:
        legal.append(4)  # all-in
    return legal


def map_action_index(idx, game_state, stack):
    """Translate an action index into a concrete (action, amount)."""
    call_amount = game_state.get("call_amount", 0)
    pot = game_state.get("pot", 1)
    min_raise = game_state.get("min_raise", max(call_amount * 2, 1))
    current_bet = game_state.get("current_bet", 0)

    if idx == 0:
        return ACTION_FOLD, 0
    if idx == 1:
        if call_amount == 0:
            return ACTION_CHECK, 0
        if call_amount >= stack:
            return ACTION_ALL_IN, stack
        return ACTION_CALL, call_amount
    if idx in (2, 3):
        can_raise = stack > call_amount and stack >= min_raise
        if not can_raise:
            # fall back to passive
            return map_action_index(1, game_state, stack)
        frac = 0.5 if idx == 2 else 1.0
        target = current_bet + int(pot * frac)
        target = max(min_raise, min(target, stack))
        if target >= stack:
            return ACTION_ALL_IN, stack
        return ACTION_RAISE, target
    # all-in
    return ACTION_ALL_IN, stack


def legal_action_indices_ext(game_state, stack):
    """Indices into EXTENDED_ACTION_NAMES that are legal in this state."""
    call_amount = game_state.get("call_amount", 0)
    min_raise = game_state.get("min_raise", max(call_amount * 2, 1))
    legal = [0, 1]  # fold + passive always available
    can_raise = stack > call_amount and stack >= min_raise
    if can_raise:
        legal += [2, 3, 4, 5]  # raise_quarter, raise_half, raise_two_thirds, raise_pot
    if stack > 0:
        legal.append(6)  # all-in
    return legal


# Raise fractions for the extended grid (indices 2-5).
_EXT_RAISE_FRACS = [0.25, 0.5, 0.667, 1.0]


def map_action_index_ext(idx, game_state, stack):
    """Translate an extended action index into a concrete (action, amount)."""
    call_amount = game_state.get("call_amount", 0)
    pot = game_state.get("pot", 1)
    min_raise = game_state.get("min_raise", max(call_amount * 2, 1))
    current_bet = game_state.get("current_bet", 0)

    if idx == 0:
        return ACTION_FOLD, 0
    if idx == 1:
        if call_amount == 0:
            return ACTION_CHECK, 0
        if call_amount >= stack:
            return ACTION_ALL_IN, stack
        return ACTION_CALL, call_amount
    if idx in (2, 3, 4, 5):
        can_raise = stack > call_amount and stack >= min_raise
        if not can_raise:
            return map_action_index_ext(1, game_state, stack)
        frac = _EXT_RAISE_FRACS[idx - 2]
        target = current_bet + int(pot * frac)
        target = max(min_raise, min(target, stack))
        if target >= stack:
            return ACTION_ALL_IN, stack
        return ACTION_RAISE, target
    # idx == 6: all-in
    return ACTION_ALL_IN, stack


# ---------------------------------------------------------------------------
# Replay buffer (torch-free)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity transition buffer of (features, action_idx, return)."""

    def __init__(self, capacity=10000):
        self._buf = deque(maxlen=capacity)

    def push(self, features, action_idx, reward, next_features=None,
             next_legal=None, done=True):
        self._buf.append((features, action_idx, reward, next_features,
                          next_legal, done))

    def sample(self, batch_size, rng):
        k = min(batch_size, len(self._buf))
        return rng.sample(list(self._buf), k)

    def __len__(self):
        return len(self._buf)


def _require_torch():
    if not _HAVE_TORCH:
        raise ImportError(
            "This component requires PyTorch. Install torch to use the RL "
            "agent (the rest of Phase C works without it)."
        )


# ---------------------------------------------------------------------------
# Torch components
# ---------------------------------------------------------------------------

if _HAVE_TORCH:

    class QNetwork(nn.Module):
        """3-layer MLP value head: phi(s) -> Q for each discrete action."""

        def __init__(self, input_dim=FEATURE_DIM, hidden=64,
                     n_actions=N_ACTIONS):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )

        def forward(self, x):
            return self.net(x)

else:  # pragma: no cover - placeholder when torch is unavailable

    class QNetwork:  # type: ignore
        def __init__(self, *args, **kwargs):
            _require_torch()


class RLBotPlayer(BotPlayer):
    """
    BotPlayer that acts by argmax-Q over the legal discrete actions.

    Requires torch. With `training=True` it explores epsilon-greedily and logs
    (features, action_idx) per decision so a SelfPlayTrainer can attach the
    hand's realised return.

    feature_mode controls which feature vector is built:
        'base'    - featurize() (18 dims, default, byte-identical to before)
        'horizon' - featurize_extended(..., horizon=_horizon_info)  (19 dims)
        'belief'  - featurize_extended(..., belief=_belief)         (20 dims)
        'full'    - featurize_extended(..., horizon=..., belief=...) (21 dims)
    The trainer sets `_horizon_info` (a (remaining, total) tuple or None) before
    each hand in multi-hand mode.
    """

    def __init__(self, *args, qnet=None, epsilon=0.1, training=False,
                 feature_mode='base', **kwargs):
        _require_torch()
        super().__init__(*args, **kwargs)
        self.qnet = qnet if qnet is not None else QNetwork()
        self.epsilon = epsilon
        self.training = training
        self.feature_mode = feature_mode
        self._episode_log = []
        # Set by SelfPlayTrainer before each hand in multi-hand mode.
        self._horizon_info = None   # (hands_remaining, hands_per_episode) or None
        # Belief is used as a FEATURE only (the engine updates `belief_state` via
        # observe_action); equity stays the fast vanilla MC estimate, so the
        # belief-conditioning is an explicit input to the Q-net, not a hidden
        # change to the equity feature.
        self._belief = self.belief_state
        self.use_belief_equity = False

    def decide(self, game_state):
        equity = (self._estimate_equity(game_state)
                  if self.mc_engine is not None else None)

        if self.feature_mode == 'base':
            features = featurize(game_state, self, equity=equity)
            legal = legal_action_indices(game_state, self.stack)
        else:
            horizon = self._horizon_info if self.feature_mode in ('horizon', 'full') else None
            belief = self._belief if self.feature_mode in ('belief', 'full') else None
            features = featurize_extended(game_state, self, equity=equity,
                                          horizon=horizon, belief=belief)
            legal = legal_action_indices(game_state, self.stack)

        with torch.no_grad():
            q = self.qnet(torch.tensor(features, dtype=torch.float32)).tolist()

        if self.training and self._rng.random() < self.epsilon:
            idx = self._rng.choice(legal)
        else:
            idx = max(legal, key=lambda i: q[i])

        if self.training:
            # Log the legal set too so a TD trainer can take the bootstrap max
            # over the legal actions of this state when it becomes a "next
            # state" for the preceding decision.
            self._episode_log.append((features, idx, legal))
        return map_action_index(idx, game_state, self.stack)


def evaluate_vs_baseline(qnet, n_seeds=50, n_hands=200, mc_sims=100,
                         stack0=1000, small_blind=10, big_blind=20,
                         tight_threshold=0.2, aggression=0.5, seed_start=0):
    """
    Headline metric: play a greedy `RLBotPlayer(qnet)` heads-up against the
    myopic EV `BotPlayer` over `n_seeds` seeded bankroll matches.

    Each seed is one self-contained `n_hands`-hand match with persistent stacks
    (a busted player ends the match early) — the exact shape of the Kelly DoD
    test `TestKellyBot.test_beats_myopic_over_many_seeds`, so the RL number is
    directly comparable. A single seed drives the deck shuffle and both bots'
    MC sampling, so each seed is a paired observation (same decks for both
    seats). Note two properties of this format: (1) almost every match ends in
    a bust, so the per-seed diff is ~±2·stack0 and `wins` is effectively a
    Bernoulli count — prefer `paired_t_test`/many seeds over a raw win
    threshold; (2) the shared rng means each bot's MC draws interleave in a
    policy-dependent order, so the pairing controls the decks, not every draw.
    Hold `mc_sims` constant across paired calls (e.g. trained vs random-init).

    Args:
        qnet (QNetwork): The policy to evaluate (played greedily, epsilon=0).
        n_seeds (int): Number of seeded matches (seeds 0..n_seeds-1).
        n_hands (int): Hands per match.
        mc_sims (int): Monte-Carlo simulations per equity estimate.
        stack0 (int): Starting stack for both bots.
        small_blind, big_blind (int): Blind structure.
        tight_threshold, aggression (float): Myopic baseline personality
            (defaults match the Kelly DoD test's baseline).
        seed_start (int): First match seed (matches use seeds
            seed_start..seed_start+n_seeds-1). Use a non-zero start to evaluate
            on a held-out seed range disjoint from any earlier measurement.

    Returns:
        dict: {
            "wins": int,                # matches the RL bot finished ahead
            "n_seeds": int,
            "mean_chip_diff": float,    # mean (rl_stack - myopic_stack)
            "per_seed_diffs": list[int],
        }
    """
    _require_torch()
    import random as _random
    from src.game import GameEngine

    diffs = []
    wins = 0
    for seed in range(seed_start, seed_start + n_seeds):
        rng = _random.Random(seed)
        mc = MonteCarloEngine(n_simulations=mc_sims, rng=rng)
        rl_bot = RLBotPlayer(1, "RL", stack0, qnet=qnet, epsilon=0.0,
                             training=False, mc_engine=mc, rng=rng)
        myopic = BotPlayer(2, "Myopic", stack0, tight_threshold=tight_threshold,
                           aggression=aggression, mc_engine=mc, rng=rng)
        engine = GameEngine([rl_bot, myopic], small_blind, big_blind,
                            verbose=False, rng=rng)
        for _ in range(n_hands):
            if sum(1 for p in (rl_bot, myopic) if p.stack > 0) < 2:
                break
            engine.play_hand()
        diff = rl_bot.stack - myopic.stack
        diffs.append(diff)
        if rl_bot.stack > myopic.stack:
            wins += 1
    return {
        "wins": wins,
        "n_seeds": n_seeds,
        "mean_chip_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "per_seed_diffs": diffs,
    }


def paired_t_test(diffs):
    """
    One-sample (paired) t-test of per-seed chip diffs against 0.

    Each diff is already a paired observation (rl_stack - myopic_stack on the
    same seed), so testing the diffs against 0 is the paired test the handoff
    asks for. Uses scipy (exact t-distribution) when available, else a normal
    approximation — which is anti-conservative for small df (it understates p
    by ~2x near t=3 at n=50), so trust the scipy path for reported p-values.

    Returns:
        dict: {"mean", "n", "t", "p_value"} (p two-sided; None if degenerate).
    """
    n = len(diffs)
    mean = sum(diffs) / n if n else 0.0
    if n < 2:
        return {"mean": mean, "n": n, "t": None, "p_value": None}
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0:
        return {"mean": mean, "n": n, "t": None, "p_value": None}
    se = (var / n) ** 0.5
    t = mean / se
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    except Exception:
        # Normal approximation when scipy is unavailable.
        import math
        p = float(math.erfc(abs(t) / math.sqrt(2)))
    return {"mean": mean, "n": n, "t": float(t), "p_value": p}


class SelfPlayTrainer:
    """
    DQN trainer with TD(0) bootstrapped targets (target net + Huber loss).

    Each learner's consecutive decisions are chained into (s, a, r, s', done)
    transitions; the target is `r + gamma * max_a' Q_target(s', a')` over the
    legal next actions (terminal transitions use `r` alone). `train(n_steps)`
    performs n_steps minibatch Huber-regression updates toward those targets.

    Episode modes:
        single-hand (default, multi_hand=False) - stacks reset every hand, each
            hand is its own episode, reward = normalized clipped chip delta.
            Risk-neutral per-hand EV; the strongest beats-baseline number.
        multi-hand  (multi_hand=True) - stacks PERSIST across hands_per_episode
            hands, reward = per-hand change in log-utility of the stack
            (log(S'/S)), or in ICM prize equity when icm_prize_structure is set;
            decisions chain into one episode-long TD sequence. The agent
            optimizes long-run log-bankroll-growth (Kelly / ROADMAP s11).

    Opponent modes (the M1->M4 progression in docs/RL_HANDOFF.md):
        "self"     - all n_players bots are learners sharing the live net
                     (non-stationary; the original scaffold behavior).
        "fixed"    - one learner vs frozen myopic `BotPlayer`s (a STATIONARY
                     target; the most direct route to beating the baseline).
        "snapshot" - one learner vs frozen past *snapshots* of itself, sampled
                     from a growing pool (fictitious self-play; stable).

    Learners always estimate real equity via an injected `MonteCarloEngine`
    (M1) — without it feature[0] is a constant 0.5 and the net is near-blind.
    """

    def __init__(self, n_players=2, stack0=1000, hidden=64, lr=1e-3,
                 epsilon=0.2, seed=0, small_blind=10, big_blind=20,
                 opponent_mode="self", mc_sims=100, reward_clip=None,
                 epsilon_start=None, epsilon_end=None, snapshot_every=500,
                 baseline_kwargs=None, gamma=0.97, target_update_every=50,
                 multi_hand=False, hands_per_episode=15, util_floor=1.0,
                 icm_prize_structure=None, extended_features=False,
                 feature_mode='base', opponent_factory=None,
                 reward_mode='log', learner_belief_factory=None,
                 opponent_factories=None, tilt_reward_bonus=0.0):
        _require_torch()
        import random as _random
        from src.game import GameEngine

        if icm_prize_structure is not None and not multi_hand:
            raise ValueError(
                "icm_prize_structure requires multi_hand=True; pass "
                "multi_hand=True to activate bankroll-episode mode."
            )

        self.stack0 = stack0
        self.reward_scale = float(stack0)
        # Multi-hand (bankroll) episodes: stacks PERSIST across hands_per_episode
        # hands and the reward is the per-hand change in log-utility of the
        # stack (the log-utility Bellman of ROADMAP s11) -> growth-optimal,
        # risk-averse near bust. Single-hand mode (default) keeps the original
        # i.i.d. per-hand chip-delta reward and is unchanged.
        self.multi_hand = multi_hand
        self.hands_per_episode = hands_per_episode
        self.util_floor = float(util_floor)
        # ICM reward mode: per-hand reward is change in ICM equity (prize-pool
        # weighted by stack shares). Requires multi_hand=True.
        self.icm_prize_structure = icm_prize_structure
        # Extended feature mode.
        self.extended_features = extended_features
        self.feature_mode = feature_mode
        # Multi-hand reward_mode: 'log' = per-hand log-utility change (risk-averse,
        # default), 'chips' = normalized chip delta (RISK-NEUTRAL, so the agent
        # gambles into a spewing opponent), 'icm' implied when icm_prize_structure
        # is set. learner_belief_factory wires an opponent belief into each learner
        # (engine auto-updates it; used as a feature). opponent_factories is a pool
        # rotated per episode (domain randomization -> resists overfitting to one
        # opponent).
        self.reward_mode = 'icm' if icm_prize_structure is not None else reward_mode
        self.learner_belief_factory = learner_belief_factory
        self.opponent_factories = opponent_factories
        # Tilt-exploitation shaping (chips reward, belief on): scale a hand's
        # reward by (1 + tilt_reward_bonus * p_tilted) so the value head weights
        # hands where the opponent is detected as tilted more heavily — it learns
        # to press its edge against a spewing opponent. 0.0 = off (no shaping).
        self.tilt_reward_bonus = tilt_reward_bonus
        # Log-utility per-hand deltas span a wide range (a bust is ~-log(stack0)),
        # so log-mode multi-hand wants a looser clip; 'chips' stays on the tight
        # chip-delta scale like single-hand.
        if reward_clip is None:
            loose = multi_hand and self.reward_mode != 'chips'
            reward_clip = 3.0 if loose else 1.0
        self.reward_clip = reward_clip
        self.rng = _random.Random(seed)

        # Compute the QNetwork input dimension from feature_mode.
        if extended_features:
            _mode = feature_mode
            if _mode == 'horizon':
                _input_dim = FEATURE_DIM_HORIZON
            elif _mode == 'belief':
                _input_dim = FEATURE_DIM_BELIEF
            elif _mode == 'full':
                _input_dim = FEATURE_DIM_FULL
            else:
                _input_dim = FEATURE_DIM
        else:
            _input_dim = FEATURE_DIM

        self.qnet = QNetwork(input_dim=_input_dim, hidden=hidden)
        # Target network for the TD bootstrap (frozen between periodic syncs);
        # gives a stable regression target so the value estimates don't chase
        # their own tail.
        self.target_qnet = QNetwork(input_dim=_input_dim, hidden=hidden)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.gamma = gamma
        self.target_update_every = target_update_every
        self._grad_count = 0
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=50000)
        self.opponent_mode = opponent_mode
        self.snapshot_every = snapshot_every
        self.epsilon_start = epsilon if epsilon_start is None else epsilon_start
        self.epsilon_end = self.epsilon_start if epsilon_end is None else epsilon_end
        self.history = []

        # One shared MC engine for every learner/opponent that needs equity.
        # A dedicated rng keeps MC sampling reproducible and decoupled from the
        # minibatch-sampling rng.
        self.mc = MonteCarloEngine(n_simulations=mc_sims,
                                   rng=_random.Random(seed + 200))
        bk = baseline_kwargs or dict(tight_threshold=0.2, aggression=0.5)

        _fm = feature_mode if extended_features else 'base'

        def _learner(i):
            belief = (learner_belief_factory()
                      if learner_belief_factory is not None else None)
            return RLBotPlayer(i, f"RL{i}", stack0, qnet=self.qnet,
                               epsilon=self.epsilon_start, training=True,
                               mc_engine=self.mc, rng=_random.Random(seed + i),
                               feature_mode=_fm, belief_state=belief)

        if opponent_mode == "self":
            self.learners = [_learner(i) for i in range(1, n_players + 1)]
            self.opponents = []
        elif opponent_mode == "fixed":
            self.learners = [_learner(1)]
            if self.opponent_factories:
                # A pool of opponent archetypes; rotated per episode in
                # _collect_episode (domain randomization). Seat one to start.
                self.opponents = [
                    self.rng.choice(self.opponent_factories)(
                        i, stack0, self.mc, _random.Random(seed + i))
                    for i in range(2, n_players + 1)
                ]
            elif opponent_factory is not None:
                # Custom frozen opponents (e.g. adaptive / tilt bots) so the
                # learner trains against a non-myopic, possibly non-stationary
                # target. Factory signature: (player_id, stack, mc_engine, rng).
                self.opponents = [
                    opponent_factory(i, stack0, self.mc,
                                     _random.Random(seed + i))
                    for i in range(2, n_players + 1)
                ]
            else:
                self.opponents = [
                    BotPlayer(i, f"Fix{i}", stack0, mc_engine=self.mc,
                              rng=_random.Random(seed + i), **bk)
                    for i in range(2, n_players + 1)
                ]
        elif opponent_mode == "snapshot":
            self.learners = [_learner(1)]
            # Opponents are RL bots driven by frozen weight snapshots; they
            # never train and their logs are discarded.
            self.opponents = [
                RLBotPlayer(i, f"Snap{i}", stack0,
                            qnet=QNetwork(input_dim=_input_dim, hidden=hidden),
                            epsilon=0.05, training=False, mc_engine=self.mc,
                            rng=_random.Random(seed + i))
                for i in range(2, n_players + 1)
            ]
            self.snapshot_pool = [copy.deepcopy(self.qnet.state_dict())]
        else:
            raise ValueError(f"unknown opponent_mode {opponent_mode!r}")

        self.players = self.learners + self.opponents
        self.engine = GameEngine(self.players, small_blind=small_blind,
                                 big_blind=big_blind, verbose=False,
                                 rng=_random.Random(seed + 100))

    def _take_snapshot(self):
        self.snapshot_pool.append(copy.deepcopy(self.qnet.state_dict()))

    def _reseat_snapshot_opponents(self):
        """Load a random pooled snapshot into each frozen opponent net."""
        for opp in self.opponents:
            state = self.rng.choice(self.snapshot_pool)
            opp.qnet.load_state_dict(state)

    def _utility(self, stack):
        """Log-utility of a chip stack (floored so a bust is finite-bad)."""
        return math.log(max(stack, self.util_floor))

    def _clip(self, r):
        return max(-self.reward_clip, min(self.reward_clip, r))

    def _icm_utility(self, stacks_list):
        """
        ICM equity for the learner given current stacks.

        Args:
            stacks_list (list[float]): All player stacks in self.players order.

        Returns:
            float: The learner's (index 0 in self.learners) ICM prize equity.
        """
        # The learner is always self.learners[0]; its index in self.players is 0.
        hero_index = self.players.index(self.learners[0])
        return icm_equity(stacks_list, self.icm_prize_structure)[hero_index]

    def _collect(self, n_hands):
        if self.multi_hand:
            self._collect_multi(n_hands)
        else:
            self._collect_single(n_hands)

    def _collect_single(self, n_hands):
        for _ in range(n_hands):
            if self.opponent_mode == "snapshot":
                self._reseat_snapshot_opponents()
            before = {}
            for p in self.players:
                p.stack = self.stack0
                if hasattr(p, "_episode_log"):
                    p._episode_log = []
                before[p.player_id] = p.stack
            self.engine.play_hand()
            for b in self.learners:
                raw = (b.stack - before[b.player_id]) / self.reward_scale
                reward = self._clip(raw)
                log = b._episode_log
                # Chain this bot's consecutive decisions into TD transitions:
                # the realised hand reward lands only on the terminal decision;
                # earlier decisions carry r=0 and bootstrap off the NEXT state
                # this same bot faced. The bootstrap is a max over legal actions
                # (DQN), so a decision is valued as "act, then play optimally"
                # rather than being blamed for the bot's own later mistakes.
                for i, (feat, idx, legal) in enumerate(log):
                    if i + 1 < len(log):
                        nxt_feat, _, nxt_legal = log[i + 1]
                        self.buffer.push(feat, idx, 0.0, nxt_feat, nxt_legal,
                                         False)
                    else:
                        self.buffer.push(feat, idx, reward, None, None, True)

    def _collect_multi(self, n_hands):
        """
        Bankroll episodes: play matches of up to `hands_per_episode` hands with
        PERSISTENT stacks. Each hand's reward is the change in the learner's
        log-utility of its stack (or ICM equity when icm_prize_structure is
        set); decisions are chained into one TD sequence spanning the whole
        episode. `n_hands` is the approximate hand budget for this call.
        """
        n_episodes = max(1, math.ceil(n_hands / self.hands_per_episode))
        for _ in range(n_episodes):
            self._collect_episode()

    def _reseat_rotating_opponents(self):
        """
        Rebuild each frozen opponent from a randomly-chosen factory in
        `opponent_factories` and reseat it (domain randomization over opponent
        archetypes — resists overfitting to any single opponent).
        """
        import random as _random
        new_opps = []
        for opp in self.opponents:
            fac = self.rng.choice(self.opponent_factories)
            new_opps.append(fac(opp.player_id, self.stack0, self.mc,
                                _random.Random(self.rng.randint(0, 2**31 - 1))))
        self.opponents = new_opps
        self.players = self.learners + self.opponents
        self.engine.players = self.players

    def _collect_episode(self):
        if self.opponent_mode == "snapshot":
            self._reseat_snapshot_opponents()
        elif self.opponent_factories:
            self._reseat_rotating_opponents()
        # Fresh opponent belief per episode (the opponent's regime resets too).
        if self.learner_belief_factory is not None:
            for b in self.learners:
                b.belief_state = self.learner_belief_factory()
                b._belief = b.belief_state
        # Reset stacks ONCE, at the start of the episode.
        for p in self.players:
            p.stack = self.stack0
            if hasattr(p, "_episode_log"):
                p._episode_log = []

        # Per learner, accumulate the episode's decisions as mutable
        # [feat, idx, legal, reward] rows. `carry` holds utility change from
        # hands in which the bot never acted (e.g. the BB winning when the SB
        # folds pre-flop) so no bankroll change is dropped.
        rows = {b.player_id: [] for b in self.learners}
        carry = {b.player_id: 0.0 for b in self.learners}

        for hand_num in range(self.hands_per_episode):
            if sum(1 for p in self.players if p.stack > 0) < 2:
                break
            before_stacks = {b.player_id: b.stack for b in self.learners}
            all_before = [p.stack for p in self.players]
            for b in self.learners:
                b._episode_log = []
                # Update horizon info for extended feature modes.
                if self.extended_features and self.feature_mode in ('horizon', 'full'):
                    b._horizon_info = (self.hands_per_episode - hand_num,
                                       self.hands_per_episode)
            self.engine.play_hand()

            for b in self.learners:
                # Per-hand reward: ICM equity, risk-neutral chip delta, or
                # log-utility change (self.reward_mode).
                if self.reward_mode == 'icm':
                    all_after = [p.stack for p in self.players]
                    d_util = (self._icm_utility(all_after)
                              - self._icm_utility(all_before))
                elif self.reward_mode == 'chips':
                    d_util = ((b.stack - before_stacks[b.player_id])
                              / self.reward_scale)
                    if (self.tilt_reward_bonus and b.belief_state is not None
                            and d_util > 0):
                        # Up-weight WINS against a detected-tilted opponent (press
                        # the edge). Gains only: amplifying losses too makes the
                        # agent cautious exactly when it should attack.
                        d_util *= (1.0 + self.tilt_reward_bonus
                                   * b.belief_state.p_tilted())
                else:
                    d_util = (self._utility(b.stack)
                              - self._utility(before_stacks[b.player_id]))

                decisions = b._episode_log
                if decisions:
                    bonus = carry[b.player_id]
                    carry[b.player_id] = 0.0
                    last = len(decisions) - 1
                    for j, (feat, idx, legal) in enumerate(decisions):
                        r = (d_util + bonus) if j == last else 0.0
                        rows[b.player_id].append([feat, idx, legal, r])
                else:
                    carry[b.player_id] += d_util

        for b in self.learners:
            seq = rows[b.player_id]
            if not seq:
                # Learner never acted the whole episode -> no decision to credit.
                # Practically unreachable heads-up; negligible and conservative.
                continue
            # Flush any trailing carry (actionless final hands) onto the last
            # decision so the episode's total utility change is conserved.
            seq[-1][3] += carry[b.player_id]
            for k, (feat, idx, legal, r) in enumerate(seq):
                r = self._clip(r)
                if k + 1 < len(seq):
                    nxt_feat, _, nxt_legal, _ = seq[k + 1]
                    self.buffer.push(feat, idx, r, nxt_feat, nxt_legal, False)
                else:
                    self.buffer.push(feat, idx, r, None, None, True)

    def _grad_step(self, batch):
        feats = torch.tensor([t[0] for t in batch], dtype=torch.float32)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        targets = torch.tensor([t[2] for t in batch], dtype=torch.float32)

        # Bootstrap the non-terminal targets: reward (0) + gamma * max_a'
        # Q_target(s', a') over the LEGAL actions of the next state.
        nonterm = [i for i, t in enumerate(batch) if not t[5]]
        if nonterm:
            nxt = torch.tensor([batch[i][3] for i in nonterm],
                               dtype=torch.float32)
            with torch.no_grad():
                q_next = self.target_qnet(nxt)
            for j, i in enumerate(nonterm):
                legal = batch[i][4]
                maxq = max(q_next[j][a].item() for a in legal)
                # Bellman target r + gamma*max Q(s',a'). The stored reward
                # (batch[i][2]) is 0 for intermediate decisions today, but
                # keep it in the sum so any future intermediate reward shaping
                # is honoured instead of silently dropped.
                targets[i] = batch[i][2] + self.gamma * maxq

        q = self.qnet(feats)
        q_a = q.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Huber (smooth-L1) is robust to the rare all-in chip swings that would
        # otherwise dominate an MSE loss.
        loss = nn.functional.smooth_l1_loss(q_a, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._grad_count += 1
        if self._grad_count % self.target_update_every == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())
        return float(loss.item())

    def _set_epsilon(self, frac):
        eps = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac
        for b in self.learners:
            b.epsilon = eps

    def train(self, n_steps, batch_size=32, refresh_every=10,
              hands_per_refresh=12, eval_every=None, eval_seeds=20,
              eval_hands=100, eval_mc_sims=100, eval_seed_start=1000):
        """
        Run n_steps gradient updates; return the list of per-step losses.

        When `eval_every` is set, every `eval_every` steps an
        `evaluate_vs_baseline` snapshot is appended to `self.history` (the
        learning curve for the report). The return value is always the loss
        list, so the default call `train(n)` is unchanged.

        Intermediate evals use seeds `eval_seed_start..` (default 1000),
        deliberately disjoint from the headline range (0..49) so reading the
        learning curve can never amount to picking the checkpoint that looks
        best on the *headline* decks.
        """
        losses = []
        self.history = []
        self._collect(hands_per_refresh)
        for step in range(n_steps):
            self._set_epsilon(step / max(1, n_steps - 1))
            if step > 0 and step % refresh_every == 0:
                self._collect(hands_per_refresh)
                if (self.opponent_mode == "snapshot"
                        and step % self.snapshot_every == 0):
                    self._take_snapshot()
            while len(self.buffer) < batch_size:
                self._collect(hands_per_refresh)
            batch = self.buffer.sample(batch_size, self.rng)
            losses.append(self._grad_step(batch))
            if eval_every and (step + 1) % eval_every == 0:
                snap = evaluate_vs_baseline(
                    self.qnet, n_seeds=eval_seeds, n_hands=eval_hands,
                    mc_sims=eval_mc_sims, seed_start=eval_seed_start)
                snap["step"] = step + 1
                self.history.append(snap)
        return losses


# ---------------------------------------------------------------------------
# Checkpointing (so a trained policy can be reloaded into the dashboard)
# ---------------------------------------------------------------------------

def save_checkpoint(qnet, path, hidden=64, input_dim=None, feature_mode="base",
                    history=None, meta=None):
    """
    Save a Q-network checkpoint the dashboard / `load_checkpoint` can reload.

    Stores the weights plus the architecture (input_dim, hidden), the
    `feature_mode` the policy expects, and the optional learning-curve `history`
    (the list of `evaluate_vs_baseline` snapshots from `SelfPlayTrainer.train`).
    """
    _require_torch()
    if input_dim is None:
        input_dim = qnet.net[0].in_features
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({
        "state_dict": qnet.state_dict(),
        "input_dim": int(input_dim),
        "hidden": int(hidden),
        "feature_mode": feature_mode,
        "history": list(history or []),
        "meta": dict(meta or {}),
    }, path)
    return path


def load_checkpoint(path):
    """
    Load a checkpoint saved by `save_checkpoint`.

    Returns (qnet, ckpt) where `ckpt` is the full dict (incl. 'history' and
    'feature_mode'), so callers can both run the net and chart its learning curve.
    """
    _require_torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    qnet = QNetwork(input_dim=ckpt["input_dim"], hidden=ckpt["hidden"])
    qnet.load_state_dict(ckpt["state_dict"])
    qnet.eval()
    return qnet, ckpt


def save_trainer_checkpoint(trainer, path, meta=None):
    """Convenience: checkpoint a SelfPlayTrainer's live policy + learning curve."""
    _require_torch()
    qnet = trainer.qnet
    return save_checkpoint(
        qnet, path,
        hidden=qnet.net[0].out_features,
        input_dim=qnet.net[0].in_features,
        feature_mode=getattr(trainer, "feature_mode", "base"),
        history=getattr(trainer, "history", []),
        meta=meta,
    )
