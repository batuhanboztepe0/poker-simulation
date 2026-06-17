"""
monte_carlo.py
--------------
Monte Carlo equity estimator for Texas Hold'em.

Estimates a player's probability of winning a hand by simulating
the remaining community cards N times and evaluating all hands
at each simulation.

Equity formula:
    equity = (wins + 0.5 * ties) / N

The 0.5 tie weighting reflects the economic reality of split pots:
a tie is worth half a win in a heads-up pot, and 1/k in a k-way tie.

Usage:
    engine = MonteCarloEngine(n_simulations=1000)
    equity = engine.estimate_equity(
        hero_hole_cards,
        opponent_hole_cards_list,
        community_cards
    )
"""

import random
from src.card import Card, Deck, RANKS, SUITS
from src.hand_evaluator import HandEvaluator

DEFAULT_N_SIMULATIONS = 1000
MIN_N_SIMULATIONS = 100
MAX_N_SIMULATIONS = 100_000

# Pre-flop equity lookup for common hand classes (heads-up approximation).
# Values derived from standard poker equity tables.
PREFLOP_EQUITY_TABLE = {
    # Premium pairs
    ("A", "A", False): 0.85,
    ("K", "K", False): 0.82,
    ("Q", "Q", False): 0.80,
    ("J", "J", False): 0.77,
    ("T", "T", False): 0.75,
    # Mid pairs
    ("9", "9", False): 0.72,
    ("8", "8", False): 0.69,
    ("7", "7", False): 0.66,
    ("6", "6", False): 0.63,
    ("5", "5", False): 0.60,
    ("4", "4", False): 0.57,
    ("3", "3", False): 0.53,
    ("2", "2", False): 0.50,
    # Suited broadways
    ("A", "K", True): 0.67,
    ("A", "Q", True): 0.66,
    ("A", "J", True): 0.65,
    ("A", "T", True): 0.63,
    ("K", "Q", True): 0.63,
    ("K", "J", True): 0.62,
    ("Q", "J", True): 0.60,
    ("K", "T", True): 0.61,
    ("Q", "T", True): 0.59,
    ("J", "T", True): 0.58,
    # Offsuit broadways
    ("A", "K", False): 0.65,
    ("A", "Q", False): 0.64,
    ("A", "J", False): 0.63,
    ("A", "T", False): 0.61,
    ("K", "Q", False): 0.61,
    ("K", "J", False): 0.60,
    ("Q", "J", False): 0.58,
    ("K", "T", False): 0.59,
    ("Q", "T", False): 0.57,
    ("J", "T", False): 0.56,
}

PREFLOP_EQUITY_DEFAULT = 0.50


class MonteCarloEngine:
    """
    Estimates hand equity via Monte Carlo simulation.

    For each simulation:
        1. Draw remaining community cards from the reduced deck.
        2. Assign hole cards to any unknown opponents.
        3. Evaluate all players' hands.
        4. Record win/tie/loss for the hero.

    Equity = (wins + 0.5 * ties) / N.

    Passing opponent_holes=[] is equivalent to calling
    estimate_equity_unknown_opponents with n_opponents=1.
    """

    def __init__(self, n_simulations=DEFAULT_N_SIMULATIONS, rng=None):
        """
        Initialize the Monte Carlo engine.

        Args:
            n_simulations (int): Simulations per equity call.
                Range: 100-100000. Default 1000.
            rng (random.Random | None): Random source for card sampling.
                When None, the module-level `random` is used (unseeded).
                Inject a seeded `random.Random(seed)` for reproducibility.

        Raises:
            ValueError: If n_simulations is out of valid range.
        """
        if not (MIN_N_SIMULATIONS <= n_simulations <= MAX_N_SIMULATIONS):
            raise ValueError(
                f"n_simulations must be between {MIN_N_SIMULATIONS} "
                f"and {MAX_N_SIMULATIONS}, got {n_simulations}."
            )
        self.n_simulations = n_simulations
        self._evaluator = HandEvaluator()
        self._rng = rng if rng is not None else random

    def estimate_equity(self, hero_hole, opponent_holes, community_cards):
        """
        Estimate hero's equity against one or more opponents.

        Args:
            hero_hole (list[Card]): Hero's 2 hole cards.
            opponent_holes (list[list[Card]]): Each opponent's hole cards.
                - Pass a list of [Card, Card] pairs for known opponents.
                - Pass [] or [[]] for one unknown opponent (default).
                - Pass [[], [], []] for three unknown opponents.
            community_cards (list[Card]): 0-5 known community cards.

        Returns:
            float: Equity in [0, 1].

        Raises:
            ValueError: On invalid card counts or duplicates.
        """
        self._validate_inputs(hero_hole, opponent_holes, community_cards)

        cards_needed = 5 - len(community_cards)

        # Normalise opponent_holes: treat [] as one unknown opponent
        if not opponent_holes:
            opponent_holes = [[]]

        # Pre-flop with no community cards and exactly ONE unknown opponent:
        # use the heads-up lookup table. The table is a heads-up approximation,
        # so it is only valid for a single opponent — with 2+ opponents we must
        # fall through to simulation (e.g. AA is ~0.85 heads-up but ~0.49 vs 5).
        if (cards_needed == 5
                and len(opponent_holes) == 1
                and all(not opp for opp in opponent_holes)):
            return self._preflop_equity_lookup(hero_hole)

        # Build draw pool: all 52 cards minus known cards
        known_ints = set(c.treys_int for c in hero_hole)
        known_ints.update(c.treys_int for c in community_cards)

        known_opponents = []
        n_unknown_opponents = 0
        for opp in opponent_holes:
            if opp and len(opp) == 2:
                known_ints.update(c.treys_int for c in opp)
                known_opponents.append(opp)
            else:
                n_unknown_opponents += 1

        draw_pool = self._build_draw_pool(known_ints)

        equity_sum = 0.0

        for _ in range(self.n_simulations):
            sim_community, sim_opponents = self._sample_simulation(
                draw_pool, community_cards, known_opponents,
                cards_needed, n_unknown_opponents
            )
            equity_sum += self._evaluate_simulation(
                hero_hole, sim_opponents, sim_community
            )

        equity = equity_sum / self.n_simulations
        return round(equity, 6)

    def estimate_equity_unknown_opponents(
        self, hero_hole, n_opponents, community_cards
    ):
        """
        Estimate equity when all opponent hole cards are unknown.

        Randomly assigns hole cards to opponents from the draw pool
        each simulation. Most common real-game scenario.

        Args:
            hero_hole (list[Card]): Hero's 2 hole cards.
            n_opponents (int): Number of active opponents (>= 1).
            community_cards (list[Card]): 0-5 known community cards.

        Returns:
            float: Equity estimate in [0, 1].

        Raises:
            ValueError: If n_opponents < 1.
        """
        if n_opponents < 1:
            raise ValueError(
                f"n_opponents must be >= 1, got {n_opponents}."
            )
        return self.estimate_equity(
            hero_hole,
            [[] for _ in range(n_opponents)],
            community_cards
        )

    def estimate_equity_vs_range(self, hero_hole, opponent_ranges,
                                 community_cards):
        """
        Estimate hero equity when each opponent's hand is drawn from a range.

        This is the belief-conditioned equity used in Phase 3: instead of
        assuming each opponent holds a uniformly random hand, each simulation
        samples one hand per opponent from that opponent's candidate range
        (a list of [Card, Card] pairs, e.g. produced by
        BeliefState.range_sample), completes the board, and scores hero's pot
        share. A tighter (stronger) range therefore lowers hero's equity.

        Args:
            hero_hole (list[Card]): Hero's 2 hole cards.
            opponent_ranges (list[list[list[Card]]]): One candidate range per
                opponent; each range is a non-empty list of [Card, Card] pairs.
            community_cards (list[Card]): 0-5 known community cards.

        Returns:
            float: Equity in [0, 1]. Falls back to the uniform unknown-opponent
                estimate if no usable range is supplied.
        """
        if len(hero_hole) != 2:
            raise ValueError(
                f"hero_hole must contain exactly 2 cards, got {len(hero_hole)}."
            )
        if not (0 <= len(community_cards) <= 5):
            raise ValueError(
                f"community_cards must have 0-5 cards, got {len(community_cards)}."
            )
        if not opponent_ranges or any(not r for r in opponent_ranges):
            return self.estimate_equity_unknown_opponents(
                hero_hole, max(1, len(opponent_ranges)), community_cards
            )

        known_ints = set(c.treys_int for c in hero_hole)
        known_ints.update(c.treys_int for c in community_cards)
        base_pool = self._build_draw_pool(known_ints)
        cards_needed = 5 - len(community_cards)

        equity_sum = 0.0
        valid = 0
        for _ in range(self.n_simulations):
            used = set(known_ints)
            chosen = []
            ok = True
            for candidate_range in opponent_ranges:
                hand = None
                for _attempt in range(12):
                    h = self._rng.choice(candidate_range)
                    a, b = h[0].treys_int, h[1].treys_int
                    if a != b and a not in used and b not in used:
                        hand = h
                        break
                if hand is None:
                    ok = False
                    break
                used.add(hand[0].treys_int)
                used.add(hand[1].treys_int)
                chosen.append(hand)
            if not ok:
                continue

            board_pool = [c for c in base_pool if c.treys_int not in used]
            if len(board_pool) < cards_needed:
                continue
            board_fill = (self._rng.sample(board_pool, cards_needed)
                          if cards_needed else [])
            sim_community = list(community_cards) + board_fill
            equity_sum += self._evaluate_simulation(
                hero_hole, chosen, sim_community
            )
            valid += 1

        if valid == 0:
            return self.estimate_equity_unknown_opponents(
                hero_hole, len(opponent_ranges), community_cards
            )
        return round(equity_sum / valid, 6)

    # ------------------------------------------------------------------
    # Internal simulation helpers
    # ------------------------------------------------------------------

    def _sample_simulation(
        self, draw_pool, known_community, known_opponents,
        cards_needed, n_unknown_opponents
    ):
        """
        Draw a random board completion and unknown opponent cards.

        Args:
            draw_pool (list[Card]): Available cards to draw.
            known_community (list[Card]): Already-dealt community cards.
            known_opponents (list[list[Card]]): Known opponent hole pairs.
            cards_needed (int): Community cards still to be dealt.
            n_unknown_opponents (int): Opponents needing random cards.

        Returns:
            tuple: (sim_community, all_opponent_hole_lists)
        """
        total_needed = cards_needed + n_unknown_opponents * 2
        sample = self._rng.sample(draw_pool, total_needed)

        sim_community = list(known_community) + sample[:cards_needed]
        sampled_hole_cards = sample[cards_needed:]

        sim_opponents = list(known_opponents)
        for i in range(n_unknown_opponents):
            sim_opponents.append(
                [sampled_hole_cards[i * 2], sampled_hole_cards[i * 2 + 1]]
            )

        return sim_community, sim_opponents

    def _evaluate_simulation(self, hero_hole, opponent_holes, community):
        """
        Evaluate one simulation and return hero's pot share for it.

        The share is the fraction of the pot hero wins in this showdown:
            - 1.0 if hero has the sole best hand,
            - 1/k if hero ties with (k-1) opponents (a k-way split),
            - 0.0 if any opponent beats hero.

        Crediting 1/k (not a flat 0.5) is essential for multi-way pots:
        a 3-way tie is worth 1/3 to hero, not 1/2.

        Args:
            hero_hole (list[Card]): Hero's hole cards.
            opponent_holes (list[list[Card]]): All opponents' hole cards.
            community (list[Card]): Exactly 5 community cards.

        Returns:
            float: Hero's pot share in [0, 1] for this simulation.

        Raises:
            RuntimeError: If opponent_holes is empty (logic error upstream).
        """
        if not opponent_holes:
            raise RuntimeError(
                "_evaluate_simulation called with no opponents. "
                "Ensure at least one opponent is present in simulation."
            )

        hero_score = self._evaluator.evaluate(hero_hole, community)
        opponent_scores = [
            self._evaluator.evaluate(opp, community)
            for opp in opponent_holes
        ]
        best_opponent_score = min(opponent_scores)

        if hero_score < best_opponent_score:
            return 1.0
        if hero_score > best_opponent_score:
            return 0.0

        # Tie at the top: split among hero + every opponent sharing the best.
        tied_opponents = sum(
            1 for s in opponent_scores if s == hero_score
        )
        return 1.0 / (tied_opponents + 1)

    def _build_draw_pool(self, known_treys_ints):
        """
        Build the drawable card pool by excluding known cards.

        Args:
            known_treys_ints (set[int]): treys integers of known cards.

        Returns:
            list[Card]: Cards available to draw.
        """
        pool = []
        for suit in SUITS:
            for rank in RANKS:
                card = Card(rank, suit)
                if card.treys_int not in known_treys_ints:
                    pool.append(card)
        return pool

    def _preflop_equity_lookup(self, hero_hole):
        """
        Return pre-flop equity from lookup table.

        Falls back to PREFLOP_EQUITY_DEFAULT for hands not in the table.

        Args:
            hero_hole (list[Card]): Hero's 2 hole cards.

        Returns:
            float: Pre-flop equity estimate.
        """
        c1, c2 = hero_hole
        suited = c1.suit == c2.suit
        r1, r2 = c1.rank, c2.rank

        # Normalise: higher rank first
        if RANKS.index(r1) < RANKS.index(r2):
            r1, r2 = r2, r1

        key = (r1, r2, suited)
        return PREFLOP_EQUITY_TABLE.get(key, PREFLOP_EQUITY_DEFAULT)

    def _validate_inputs(self, hero_hole, opponent_holes, community_cards):
        """
        Validate inputs before simulation.

        Raises:
            ValueError: On invalid card counts or duplicate cards.
        """
        if len(hero_hole) != 2:
            raise ValueError(
                f"hero_hole must contain exactly 2 cards, got {len(hero_hole)}."
            )
        if not (0 <= len(community_cards) <= 5):
            raise ValueError(
                f"community_cards must have 0-5 cards, got {len(community_cards)}."
            )

        all_known = list(hero_hole) + list(community_cards)
        for opp in opponent_holes:
            if opp:
                all_known.extend(opp)

        treys_ints = [c.treys_int for c in all_known]
        if len(treys_ints) != len(set(treys_ints)):
            raise ValueError(
                "Duplicate cards detected in inputs. "
                "Each card may appear only once."
            )
