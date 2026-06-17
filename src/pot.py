"""
pot.py
------
Pot and side pot management for Texas Hold'em.

Side pots arise when one or more players are all-in for less than
the full bet. This module handles the calculation and distribution
of all pots correctly, including split pots (ties).
"""


class SidePot:
    """
    Represents a single pot (main or side) in a poker hand.

    Each pot has an amount and a set of eligible players.
    A player is eligible for a pot only if they contributed to it.
    """

    def __init__(self, amount, eligible_player_ids):
        """
        Initialize a side pot.

        Args:
            amount (int): Chips in this pot.
            eligible_player_ids (set): Player IDs who can win this pot.
        """
        self.amount = amount
        self.eligible_player_ids = set(eligible_player_ids)

    def __repr__(self):
        return (
            f"SidePot(amount={self.amount}, "
            f"eligible={self.eligible_player_ids})"
        )


class PotManager:
    """
    Manages chip collection and pot calculation across a hand.

    Tracks player contributions per round, then calculates the
    correct main pot and side pots at showdown.
    """

    def __init__(self):
        """Initialize an empty pot manager."""
        self._contributions = {}  # {player_id: total_chips_invested}
        self._folded_players = set()
        self._pots = []  # List[SidePot], populated by calculate_pots()

    def add_contribution(self, player_id, amount):
        """
        Record chips contributed by a player.

        Args:
            player_id (int): Player ID.
            amount (int): Chips contributed in this action.
        """
        if amount < 0:
            raise ValueError(
                f"Contribution amount must be non-negative, got {amount}."
            )
        self._contributions[player_id] = (
            self._contributions.get(player_id, 0) + amount
        )

    def mark_folded(self, player_id):
        """
        Mark a player as folded.

        Folded players cannot win pots — their contributions remain
        in the pot but their eligibility is removed.

        Args:
            player_id (int): Player ID.
        """
        self._folded_players.add(player_id)

    def total_pot(self):
        """
        Return the total chips in all pots.

        Returns:
            int: Sum of all contributions.
        """
        return sum(self._contributions.values())

    def calculate_pots(self):
        """
        Calculate main pot and side pots based on player contributions.

        Side pots are created whenever an all-in player contributed
        less than the maximum bet. Uses the standard poker algorithm:
        process players from lowest to highest total investment.

        Returns:
            list[SidePot]: Ordered list of pots (main first).
        """
        if not self._contributions:
            return []

        # Work with a copy to avoid mutating state
        remaining = dict(self._contributions)
        active_players = set(remaining.keys())
        pots = []

        while remaining:
            # Find the minimum contribution among remaining players
            min_contrib = min(remaining.values())
            if min_contrib == 0:
                # Remove players who have nothing left to contribute to
                remaining = {pid: amt for pid, amt in remaining.items() if amt > 0}
                active_players = set(remaining.keys())
                continue

            # Each player contributes up to min_contrib to this level
            pot_amount = 0
            for pid in list(remaining.keys()):
                contribution = min(remaining[pid], min_contrib)
                pot_amount += contribution
                remaining[pid] -= contribution

            # Eligible = contributed to this level AND not folded
            eligible = active_players - self._folded_players

            if pot_amount > 0:
                pots.append(SidePot(pot_amount, eligible))

            # Remove players who have exhausted their contribution
            players_at_min = {pid for pid, amt in remaining.items() if amt == 0}
            for pid in players_at_min:
                del remaining[pid]
            active_players = set(remaining.keys())

        self._pots = pots
        return pots

    def distribute(self, winner_resolver):
        """
        Distribute pots to winners using a resolver function.

        Args:
            winner_resolver (callable): Takes (eligible_player_ids) and
                returns a list of winner IDs (handles ties internally).

        Returns:
            dict: {player_id: chips_won}
        """
        pots = self.calculate_pots()
        winnings = {}

        for pot in pots:
            winners = winner_resolver(pot.eligible_player_ids)
            if not winners:
                # Edge case: all eligible players folded — pot goes to last
                # active player. Should not occur in well-formed game logic.
                raise RuntimeError(
                    f"No winners found for pot {pot}. "
                    "This indicates a game logic error."
                )

            share, remainder = divmod(pot.amount, len(winners))

            for winner_id in winners:
                winnings[winner_id] = winnings.get(winner_id, 0) + share

            # Remainder chips (from indivisible split) go to first winner
            # (standard poker rule: first player left of the dealer)
            if remainder > 0:
                winnings[winners[0]] = winnings.get(winners[0], 0) + remainder

        return winnings

    def reset(self):
        """Reset the pot manager for a new hand."""
        self._contributions = {}
        self._folded_players = set()
        self._pots = []

    def __repr__(self):
        return f"PotManager(total={self.total_pot()}, pots={self._pots})"
