"""
main.py
-------
Terminal entry point for the poker-quant Texas Hold'em simulator.

Runs a Human vs. Bot session. Configure players and blind structure here.
"""

import sys
import os

# Ensure the project root is in the path regardless of where this is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.player import HumanPlayer, BotPlayer
from src.game import GameEngine

# Session configuration
STARTING_STACK = 1000
SMALL_BLIND = 10
BIG_BLIND = 20
DEFAULT_HANDS = 10


def run_session(num_hands=DEFAULT_HANDS):
    """
    Run a multi-hand Texas Hold'em session.

    Args:
        num_hands (int): Number of hands to play.
    """
    print("\n  Welcome to poker-quant: Texas Hold'em Simulator")
    print("  ================================================\n")

    human_name = input("  Enter your name: ").strip() or "Human"

    players = [
        HumanPlayer(player_id=1, name=human_name, stack=STARTING_STACK),
        BotPlayer(
            player_id=2, name="Bot-Balanced",
            stack=STARTING_STACK,
            tight_threshold=0.4,
            aggression=0.5
        ),
    ]

    engine = GameEngine(
        players=players,
        small_blind=SMALL_BLIND,
        big_blind=BIG_BLIND,
        verbose=True
    )

    for hand_num in range(1, num_hands + 1):
        active = [p for p in players if p.stack > 0]
        if len(active) < 2:
            print("\n  A player has been eliminated. Session over.")
            break

        try:
            engine.play_hand()
        except KeyboardInterrupt:
            print("\n\n  Session interrupted by user.")
            break

        play_again = input("\n  Continue to next hand? [Enter / q to quit]: ").strip()
        if play_again.lower() == "q":
            break

    print("\n  === Final Stacks ===")
    for p in players:
        print(f"  {p.name:14s}: {p.stack} chips")
    print()


if __name__ == "__main__":
    run_session()
