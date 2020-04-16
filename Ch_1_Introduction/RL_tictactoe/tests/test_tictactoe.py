import pytest

from src.tictactoe import TicTacToe
from src.logger import Logger

def test_play_move():
    logger = Logger()
    game = TicTacToe(logger)
    game.play_move(1,0)
    assert game.game_state[0] == 1
