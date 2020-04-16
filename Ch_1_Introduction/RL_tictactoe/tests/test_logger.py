import pytest

from src.tictactoe import TicTacToe
from src.logger import Logger


# Test initial wins
def test_agent_1_initial():
    logger = Logger()
    assert logger.agent_1_wins == 0


# Test initial losses
def test_agent_2_initial():
    logger = Logger()
    assert logger.agent_2_wins == 0


# Test initial ties
def test_agent_2_initial():
    logger = Logger()
    assert logger.ties == 0


# Test winning
def test_agent_win():
    # apply win
    logger = Logger()
    logger.log_agent_win(1)
    
    assert logger.agent_1_wins == 1


# Test losing
def test_agent_lose():
    # apply loss
    logger = Logger()
    logger.log_agent_win(2)

    assert logger.agent_2_wins == 1


# Test ties
def test_agent_tie():
    #apply tie
    logger = Logger()
    logger.log_tie()

    assert logger.ties == 1