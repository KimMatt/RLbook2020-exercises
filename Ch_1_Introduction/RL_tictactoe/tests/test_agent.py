# Dependencies
import pytest

from src.logger import Logger 
from src.agent import Agent 

 # Test empty game_state normalization
def test_normalize_game_state():
    agent = Agent(player_n=1, exploration=0.5) 
    initial_game_state = [0 for i in range(9)]
    
    game_state = agent.normalized_game_state(initial_game_state)
    assert game_state == initial_game_state


# Test 
def test_normalize_game_state_player():
    agent = Agent(player_n=1, exploration=0.5)
    initial_game_state = [1, 0, 0, 1, 0, 0, 0, 0, 1]

    game_state = agent.normalized_game_state(initial_game_state)
    assert game_state == initial_game_state 


# Test enemy game_state normalization
def test_normalize_game_state_enemy():
    agent = Agent(player_n=1, exploration=0.5)
    initial_game_state = [2, 0, 0, 0, 2, 2, 2, 2, 2]
    
    game_state = agent.normalized_game_state(initial_game_state)
    assert game_state == [-1, 0, 0, 0, -1, -1, -1, -1, -1]


# Test with both players having turns
def test_normalize_game_state_both():
    agent = Agent(player_n=1, exploration=0.5)
    initial_game_state = [1, 2, 0, 1, 0, 0, 0, 2]

    game_state = agent.normalized_game_state(initial_game_state)
    assert game_state == [1, -1, 0, 1, 0, 0, 0, -1]
