import pickle

from src.logger import Logger
from src.tictactoe import TicTacToe
from src.agent import Agent
from src.experiment import train, test

if __name__ == "__main__":

    AGENT_1 = Agent(1, 0.8)
    AGENT_2 = Agent(2, 0.8, dummy=True)
    train(50000, AGENT_1, AGENT_2)
    AGENT_1.set_exploration(0)
    AGENT_3 = Agent(2, 0.5, dummy=True)
    test(50000, AGENT_1, AGENT_3)
    pickle.dump(AGENT_1.policy, open("policies/dummy_trained_agent.py", "wb"))

    AGENT_1 = Agent(1, 0.8)
    AGENT_2 = Agent(2, 0.8)
    WINNER = train(50000, AGENT_1, AGENT_2)
    WINNER.player_n = 1
    WINNER.set_exploration(0)
    AGENT_3 = Agent(2, 0.5, dummy=True)
    test(50000, WINNER, AGENT_3)
    pickle.dump(WINNER.policy, open("policies/agent_trained_agent.py", "wb"))

    print("Now with symmetric awareness")
    AGENT_1 = Agent(1, 0.05, symmetric_aware=True)
    AGENT_2 = Agent(2, 0.05)
    WINNER = train(50000, AGENT_1, AGENT_2)
    WINNER.player_n = 1
    WINNER.set_exploration(0)
    AGENT_3 = Agent(2, 0.05, dummy=True)
    test(50000, WINNER, AGENT_3)

    print("Now with self play")
    AGENT_1 = Agent(1, 0.05)
    AGENT_2 = Agent(2, 0.05)
    AGENT_2.set_policy(AGENT_1.policy)
    WINNER = train(50000, AGENT_1, AGENT_2)
    WINNER.player_n = 1
    WINNER.set_exploration(0)
    AGENT_3 = Agent(2, 0.05, dummy=True)
    test(50000, WINNER, AGENT_3)

    print("Now with self play & symmetric awareness")
    AGENT_1 = Agent(1, 0.05)
    AGENT_2 = Agent(2, 0.05)
    AGENT_2.set_policy(AGENT_1.policy)
    WINNER = train(50000, AGENT_1, AGENT_2)
    WINNER.player_n = 1
    WINNER.set_exploration(0)
    AGENT_3 = Agent(2, 0.05, dummy=True)
    test(50000, WINNER, AGENT_3)
