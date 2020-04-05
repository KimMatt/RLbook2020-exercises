import pickle
import multiprocessing
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import time

from src.logger import Logger
from src.tictactoe import TicTacToe
from src.agent import Agent, ExpertPlayer
from src.experiment import train, test


# Train Parallel
def train_parallel(agents):
    # separate agents
    a1 = agents[0]
    a2 = agents[1]

    # Train
    train(50000000, a1, a2, random=True)
    a1.set_exploration(0)
    return a1


# Test Parallel
def test_parallel(agents):
    #separate agents
    a1 = agents[0]
    a2 = agents[1]

    # Test
    test(100000, a1, a2)


# Get the policymap average
def dictionary_means(dict_list):
    df = pd.DataFrame(dict_list)
    df_means = df.mean()

    result = defaultdict(float)
    for key, val in zip(df_means.index, df_means[:]):
        result[key] = val
    return df_means


# TESTING PARALLEL AGAINST DUMB AGENTS
def train_meta_agent(against_dummy = True):
    # use half threads as processes
    threads = int(os.cpu_count() / 2)


    # Build training agents  
    agent_list_train = []
    for i in range(threads):
        iter_list = (Agent(1, 0.8),
                     Agent(2, 0.8, dummy=True) if against_dummy==True 
                        else ExpertPlayer(2)
                    )
        agent_list_train.append(iter_list)


    # Train agents in parallel with random seeds
    trained_agents_list = []
    with multiprocessing.Pool(threads) as p:
        result = p.map(train_parallel, agent_list_train)
        trained_agents_list.append(result)


    # Build meta agent
    # - new policy mapping
    policy_list = [agent.get_policy() for agent in trained_agents_list[0]]
    df_all = pd.DataFrame(policy_list)
    new_policymap = dictionary_means(policy_list)
    # - meta agent
    meta_agent = Agent(1, 0)
    meta_agent.set_policy(new_policymap)


    # List of (trained + meta agents, test_agent)
    test_agent = Agent(2, 0.8, dummy=False) if against_dummy==True else ExpertPlayer(2)
    agent_list_test = [(train_agent, test_agent) for train_agent in trained_agents_list[0]]
    agent_list_test.append((meta_agent, test_agent))


    # Test all of the agents 
    p = multiprocessing.Pool(len(agent_list_test))
    p.map(test_parallel, agent_list_test)

    return meta_agent
 







# MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":


    meta_agent = train_meta_agent()
    pickle.dump(meta_agent.policy, open('policies/meta_agent.p', 'wb'))

    test(100000, meta_agent, ExpertPlayer(2))