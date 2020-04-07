import pickle
import multiprocessing
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import time
import copy

from src.logger import Logger
from src.tictactoe import TicTacToe
from src.agent import Agent, ExpertPlayer
from src.experiment import train, test


# Train Parallel
def train_in_pool(agents):
    # separate agents
    train_iter = agents[0]
    a1 = agents[1]
    a2 = agents[2]

    # Train
    sys.stdout = open(os.devnull, 'w')
    train(train_iter, a1, a2, random=True)
    sys.stdout = sys.__stdout__
    #a1.set_exploration(0)
    return a1


# Test Parallel
def test_in_pool(agents):
    # separate agents
    test_iter = agents[0]
    a1 = agents[1]
    a2 = agents[2]

    # Test
    test(test_iter, a1, a2)


# Get the policymap average
def dictionary_means(dict_list):
    df = pd.DataFrame(dict_list)
    df_means = df.mean()

    result = defaultdict(float)
    for key, val in zip(df_means.index, df_means[:]):
        result[key] = val
    return df_means


# TESTING PARALLEL AGAINST DUMMY AGENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Train Agents in a list
def train_parallel(input_list, against_dummy = True):

    # threads
    train_iter = input_list[0]
    agent_list = input_list[1:]
    threads = len(agent_list)

    # train and test agents
    agent_list = [(train_iter, a, Agent(2, 0.8, dummy=True)) for a in agent_list]

    # Train agents in parallel with random seeds
    trained_agents_list = []
    with multiprocessing.Pool(threads) as p:
        result = p.map(train_in_pool, agent_list)
        trained_agents_list.append(result)

    return trained_agents_list

# Test Agents in a list
def test_parallel(input_list, against_dummy=True):
    # # List of (trained + meta agents, test_agent)
    # test_agent = Agent(2, 0.8, dummy=False) if against_dummy==True else ExpertPlayer(2)
    # agent_list_test = [(train_agent, test_agent) for train_agent in trained_agents_list[0]]
    # agent_list_test.append((meta_agent, test_agent))

    # threads
    test_iter = input_list[0]
    agent_list = input_list[1:]
    threads = len(agent_list)

    agent_list = [(test_iter, a, Agent(2, 0.8, dummy=True)) for a in agent_list]

    # Test all of the agents 
    p = multiprocessing.Pool(threads)
    p.map(test_in_pool, agent_list)


# Combine trained agents into meta agent
def ensemble_agents(trained_agent_list):

    # Build meta agent
    # - new policy mapping
    policy_list = [agent.get_policy() for agent in trained_agent_list]
    df_all = pd.DataFrame(policy_list)
    new_policymap = dictionary_means(policy_list)
    # - meta agent
    meta_agent = Agent(1, 0)
    meta_agent.set_policy(new_policymap)

    return meta_agent
 
# Recursive iteration
def meta_agent_iter(iter=3, num_agents=8):

    # initial agent list
    agent = Agent(1, 0.5)
    agent_list = [copy.deepcopy(agent) for i in range(num_agents)]
    trained_agent_list = train_parallel(agent_list)[0]

    total_time = 0
    # ensemble, copy, train
    for i in range(iter):
        x = time.time()

        # ensemble
        meta_agent = ensemble_agents(trained_agent_list)

        # train vs meta copy
        meta_agent.set_exploration(0.1)
        meta_agent_copy = Agent(2, 0)
        policy = copy.deepcopy(meta_agent.get_policy())
        meta_agent_copy.set_policy(policy)
        train(50000, meta_agent, meta_agent_copy)

        # test and set explorations before/after
        meta_agent.set_exploration(0)
        test(10000, meta_agent, Agent(2, 0.8, dummy=True))
        meta_agent.set_exploration(0.5)

        # train parallel
        trained_agent_list = [Agent(1, 0.5) for i in range(num_agents)]
        for a in trained_agent_list:
            p = copy.deepcopy(meta_agent.get_policy())
            a.set_policy(p)
        trained_agent_list = train_parallel(trained_agent_list)[0]

        y = time.time()
        total_time += y-x
        print('Iteration ', i, '  |  time ', y-x, '  | total time ', total_time)
        print(' ')
    # return final ensemble
    meta_agent_final = ensemble_agents(trained_agent_list)
    return meta_agent_final

def build_meta_agent():
     # iterate through training
    meta_agent_final = meta_agent_iter(iter=200, num_agents=8)
    meta_agent_final.set_exploration(0)

    # test meta_agent
    print(' ')
    print('Final Model:')
    test(10000, meta_agent_final, Agent(2, 0.8, dummy=True))

    pickle.dump(meta_agent_final.policy, open('policies/meta_agent.p', 'wb'))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    agent = Agent(1, 0.5)
    agent_sym = Agent(1, 0.5, symmetric_aware=True)

    agent_list = [1000, agent, agent_sym]

    train_parallel(agent_list)
    test_parallel(agent_list)

    assert agent.get_policy() == agent_sym.get_policy()
