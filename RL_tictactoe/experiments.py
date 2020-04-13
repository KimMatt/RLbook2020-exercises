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


# Proxy function for Pool parallel training and testing
def run_in_pool(Dict):
    """ Proxy function for Pool parallel training or testing.

        @param Dict: dictionary matching experiment.train and experiment.test
                     keys: iterations, a1, a2, random, return_both

        @return agent: an Agent or list of Agent object(s) from the training or testing
        @raise ValueError: raises exception if specified method is not 'train' or 'test'
    """
    # suppress print
    stdout_save = sys.stdout
    if not Dict['print']:
        sys.stdout = open(os.devnull, 'w')

    # set train_or_test to train or test
    if Dict['train_or_test'] == 'train':
        train_or_test = train
    elif Dict['train_or_test'] == 'test':
        train_or_test = test
    else:
        raise ValueError('arg["method"] must be "train" or "test"')

    # run train_or_test
    agent = train_or_test(iterations = Dict['iterations'], a1 = Dict['a1'], a2 = Dict['a2'],
                           random = Dict['random'], return_both = Dict['return_both'])

    # restore print suppress after training
    sys.stdout = stdout_save
    return agent


# Run proxy functions in parallel
def run_parallel(iterations=1000, player_list=[], opponent_list=[],
                   return_both=False, random=True, train_or_test='train', print=False):
    """ Runs proxy functions in parallel based on number of agents.

        @param iterations: number of training or testing games to run
        @param player_list: list of Agents to train or test, defaults to 4 Agents
        @param opponent_list: list of Agents for player_list to train or test against
        @param return_both: False returns only players,
                            True returns both players and opponents
        @param random: reset numpy random seed before training or test
        @param method: specify whether proxy function should train or test Agents
        @param print: allow printing of results upon train or test game
        @return agents: player_list after running proxy functions
                        OR zipped list of players and opponents if return_both=True
    """
    # test against random agents if opponent_list = False
    if not player_list:
        player_list = [Agent(1, 0.5, dummy=True) for i in range(4)]
    if not opponent_list:
        opponent_list = [Agent(2, 0.5, dummy=True) for a in player_list]

    # number of processes to run in parallel
    threads = len(player_list)

    # build list of dictionary kwargs for proxy function
    training_testing_list = [{'iterations': iterations, 'a1': player,
                              'a2': opponent, 'random': random,
                              'return_both': return_both, 'train_or_test': train_or_test,
                              'print': print}
                                for player, opponent in zip(player_list, opponent_list)]

    # apply pool map
    result_list = []
    with multiprocessing.Pool(threads) as p:
        results = p.map(run_in_pool, training_testing_list)
        result_list.append(results)

    if train_or_test == 'train':
        return result_list[0]
    else:
        return result_list[0]


# Combine trained agents into meta agent
def ensemble_agents(agent_list, **kwargs):
    """ Build meta agent

        @param agent_list: list of Agents to ensemble using policy means
        @return meta_agent: Agent with policy of agent_list policy averages
    """
    # Build meta agent
    # - new policy mapping
    policy_list = [agent.get_policy() for agent in agent_list]
    df_policies = pd.DataFrame(policy_list)
    df_means = df_policies.mean()
    dict_policies = df_means.to_dict()

    # - meta agent
    meta_agent = Agent(**kwargs)
    meta_agent.set_policy(dict_policies)

    return meta_agent


# Build Meta Agent
def build_meta_agent(steps = 3, train_iterations = 1000, test_iterations = 1000,
                     player_list = [], opponent_list = [],
                     train_random = True, test_random = False,
                     print_train = False, print_test = False, print_meta_test = True,
                     get_scores = False,
                     meta_exploration = 0.5,
                     meta_opponent = Agent(2, 0.5, dummy=True),
                     self_play = False, self_play_iterations = 1000,
                     meta_player_n = 1, meta_symmetric_aware = False):
    """ Build meta agent of ensembled mean policy maps.

        @param steps: number of ensembles to perfo rm
        @param train_iterations: number of training games per step, applies to self play too
        @param test_iterations: number of testing games per step
        @param train_random, test_random:
                    reset numpy random seed before training or test
        @param print_train, print_test, print_meta_test:
                    allow printing of results upon train or test game
        @param get_scores: return meta agent's testing scores
        @param meta_exploration: set exploration of meta agent at each step
        @param meta_opponent: opponent for meta agent to test against
        @param self_play, self_play_iterations: train meta agent against itself between steps
        @return final_meta_agent: ensembled agent
        @return score_list: list of meta_agent's score dictionaries from each step's test
    """
    # Iterate 'step' number of layers to build meta agent
    scores_list = [0] * steps
    for step in range(steps):
        # train initial agents
        player_list = run_parallel(iterations = train_iterations, player_list = player_list,
                                opponent_list = opponent_list, random = train_random,
                                train_or_test = 'train',
                                print = print_train)
        # agent test scores
        if print_test:
            run_parallel(iterations = test_iterations, player_list = player_list,
                        opponent_list = opponent_list, random = test_random,
                        train_or_test = 'test',
                        print = print_test)

        # build meta agent with mean ensemble
        meta_agent = ensemble_agents(player_list, player_n = meta_player_n,
                                     exploration = meta_exploration,
                                     symmetric_aware = meta_symmetric_aware)

        # self train meta_agent
        if self_play:
            # build mirror opponent
            meta_opponent = Agent(2, 0)
            meta_opponent.set_policy(meta_agent.get_policy())
            meta_opponent.set_exploration(meta_agent.exploration)

            # train against mirror opponent
            meta_agent = run_in_pool(dict(iterations = train_iterations,
                                    a1 = meta_agent, a2 = meta_opponent,
                                    random = train_random,
                                    train_or_test = 'train',
                                    print = print_train,
                                    return_both=False))

        # meta agent test score
        if print_meta_test:
            print('Step', step, 'Meta Agent Testing --- ')
            scores = test(test_iterations, meta_agent, meta_opponent)
            print(' ')
        if get_scores:
            scores_list[step] = scores


        # deep copy meta agents into list
        player_list = [copy.deepcopy(meta_agent) for p in player_list]

    # build final meta agent
    final_meta_agent = ensemble_agents(player_list, player_n = meta_player_n,
                                exploration = 0,
                                symmetric_aware = meta_symmetric_aware)

    # Report final testing score
    print('Final Meta Agent Test')
    test(test_iterations, final_meta_agent, meta_opponent)

    # return meta agent and (optional) scores
    if get_scores:
        return final_meta_agent, scores_list
    else:
        return final_meta_agent


# Build and compare Agents
def compare_agents(steps = 100, train_iterations = 1000, test_iterations = 1000,
            player_list = [], opponent_list = [],
            train_random = True, test_random = False,
            print_train = False, print_test = False,
            get_scores = True):
    """ Build and compare performances of Agents

        @param steps: number of ensembles to perfo rm
        @param train_iterations: number of training games per step
        @param test_iterations: number of testing games per step
        @param train_random, test_random:
                    reset numpy random seed before training or test
        @param print_train, print_test:
                    allow printing of results upon train or test game
        @param get_scores: return testing scores
        @return player_list: list of trained agents from input
        @return score_list: list of list of dictionary scores for each player at test iterations
    """
    # Iterate steps of training
    score_list = [0] * steps
    for step in range(steps):
        # train agents
        player_list = run_parallel(iterations = train_iterations,
                                  player_list = player_list, opponent_list = opponent_list,
                                  return_both = False,
                                  random = train_random, train_or_test = 'train',
                                  print = print_train)

        # test agents
        scores = run_parallel(iterations = test_iterations,
                              player_list = player_list, opponent_list = opponent_list,
                              return_both = False,
                              random = test_random, train_or_test = 'test',
                              print = print_test)

        # total score list
        score_list[step] = scores

    if get_scores:
        return player_list, score_list
    else:
        return player_list


# # MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    # Build meta agent ----------
    meta_agent, meta_scores = build_meta_agent(get_scores=True)


    # Comparison -----------
    # - players [agent, symmetric agent, self-play agent]
    player_list = [Agent(1, 0.5), Agent(1, 0.5, symmetric_aware=True), Agent(1, 0.5)]
    # - opponents [random agent, randon agent, self-play agent opponent]
    self_agent = Agent(2, 0.5)
    self_agent.set_exploration(player_list[2].exploration)
    opponent_list = [Agent(2, 0.5, dummy=True), Agent(2, 0.5, dummy=True), self_agent]

    # run comparisons
    print(' ')
    print('Running Comparisons...')
    print(' ')
    comparison_agent_list = compare_agents(steps = 1, train_iterations = 50000,
                                           player_list = player_list,
                                           opponent_list = opponent_list,
                                           get_scores = False)


    # Outputs
    print('META AGENT TEST ---- ')
    test(10000, meta_agent, Agent(2, 0.5, dummy=True))
    print(' ')

    scores = run_parallel(player_list = comparison_agent_list,
                          train_or_test = 'test',
                          print=False)
    print('COMPARISON TESTS SCORES vs Random Agents (50K iterations) ----- ')
    print('Control Agent:', scores[0])
    print('Symmetrically Aware Agent:', scores[1])
    print('Self-Play Agent:', scores[2])