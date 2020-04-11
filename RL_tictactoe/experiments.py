import pickle
import multiprocessing
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import time
import copy

import seaborn as sns 
import matplotlib.pyplot  as plt

from src.logger import Logger
from src.tictactoe import TicTacToe
from src.agent import Agent, ExpertPlayer
from src.experiment import train, test


# Train Parallel
def train_in_pool(iter_agent_list):

    num_iter = iter_agent_list[0]
    a1 = iter_agent_list[1]
    a2 = iter_agent_list[2]


    # Train
    sys.stdout = open(os.devnull, 'w')
    train(num_iter, a1, a2, random=True)
    sys.stdout = sys.__stdout__
    #a1.set_exploration(0)
    return a1


# Test Parallel
def test_in_pool(iter_agent_list):

    num_iter = iter_agent_list[0]
    a1 = iter_agent_list[1]
    a2 = iter_agent_list[2]

    # Test
    scores = test(1000, a1, a2)
    return scores


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
def train_parallel(num_iter, agent_list, test_list = False, against_dummy = True):

    # threads
    threads = len(agent_list)

    if test_list:
        iter_agent_list = [(num_iter, a, t) for a,t in zip(agent_list, test_list)]

    # train and test agents
    iter_agent_list = [(num_iter, a, Agent(2, 0.8, dummy=True)) for a in agent_list]

    # Train agents in parallel with random seeds
    trained_agents_list = []
    with multiprocessing.Pool(threads) as p:
        result = p.map(train_in_pool, iter_agent_list)
        trained_agents_list.append(result)

    # close processes
    p.close()
    p.terminate()
    p.join()

    return trained_agents_list

# Test Agents in a list
def test_parallel(num_iter, agent_list, test_list = [], against_dummy=True, agent=None):
    # # List of (trained + meta agents, test_agent)
    # test_agent = Agent(2, 0.8, dummy=False) if against_dummy==True else ExpertPlayer(2)
    # agent_list_test = [(train_agent, test_agent) for train_agent in trained_agents_list[0]]
    # agent_list_test.append((meta_agent, test_agent))

    # threads
    threads = len(agent_list)

    # build testing opponents
    if test_list:
        iter_agent_list = [(num_iter, a, t) for a,t in zip(agent_list, test_list)]

    else:
        iter_agent_list = [(num_iter, a, Agent(2, 0.8, dummy=against_dummy)) for a in agent_list]

    # Test all of the agents 
    p = multiprocessing.Pool(threads)
    scores = p.map(test_in_pool, iter_agent_list)

    # close processes
    p.close()
    p.terminate()
    p.join()
    
    return scores


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


# Train aware and not aware agents, save to csv
def compare_symmetric_learning_rate(iterations):
       # agents to compare
    agent = Agent(1, 0.5)
    agent_sym = Agent(1, 0.5, symmetric_aware=True)
    agent_list = [agent, agent_sym]

    ## FIRST ITERATION
    #train
    trained_list = train_parallel(1000, agent_list)[0]
    # get score
    trained_list[0].set_exploration(0)
    trained_list[1].set_exploration(0)
    scores = test_parallel(10000, trained_list)

        # append scores
    aware_win = [scores[1]['wins']]
    aware_loss = [scores[1]['losses']]
    aware_tie = [scores[1]['ties']]

    not_aware_win = [scores[0]['wins']]
    not_aware_loss = [scores[0]['losses']]
    not_aware_tie = [scores[0]['ties']]

    for i in range(iterations-1):
        print(i, '-----------------------')
        # get score
        trained_list[0].set_exploration(0)
        trained_list[1].set_exploration(0)
        scores = test_parallel(10000, trained_list)

        # append scores
        aware_win.append(scores[1]['wins'])
        aware_loss.append(scores[1]['losses'])
        aware_tie.append(scores[1]['ties'])

        not_aware_win.append(scores[0]['wins'])
        not_aware_loss.append(scores[0]['losses'])
        not_aware_tie.append(scores[0]['ties'])

        # train
        trained_list[0].set_exploration(0.5)
        trained_list[1].set_exploration(0.5)
        trained_list = train_parallel(1000, trained_list)[0]
        print(' ')
    
    df = pd.DataFrame({'aware_win': aware_win, 
                       'aware_loss': aware_loss,
                       'aware_tie': aware_tie,
                       'not_aware_win': not_aware_win,
                       'not_aware_loss': not_aware_loss,
                       'not_aware_tie': not_aware_tie})
    df.to_csv('output/symmetric_comparison_scores.csv', index=False)


def compare_dummy_learning_rate():
        # setup
    scores1, scores2 = [], []
    
    # agent to train
    agent1 = Agent(1, 0.5)
    agent_not_dummy = Agent(2, 0.5)
    agent2 = Agent(1, 0.5)
    agent_dummy = Agent(2, 0.5, dummy=True)

    # track scores
    dummy_win, dummy_loss, dummy_tie = [], [], [] 
    not_dummy_win, not_dummy_loss, not_dummy_tie = [], [], []

    for i in range(100):
        print(i, '-------')
        # train
        agent_list = train(1000, agent1, agent_not_dummy, return_both=True)
        agent1 = agent_list[0]
        agent_not_dummy = agent_list[1]
        agent2 = train(1000, agent2, agent_dummy)

        # test
        agent1.set_exploration(0)
        agent2.set_exploration(0)

        scores = test_parallel(10000, [agent1, agent2])
       
        agent1.set_exploration(0.5)
        agent2.set_exploration(0.5)

        # append scores
        not_dummy_win.append(scores[0]['wins'])
        not_dummy_loss.append(scores[0]['losses'])
        not_dummy_tie.append(scores[0]['ties'])

        dummy_win.append(scores[1]['wins'])
        dummy_loss.append(scores[1]['losses'])
        dummy_tie.append(scores[1]['ties'])

        print(' ')



    # output dataframe
    df = pd.DataFrame({'dummy_win': dummy_win, 
                       'dummy_loss': dummy_loss,
                       'dummy_tie': dummy_tie,
                       'not_dummy_win': not_dummy_win,
                       'not_dummy_loss': not_dummy_loss,
                       'not_dummy_tie': not_dummy_tie})
    df.to_csv('output/dummy_comparison_scores.csv', index=False)


# PLOT
def plot_symmetric_learning_rate(file_name):

    df = pd.read_csv('output/scores.csv')
    df.rename({'Unnamed: 0':'Index'})

    num = 0
    for col in ['aware_win', 'not_aware_win', 'aware_loss', 'not_aware_loss']:
        num+=1
        plt.plot(df.index, df[col], 
                    alpha=0.9,
                    label=col,
                    linewidth=2)
    
    plt.legend(loc=0, fontsize=10)

    plt.title('Symmetric Aware vs Not Aware Learning Rate', fontsize=14)
    plt.xlabel('Iterations', fontsize=11)
    plt.ylabel('%', fontsize=11)

    plt.savefig(file_name)


def set_exploration(agent_list, exploration=0):
    return [a.set_exploration(0) for a in agent_list]
 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def self_dummy_comparison_scores():

    # Build Agents
    a1 = Agent(1, 0.5)
    a2 = Agent(1, 0.5)
    agent_self = Agent(2, 0.5)
    agent_dummy = Agent(2, 0.5, dummy=True)

    # Score Lists
    a1_meta, a1_self, a1_dummy = [], [], []
    a2_meta, a2_self, a2_dummy = [], [], []

    # Load Meta Agent
    policy = pickle.load(open('policies/meta_agent.p', 'rb'))
    meta_agent = Agent(2, 0)
    meta_agent.set_policy(policy)

    # Loop train and test
    for i in range(100):
        print('Iteration: ', i+1, '---------')
        
        # Train
        set_exploration([a1, a2, agent_self], exploration=0.5)
        agent_self.set_policy(a1.get_policy())
        a1, agent_self = train(1000, a1, agent_self, return_both=True)
        a2 = train_in_pool([1000, a2, agent_dummy])

        # Test Parallel
        set_exploration([a1, a2, agent_self], exploration=0)
        scores_meta = test_parallel(10000, [a1, a2], [meta_agent, meta_agent])
        scores_self = test_parallel(10000, [a1, a2], [agent_self, agent_self])
        scores_dummy = test_parallel(10000, [a1, a2], [agent_dummy, agent_dummy])
        
        # Store Scores
        a1_meta.append(scores_meta[0])
        a1_self.append(scores_self[0])
        a1_dummy.append(scores_dummy[0])

        a2_meta.append(scores_meta[1])
        a2_self.append(scores_self[1])
        a2_dummy.append(scores_dummy[1])
        
        #print \n
        print(' ')

    # Build DataFrames
    df_a1_meta = pd.DataFrame(a1_meta)
    df_a1_self = pd.DataFrame(a1_self)
    df_a1_dummy = pd.DataFrame(a1_dummy)

    df_a2_meta = pd.DataFrame(a2_meta)
    df_a2_self = pd.DataFrame(a2_self)
    df_a2_dummy = pd.DataFrame(a2_dummy)

    # To Excel
    with pd.ExcelWriter('output/self_dummy_comparison_scores.xlsx') as writer:
        df_a1_meta.to_excel(writer, 'self_vs_meta', index=False)
        df_a1_self.to_excel(writer, 'self_vs_self', index=False)
        df_a1_dummy.to_excel(writer, 'self_vs_dummy', index=False)
        df_a2_meta.to_excel(writer, 'dummy_vs_meta', index=False)
        df_a2_self.to_excel(writer, 'dummy_vs_self', index=False)
        df_a2_dummy.to_excel(writer, 'dummy_vs_dummy', index=False)


def plot_self_random_comparison_scores():
    # Build plots
    PATH = 'output/self_dummy_comparison_scores.xlsx'
    sheets = pd.ExcelFile(PATH).sheet_names
    
    f = plt.figure()
    legend_list = [] 
    for sheet in sheets:
        df_temp = pd.read_excel(PATH, sheet)
        plt.plot(df_temp.index, df_temp['losses'])

        legend_list.append(sheet)

    legend_list = ['Self-Play vs Meta', 'Self-Play vs Self-Play', 'Self-Play vs Random',
                   'Random vs Meta', 'Random vs Self-Play', 'Random vs Random']
    # plot formatting
    plt.title('Self-Play vs Random Training: Learning Rate', fontsize=18)
    plt.legend(legend_list, fontsize=13)
    plt.xlabel('Iterations (000s)', fontsize=13)
    plt.ylabel('Loss (%)', fontsize=13)

    # show plot
    plt.show()



def exploration_vs_winrate_comparison_scores():
    # Build Agent
    agent = Agent(1, 0)

    # Load Meta Agent
    policy = pickle.load(open('policies/meta_agent.p', 'rb'))
    meta_agent = Agent(2, 0)
    meta_agent.set_policy(policy)

    # train test
    df_scores = pd.DataFrame()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

    # Build Agents
    a1 = Agent(1, 0.0)
    a2 = Agent(1, 0.2)
    a3 = Agent(1, 0.4)
    a4 = Agent(1, 0.6)
    a5 = Agent(1, 0.8)
    a6 = Agent(1, 1.0)

    # Build agent_list
    exploration_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    agent_list = [a1, a2, a3, a4, a5, a6]
    test_list = [meta_agent] * 6

    # Training Testing Iterations
    df1, df2, df3, df4, df5, df6 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_list = [df1, df2, df3, df4, df5, df6]

    # iterate
    for i in range(100):
        print('Iteration: ', i)
        # train
        for a, e in zip(agent_list, exploration_list):
            a.set_exploration(e)
        agent_list = train_parallel(1000, agent_list, test_list)[0]

        # test
        set_exploration(agent_list, 0)
        scores = test_parallel(10000, agent_list, test_list)
        
        # scores
        scores_dfs = [pd.DataFrame(s, index=[0]) for s in scores]
        df_list = [df.append(s_df) for df, s_df in zip(df_list, scores_dfs)]

        print(' ')
    

    # DataFrame Columns
    for df, e in zip(df_list, exploration_list):
        df.drop('ties', axis=1, inplace=True)
        df.drop('losses', axis=1, inplace=True)
        df.columns = [str(col) + '(' + str(e) + ')' for col in df.columns]
        print(df)


    # DataFrame Output
    df_scores = pd.concat(df_list, axis=1)

    df_scores.to_excel('output/Exploration Rate vs Learning Rate.xlsx', index=False)
    
# # MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    excel = pd.ExcelFile('output/selfplay_vs_random_training.xlsx')
    sheets = excel.sheet_names


    f = plt.figure()
    # iterate plots
    for sheet in sheets:
        df_temp = pd.read_excel('output/selfplay_vs_random_training.xlsx', sheet_name=sheet)
        plt.plot(df_temp.index, df_temp['losses'])
    
    plt.title('Self-Play vs Random Loss Rate', fontsize=22)
    plt.xlabel('Training Iterations (000s)', fontsize=18)
    plt.ylabel('Losses (%)', fontsize=18)
    plt.legend(list(sheets), fontsize=18)

    plt.savefig('output/plots/selfplay_vs_random_training.png')
    plt.show()