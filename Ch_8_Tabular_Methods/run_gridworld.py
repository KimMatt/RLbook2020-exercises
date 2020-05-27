# run_griworld.py
# run experiment for exercise 8.4
import numpy as np
from multiprocessing import Pool
import pandas as pd
import copy
from grid.agent import Agent

def run_trial(agent):
    np.random.seed()
    reward_tracker = []
    cumulative_reward = 0
    while len(reward_tracker) < 6000:
        agent.respawn()
        reward = 0
        while reward != 1 and len(reward_tracker) < 6000:
            if len(reward_tracker) == 3000:
                del agent.gridworld.tiles[(8,3)]
                agent.gridworld.tiles[(0, 3)] = 0
                for action in agent.actions:
                    agent.Q[((0, 3), action)] = 0
            reward = agent.play()
            cumulative_reward += reward
            reward_tracker.append(cumulative_reward)
    return reward_tracker

def average_trials(logs):
    averaged_logs = []
    for i in range(len(logs[0])):
        average = 0
        for log in logs:
            average += log[i]
        averaged_logs.append(average / len(logs))
    return averaged_logs


if __name__ == "__main__":
    agent = Agent(False)
    agent_exp = Agent(True)
    trials = 30

    with Pool(4) as p:
        pool_args = [copy.deepcopy(agent) for i in range(
            trials)] + [copy.deepcopy(agent_exp) for i in range(trials)]
        experiment_logs = p.map(run_trial, pool_args)

    exp_logs = average_trials(experiment_logs[0:trials])
    exp_logs_exp = average_trials(experiment_logs[trials:])

    graph = pd.DataFrame({"Reg Dyna Q": exp_logs, "Eval Dyna Q+": exp_logs_exp}).plot(
        kind="line", title="Exercise 8.4")
    graph.set_xlabel("time steps")
    graph.set_ylabel("cumulative reward")
    f = graph.get_figure()
    f.savefig("./figs/ex_8.5_test.png")
