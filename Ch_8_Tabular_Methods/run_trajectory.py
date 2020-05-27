# run_griworld.py
# run experiment for exercise 8.4
import numpy as np
from multiprocessing import Pool
import pandas as pd
import copy
from trajectory.agent import Agent
from trajectory.states import States


def run_trial(agent):
    np.random.seed()
    while len(agent.start_s_values) < 200000:
        agent.play()
    return agent.start_s_values

def average_trials(logs):
    averaged_logs = []
    for i in range(len(logs[0])):
        average = 0
        for log in logs:
            average += log[i]
        averaged_logs.append(average / len(logs))
    return averaged_logs




if __name__ == "__main__":
    states = States(1, 10000)
    agent_trajectory = Agent(states, True)
    agent_uniform = Agent(states, False)
    trials = 3

    with Pool(4) as p:
        pool_args = [copy.deepcopy(agent_trajectory) for i in range(
            trials)] + [copy.deepcopy(agent_uniform) for i in range(trials)]
        experiment_logs = p.map(run_trial, pool_args)

    experiment_logs = [average_trials(experiment_logs[0:trials]), average_trials(experiment_logs[trials:])]

    graph = pd.DataFrame({"on-policy": experiment_logs[0], "uniform": experiment_logs[1]}).plot(
        kind="line", title="Exercise 8.8 b=3")
    graph.set_xlabel("computation time, in expected updates")
    graph.set_ylabel("Value of start state under greedy policy")
    f = graph.get_figure()
    f.savefig("./figs/ex_8.8_2.png")
