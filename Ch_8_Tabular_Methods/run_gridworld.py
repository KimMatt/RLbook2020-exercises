# run_griworld.py
# run experiment for exercise 8.4
import numpy as np
from multiprocessing import Pool
import pandas as pd
import copy
from src.agent import Agent

def run_trials(agent):
    np.random.seed()
    t_tracker = []
    t = 0
    for episode in range(3000):
        if episode == 1500:
            del agent.gridworld.tiles[(8,4)]
            agent.gridworld.tiles[(0, 4)] = 0
        agent.respawn()
        reward = 0
        while reward != 1:
            reward = agent.play()
            t += 1
        t_tracker.append(t)
    return t_tracker

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
    trials = 3

    with Pool(trials) as p:
        pool_args = [copy.deepcopy(agent) for i in range(
            trials)] + [copy.deepcopy(agent_exp) for i in range(trials)]
        experiment_logs = p.map(run_trials, pool_args)

    exp_logs = average_trials(experiment_logs[0:trials])
    exp_logs_exp = average_trials(experiment_logs[trials:])

    graph = pd.DataFrame({"Reg Dyna Q+": exp_logs, "Eval Dyna Q+": exp_logs_exp}).plot(
        kind="line", title="Exercise 8.4")
    newx = graph.lines[0].get_ydata()
    newy = graph.lines[0].get_xdata()
    graph.lines[0].set_xdata(newx)
    graph.lines[0].set_ydata(newy)
    graph.set_xlabel("time steps")
    graph.set_ylabel("episodes")
    f = graph.get_figure()
    f.savefig("./figs/ex_8.4.png")
