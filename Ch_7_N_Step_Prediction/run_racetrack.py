# Run racetrack for empirical experiment for exercise 7.2

import pandas as pd
import numpy as np
from src.agent import Agent


if __name__ == "__main__":

    episodes = 5001
    time_step_tracker = []
    n = 4
    agent = Agent(n)
    t_step = 0
    for i in range(episodes):
        print("episode: {}".format(i))
        reward = -1
        while reward != 1:
            reward = agent.play()
            t_step += 1
        for t in range(agent.t, agent.t + agent.n - 1):
            if t >= agent.n - 1:
                # err = np.sum([(agent.discount_rate ** i) * (agent.rewards[j] + agent.discount_rate * \
                #               agent.update_values[j+1] - agent.update_values[j]) for i,j in enumerate([k for k in
                #               range(t - agent.n + 1, agent.t)])])
                # update_state = agent.states[t - agent.n + 1]
                # agent.values[update_state] += agent.alpha * err
                G = np.sum([(agent.discount_rate ** i) * agent.rewards[j] for i, j in
                            enumerate([k for k in range(t - agent.n + 1, agent.t)])])
                # up to T which is agent.t
                G += (agent.discount_rate ** agent.n) * \
                    agent.values[agent.states[agent.t]]
                update_state = agent.states[t - agent.n + 1]
                err = G - agent.values[update_state]
                agent.values[update_state] += agent.alpha * err
        agent.reset()
        time_step_tracker.append(t_step)

    graph = pd.DataFrame({"time_steps": time_step_tracker}).plot(
        kind="line", title="Exercise 7.2 V-state t+n-1")
    newx = graph.lines[0].get_ydata()
    newy = graph.lines[0].get_xdata()
    graph.lines[0].set_xdata(newx)
    graph.lines[0].set_ydata(newy)
    graph.set_xlim([-100, t_step + 100])
    graph.set_ylim([-5, episodes + 5])
    graph.set_xlabel("time steps")
    graph.set_ylabel("episodes")
    f = graph.get_figure()
    f.savefig("./figs/ex_7.2_1.png")
