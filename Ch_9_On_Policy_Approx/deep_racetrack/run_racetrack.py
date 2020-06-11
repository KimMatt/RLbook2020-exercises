# Run racetrack but with TD(0) with a 3 layered neural network as a function approximator
# for the after-state value function
import pandas as pd
import numpy as np
from src.tdn import Agent

if __name__ == "__main__":

    episodes = 250
    time_step_tracker = []
    agent = Agent(4)

    t_step = 0
    for i in range(episodes):
        print("episode: {}".format(i))
        reward = -1
        while reward != 1:
            reward = agent.play()
            t_step += 1
        agent.td_cleanup()
        agent.reset()
        time_step_tracker.append(t_step)

    graph = pd.DataFrame({"time_steps": time_step_tracker}).plot(
        kind="line", title="TD(4) with neural network FA")
    newx = graph.lines[0].get_ydata()
    newy = graph.lines[0].get_xdata()
    graph.lines[0].set_xdata(newx)
    graph.lines[0].set_ydata(newy)
    graph.set_xlim([-100, t_step + 100])
    graph.set_ylim([-5, episodes + 5])
    graph.set_xlabel("time steps")
    graph.set_ylabel("episodes")
    f = graph.get_figure()
    f.savefig("./figs/nntd4.png")
