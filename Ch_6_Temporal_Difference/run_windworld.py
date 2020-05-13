from src.agent import Agent
import pandas as pd

if __name__ == "__main__":

    episodes = 171
    agent = Agent()

    time_step = 0
    time_step_tracker = [0]

    for i in range(episodes):
        print("episode: {}".format(i))
        reward = 0
        agent.spawn()
        action = agent.select_action()
        while reward != 1:
            reward, action = agent.play(action)
            time_step += 1
        time_step_tracker.append(time_step)

    graph = pd.DataFrame({"time_steps": time_step_tracker}).plot(kind="line", title="Exercise 6.10")
    newx = graph.lines[0].get_ydata()
    newy = graph.lines[0].get_xdata()
    graph.lines[0].set_xdata(newx)
    graph.lines[0].set_ydata(newy)
    graph.set_xlim([-100, time_step + 100])
    graph.set_ylim([-5, episodes + 5])
    graph.set_xlabel("time steps")
    graph.set_ylabel("episodes")
    f = graph.get_figure()
    f.savefig("./figs/ex_6.10.png")
