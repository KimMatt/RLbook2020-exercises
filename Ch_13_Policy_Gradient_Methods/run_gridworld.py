from src.agent import Agent
import pandas as pd
from ..utils.experiment import Experiment

if __name__ == "__main__":
    agent = Agent()
    dc_factor = agent.discount_rate
    episodes = 1000
    time_steps = []

    for episode in range(episodes):
        agent.respawn()
        print("episode: ", episode)
        # play an episode
        reward = 0
        steps = []
        time_step = 0
        while reward != 1:
            state, action, reward, _ = agent.play_step()
            steps.append((state, action, reward))
            time_step += 1
        time_steps.append(time_step)
        # calculate G values at each step
        Gs = [steps[-1][2]]
        for i in range(2,len(steps)+1):
            Gs.insert(0, steps[-i][2] + dc_factor*Gs[0])
        for t in range(len(steps)):
            agent.grad_step(steps[t][0], steps[t][1], Gs[t], t)

    graph = pd.DataFrame({"time_steps": time_steps}).plot(
        kind="line", title="Gridworld with 2 layer NN, PGM")
    graph.set_xlabel("episodes")
    graph.set_ylabel("time steps")
    f = graph.get_figure()
    f.savefig("./figs/gridworld_pgm.png")
