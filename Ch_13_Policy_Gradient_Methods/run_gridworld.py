from src.agent import Agent
import pandas as pd

if __name__ == "__main__":
    episodes = 2000
    time_steps = []
    MODE = 'eg_trace'

    if MODE == 'baseline':
        agent = Agent(0.000001, 0.000001)
        dc_factor = agent.dc_factor
        for episode in range(episodes):
            agent.respawn()
            print("beginning episode {}".format(episode))
            # play an episode
            reward = 0
            steps = []
            state = agent.state
            while reward != 0.1:
                _, action_index, reward, next_state = agent.play_step()
                steps.append((state, action_index, reward))
                state = next_state
                print("episode: {}, step: {}, state:{}".format(episode, len(steps), next_state))
            print("episode done, steps taken: {}".format(len(steps)))
            time_steps.append(len(steps))
            # calculate G values at each step
            Gs = [steps[-1][2]]
            for i in range(2,len(steps)+1):
                Gs.insert(0, steps[-i][2] + dc_factor*Gs[0])
            for t in range(len(steps)):
                agent.reinforce_with_baseline(steps[t][0], steps[t][1], Gs[t])
    elif MODE == 'actor_critic':
        agent = Agent(0.001, 0.01)
        for episode in range(episodes):
            agent.respawn()
            print("beginning episode {}".format(episode))
            # play an episode
            steps = 0
            reward = 0
            while reward != 0.1:
                state, action_index, reward, next_state = agent.play_step()
                agent.actor_critic(state, next_state, action_index, reward)
                print("episode: {}, step: {}, state:{}".format(episode, steps, next_state))
                steps += 1
            print("episode done, steps taken: {}".format(steps))
            time_steps.append(steps)
    elif MODE == "eg_trace":
        agent = Agent(0.001, 0.01)
        for episode in range(episodes):
            agent.respawn()
            print("beginning episode {}".format(episode))
            # play an episode
            steps = 0
            reward = 0
            while reward != 0.1:
                state, action_index, reward, next_state = agent.play_step()
                agent.actor_critic_eg_trace(state, next_state, action_index, reward)
                print("episode: {}, step: {}, state:{}".format(episode, steps, next_state))
                steps += 1
            print("episode done, steps taken: {}".format(steps))
            time_steps.append(steps)


    graph = pd.DataFrame({"time_steps": time_steps}).plot(
        kind="line", title="Gridworld {}".format(MODE))
    graph.set_xlabel("episodes")
    graph.set_ylabel("time steps")
    f = graph.get_figure()
    f.savefig("./figs/gridworld_{}_1layer.png".format(MODE))
