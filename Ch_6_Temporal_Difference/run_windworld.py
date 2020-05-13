from src.agent import Agent
import pickle

if __name__ == "__main__":

    iterations = 2
    agent = Agent()

    for i in range(iterations):
        reward = 0
        agent.spawn()
        action = agent.select_action()
        while reward != 1:
            reward, action = agent.play(action)

    pickle.dump(agent.Q, open("./pickles/king_qs.p", "rb"))
