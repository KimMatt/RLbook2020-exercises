# agent.py
# Definition of the agent to play nbandits

import numpy as np


class Agent:

    def __init__(self, exploration, n, val_init, bandit):
        self.exploration = exploration
        self.context_policy = None #TODO: Figure this part out
        self.policy = [val_init for i in range(n)]
        self.bandit = bandit
        self.score = 0


    def play(self):
        explore = np.random.uniform(0,1) <= self.exploration
        sorted_policy = self.policy[:]
        sorted_policy = [(x,i) for i,x in enumerate(sorted_policy)]
        list.sort(sorted_policy)
        num_optimal = 0
        for i in range(len(sorted_policy)-1, -1, -1):
            if sorted_policy[i][0] == sorted_policy[-1][0]:
                num_optimal += 1
            else:
                break
        if not explore:
            which_optimal = np.random.randint(
                len(sorted_policy)-num_optimal-1, len(sorted_policy))
        else:
            which_optimal = np.random.randint(
                0, len(sorted_policy)-num_optimal)
        reward = self.bandit.sample(sorted_policy[which_optimal][1])
        self.score += reward
        return reward


    def update_reward(self, n):
        pass
