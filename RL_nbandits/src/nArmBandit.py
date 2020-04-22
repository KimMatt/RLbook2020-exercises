# N-Armed Bandit problem
# greedy selection
# incremental actions with alpha = 1/k step-size parameter
# bandit(a) takes action and returns award

# Q_k+1 = 1/k * sum(R_1_to_k)
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Bandit():

    def __init__(self, n, epsilon):
        self.epsilon = epsilon
        self.n = n
        self.turns = [1]*n
        self.total_rewards = [0] * n
        self.avg_rewards = [0] * n
        self.rewards = [np.random.normal() for i in range(n)]

    def set_rewards(self, rewards):
        self.rewards = rewards

    def choose(self):
        # random epilson, non-greedy parameter
        choose_random = random.random() < self.epsilon

        # chooses arm based on avg reward and epsilon
        if (np.sum(self.turns) == self.n) or choose_random:
            arm = np.random.randint(self.n)
        else:
            max = np.max(self.avg_rewards)
            arm = self.avg_rewards.index(max)
        return arm

    # Update fields
    def update(self):
        # choose an arm
        arm = self.choose()

        # get reward + noise
        reward = self.rewards[arm] + np.random.normal()

        # update totalulative rewards
        self.total_rewards[arm] += reward

        # update average reward
        if self.turns[arm] == 0:
            self.avg_rewards[arm] += reward
        else:
            self.avg_rewards[arm] += (1 / self.turns[arm]) * \
                                            (reward - self.avg_rewards[arm])

        self.avg_rewards = self.avg_rewards
        # update turns
        self.turns[arm] += 1

    # Get average rewards
    def get_average(self):
        reward = np.sum(self.total_rewards) / (np.sum(self.turns) - self.n)
        return reward



