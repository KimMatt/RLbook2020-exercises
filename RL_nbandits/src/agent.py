# agent.py
# Definition of the agent to play nbandits

import numpy as np
import math


class SimpleAgent:

    accepted_types = {"means": True, "constant": True, "increasing": True}

    def __init__(self, exploration, val_init, bandit, **kwargs):
        """[summary]

        Args:
            exploration ([float]): value between 0 and 1 of how much to explore vs. exploit
            val_init ([float]): value to initialize all the policies to
            bandit ([Object]): bandit object to interface with
            kwargs:
                method ([string], optional): method of updating policies
                                            can be 'means', 'constant', or 'increasing'
                                            Defaults to means.
                constant ([float], optional): specifies the constant for constant method
        """
        self.total_plays = 0
        self.total_rewards = 0

        self.exploration = exploration
        self.bandit = bandit
        n = self.bandit.n
        self.policy = [val_init for i in range(n)]
        self.k_tracker = [1.0 for i in range(n)]

        self.method = kwargs.get("method") if kwargs.get("method") else "means"
        if not self.accepted_types.get(self.method):
            raise Exception
        self.constant = kwargs.get("constant") if kwargs.get("constant") else 0.1

    def update_reward_means_method(self, arm, reward):
        self.k_tracker[arm] += 1.0
        self.policy[arm] += float((1.0/self.k_tracker[arm]) * (reward - self.policy[arm]))

    def update_reward_constant(self, arm, reward):
        self.policy[arm] += float(self.constant * (reward - self.policy[arm]))

    def update_reward_increasing(self, arm, reward):
        self.k_tracker[arm] += 1.0
        self.policy[arm] += float(math.tanh(self.k_tracker[arm]) * self.constant * (reward - self.policy[arm]))

    def play(self):
        explore = np.random.uniform(0, 1) <= self.exploration
        max_value = np.max(np.array(self.policy))
        if explore:
            arm = np.random.randint(self.bandit.n)
        else:
            max_value = np.max(np.array(self.policy))
            arm = self.policy.index(max_value)
        reward = self.bandit.sample(arm)
        if self.method == "means":
            self.update_reward_means_method(arm, reward)
        elif self.method == "constant":
            self.update_reward_constant(arm, reward)
        elif self.method == "increasing":
            self.update_reward_increasing(arm, reward)
        self.total_rewards += reward
        return reward
