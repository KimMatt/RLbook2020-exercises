# agent.py
# Definition of the agent to play nbandits

import numpy as np


class SimpleAgent:


    def __init__(self, exploration, val_init, bandit, **kwargs):
        """[summary]

        Args:
            exploration ([float]): value between 0 and 1 of how much to explore vs. exploit
            val_init ([float]): value to initialize all the policies to
            bandit ([Object]): bandit object to interface with
            kwargs:
                method ([string], optional): method of updating policies
                                            can be 'means' or 'constant'.
                                            Defaults to means.
                constant ([float], optional): specifies the constant for constant method
        """
        self.exploration = exploration
        self.bandit = bandit
        n = self.bandit.n
        self.policy = [val_init for i in range(n)]
        self.k_tracker = [1.0 for i in range(n)]
        self.total_plays = 0
        self.method = kwargs.get("method") if kwargs.get("method") else "means"
        self.constant = 0.1 if not kwargs.get("constant") else kwargs.get("constant")

    def update_reward_means_method(self, n, reward):
        self.k_tracker[n] += 1.0
        self.policy[n] += float((1.0/self.k_tracker[n]) * (reward - self.policy(n)))

    def update_reward_constant(self, n, reward, alpha):
        self.policy[n] += float(alpha * (reward - self.policy(n)))

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
            which = np.random.randint(
                len(sorted_policy)-num_optimal-1, len(sorted_policy))
        else:
            which = np.random.randint(
                0, len(sorted_policy)-num_optimal-1)
        self.total_plays += 1
        return sorted_policy[which]
