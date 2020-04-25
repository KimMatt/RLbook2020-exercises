# bandit.py
# class to interface with for a bandit AKA a randomly distributed 'slot machine'

import numpy as np


class NBandits:

    walk_amount = 0.1

    def __init__(self, n, random=True):
        self.n = n
        if random:
            self.bandits = np.array([np.random.normal(
                loc=0.0, scale=1.0) for i in range(n)])
        else:
            self.bandits = np.array([0.0 for i in range(n)])

    def independent_random_walk(self):
        for i in range(self.n):
            sign = 1 if np.random.randint(2) else -1
            self.bandits[i] += sign * 0.2

    def alternating_dependent_random_walk(self):
        sign = 1 if np.random.randint(2) else -1
        for i in range(self.n):
            self.bandits[i] += sign * 0.2
            sign = sign * -1.0

    def total_random_walk(self):
        sign = 1 if np.random.randint(2) else -1
        self.bandits = [self.bandits[i] + sign * 0.2 for i in range(self.n)]

    def sample(self, n):
        noise = np.random.normal(loc=0.0, scale=1.0)
        return self.bandits[n] + noise

    def get_optimal(self):
        opt_arm = 0
        opt_val = self.bandits[0]
        for i in range(self.n):
            if opt_val < self.bandits[i]:
                opt_val = self.bandits[i]
                opt_arm = i
        return (opt_arm, opt_val)
