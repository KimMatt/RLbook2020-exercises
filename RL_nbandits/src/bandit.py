# bandit.py
# class to interface with for a bandit AKA a randomly distributed 'slot machine'

import numpy as np

class IncompleteBanditException(Exception):

    def __init__(self):
        super().__init__()


class Distribution:

    def __init__(self, chance):
        self.chance = chance

    def sample(self):
        pass


class UniformDistribution(Distribution):

    def __init__(self, chance, mean, std):
        super().__init__(chance)
        self.std = std
        self.mean = mean

    def sample(self):
        return np.random.normal(loc=self.mean, scale=self.std)


class Bandits:

    walk_amount = 0.2

    def __init__(self, n):
        self.n = n
        self.bandits = np.array([np.random.normal(
            loc=0.0, scale=1.0) for i in range(n)])

    def random_walk_bandits(self):
        random_walk = np.array([0.2 * (np.random.randint(3) - 1) for i in range(self.n)])
        self.bandits = np.add(self.bandits,random_walk)

    def sample(self, n):
        noise = np.random.normal(loc=0.0, scale=1.0)
        return self.bandits[n] + noise

    def get_optimal(self):
        optimal_n = 0
        optimal_value = self.bandits[0]
        for i in range(self.n):
            if optimal_value < self.bandits[i]:
                optimal_value = self.bandits[i]
                optimal_n = i
        return (optimal_n, optimal_value)
