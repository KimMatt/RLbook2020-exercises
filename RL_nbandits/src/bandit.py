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


class Bandit:

    def __init__(self, n):
        self.distributions = [[] for i in range(n)]

    def sample(self, n):
        which = np.random.uniform(0, 1)
        cumulative_chance = 0.0
        for distribution in self.distributions[n]:
            if which <= distribution.chance + cumulative_chance:
                return distribution.sample()
            cumulative_chance += distribution.chance
        raise IncompleteBanditException

    def add_distribution(self, distribution, n):
        self.distributions.append(distribution)
