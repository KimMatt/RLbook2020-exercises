import numpy as np
import math

from src.gridworld import GridWorld

class Agent():

    def __init__(self, experimental):
        """Initialize agent

        Args:
            experimental(Boolean): whether or not this is the experimental agent.
        """
        self.experimental = experimental
        self.gridworld = GridWorld()
        self.model = {}
        self.k = 0.1
        self.alpha = 0.1
        self.epsilon = 0.1
        self.t = 0
        # state action pairs that have been encountered, mapped to the time step
        # they were last used
        # ((x,y),(x_d,y_d)) = t
        self.S_A = {}
        self.n = 50
        self.actions = []
        self.discount_rate = 0.95
        for x in range(-1,0,2):
            if x == 0:
                self.actions.append((x,x))
            else:
                self.actions.append((x,0))
                self.actions.append((0,x))
        self.Q = {}
        for state in self.gridworld.tiles:
            for action in self.actions:
                self.Q[(state, action)] = 0
        self.state = (3,5)

    def get_max_action(self, state, experimental=False):
        """Return the max action

        Args:
            state ((x,y)): tuple of x and y coordinates
            experimental (boolean): if we want to use the experimental version or
            not (we want to be able to use both during the experimental version)

        Returns:
            max_action ((x_d,y_D)) tuple of action
        """
        max_q_value = 0
        for action in self.actions:
            q_value = self.Q[(state, action)]
            if experimental and self.S_A.get((state, action)):
                t_passed = self.t - self.S_A.get((state, action))
                q_value += self.k * math.sqrt(t_passed)
            if q_value >= max_q_value:
                max_q_value = q_value
                max_action = action
        return max_action, max_q_value

    def plan(self):
        """take a plan step, one step tabular Q learning over random S,A
        """
        state, action = list(self.S_A.keys())[
            np.random.randint(0, len(self.S_A))]
        r, s_prime = self.model.get((state, action))
        _, max_q_value = self.get_max_action(s_prime)
        # add option for Dyna Q+
        t_passed = self.t - self.S_A[(state, action)]
        plus_factor = 0 if self.experimental else self.k * math.sqrt(t_passed)
        self.Q[(state, action)] += self.alpha * (r + plus_factor + self.discount_rate *
                                                 max_q_value - self.Q[(state, action)])

    def get_action(self):
        """pick an action according to epsilon-greedy policy

        Returns:
            chosen_action: (x,y)
        """
        if np.random.rand() > self.epsilon:
            action, _ = self.get_max_action(self.state)
        else:
            action = self.actions[np.random.randint(0, len(self.actions))]
        return action

    def play(self):
        """Play a time_step of gridworld and return the (learned) reward
        """
        action = self.get_action()
        reward, s_prime = self.gridworld.time_step(self.state, action)
        _, max_q_value = self.get_max_action(s_prime, self.experimental)
        self.Q[(self.state, action)] += self.alpha * (reward + self.discount_rate *
                                             max_q_value - self.Q[(self.state, action)])
        self.model[(self.state, action)] = (reward, s_prime)
        # update list of encountered states
        self.S_A[(self.state, action)] = self.t
        for i in range(self.n):
            self.plan()
        # update time and state
        self.state = s_prime
        self.t += 1
        return reward

    def respawn(self):
        """respawn the agent at starting state"""
        self.state = (3, 5)
