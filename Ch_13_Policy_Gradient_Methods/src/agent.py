# Policy Gradient Method, Actor Critic
import numpy as np
import math

from src.gridworld import GridWorld
from src.approximator import NN

class Agent():

    def __init__(self):
        """Initialize agent
        """
        self.gridworld = GridWorld()
        self.alpha_theta = 0.1
        self.alpha_w = 0.1
        self.t = 0
        # state action pairs that have been encountered, mapped to the time step
        # they were last used
        # ((x,y),(x_d,y_d)) = t
        self.actions = []
        self.discount_rate = 0.95
        for x in range(-1,2):
            if x == 0:
                pass
            else:
                self.actions.append((x,0))
                self.actions.append((0,x))
        self.state = (0,3)
        self.v = NN(2)
        self.pi = NN(4)

    def update_V(self, state, action, reward, next_state):
        """Update q value for the given transition

        Args:
            state ((x,y)): current state
            action ((x_d,y_d)): action taken
            reward (int): reward given from transition
            next_state ((x,y)): resultant state of transition
        """
        _, max_q_value = self.get_max_action(next_state)
        plus_factor = 0
        self.Q[(state, action)] += self.alpha * ((reward + plus_factor + (self.discount_rate *
                                                 max_q_value)) - self.Q[(state, action)])

    def get_possible_actions(self, state):
        """Give possible actions from given state

        Args:
            state ((x,y)): state

        Returns:
            [(x_d,y_d)]: A list of actions
        """
        actions = []
        for action in self.actions:
            if self.gridworld.tiles.get((action[0] + state[0], action[1] + state[1])) is not None:
                actions.append(action)
        return actions

    def get_action(self):
        """pick an action according to stochastic policy

        Returns:
            chosen_action: (x,y)
        """
        actions = self.get_possible_actions(self.state)
        pi_values = []
        for action in actions:
            self.pi.forward(torch.tensor([action[0], action[1], self.state[0], self.state[1]]))

    def play(self):
        """Play a time_step of gridworld and return the (learned) reward
        """
        action = self.get_action()
        reward, s_prime = self.gridworld.time_step(self.state, action)
        delta = reward + self.discount_rate * self.V(s_prime, self.w) - self.V(state, self.w)
        self.w += self.alpha_w * delta * # TODO: gradient of value function
        self.theta += self.alpha_theta * delta * numpy.identity_matrix * #TODO: gradient of ln(pi(A|S,theta))
        I = self.discount_rate * I
        self.state = s_prime
        self.t += 1
        return reward

    def respawn(self):
        """respawn the agent at starting state"""
        self.state = (3, 5)
