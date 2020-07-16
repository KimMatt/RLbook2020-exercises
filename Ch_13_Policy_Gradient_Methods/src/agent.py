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
        self.loss = 
        self.optimizer = 

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
        roll = np.random.rand()
        h_vals = []
        # calculate h val for each action based on softmax
        for action in actions:
            h_vals.append(math.exp(self.pi(torch.tensor([action[0], action[1], self.state[0], self.state[1]], dtype=torch.float))))
        total = sum(h_vals)
        # with a running count, choose the action stochastically
        running_sum = 0.0
        for index, h_val in enumerate(h_vals):
            running_sum += h_val
            if running_sum / total >= roll:
                return actions[index]

    def play(self):
        """Play a time_step of gridworld and return the (learned) reward
        """
        action = self.get_action()
        reward, s_prime = self.gridworld.time_step(self.state, action)
        self.state = s_prime
        self.t = t
        return action, reward, s_prime

    def grad_step(self, state, action, G, s_prime):
        """Calculate gradients and update them accordingly based on the given time step
        """
        # calculate delta
        delta = G - self.v(torch.tensor(state, dtype=torch.float))
        # update gradients wrt their weights of both pi and v with delta 
        # get the gradient by itself
        # multiply it by delta
        # backwards without a loss function.
        self.loss.backward()
        for p in model.parameters():
            p.grad *= delta
        self.optimizer.step()
        delta = reward + self.discount_rate * self.v(t_s_prime) - self.v(t_state)
        self.w += self.alpha_w * delta * # TODO: gradient of value function wrt. w
        self.theta += self.alpha_theta * delta * numpy.identity_matrix * # TODO: gradient of ln(pi(A|S,theta)) wrt theta
        I = self.discount_rate * I
        self.state = s_prime
        self.t += 1
        return reward

    def respawn(self):
        """respawn the agent at starting state"""
        self.state = (3, 5)
 