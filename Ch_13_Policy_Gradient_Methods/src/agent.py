# Policy Gradient Method, Actor Critic
import numpy as np
import math
import torch
import torch.optim as optim

from src.gridworld import GridWorld
from src.approximator import NN

# commenting out because my gpu -> wsl integration is bad :(
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Agent():

    def __init__(self):
        """Initialize agent
        """
        self.gridworld = GridWorld()
        self.alpha_theta = 0.1
        self.alpha_w = 0.1
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
        self.v = NN(2).to(device)
        self.v_optim = optim.RMSprop(self.v.parameters(), lr=self.alpha_w)
        self.pi = NN(4).to(device)
        self.pi_optim = optim.RMSprop(self.pi.parameters(), lr=self.alpha_theta)


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
            with torch.no_grad():
                pi_val = self.pi(torch.tensor([action[0], action[1], self.state[0], self.state[1]], dtype=torch.float).to(device))
                h_vals.append(math.exp(pi_val))
        total = sum(h_vals)
        # with a running count, choose the action stochastically
        running_sum = 0.0
        for index, h_val in enumerate(h_vals):
            running_sum += h_val
            if running_sum / total >= roll:
                return actions[index]

    def play_step(self):
        """Play a time_step of gridworld and return the (learned) reward
        """
        state = self.state
        action = self.get_action()
        reward, s_prime = self.gridworld.time_step(self.state, action)
        self.state = s_prime
        return state, action, reward, s_prime

    def grad_step(self, state, action, G, t):
        """Calculate gradients and update them accordingly based on the given time step
        """
        # update w values based on value function
        self.v_optim.zero_grad()
        value = self.v(torch.tensor([state[0], state[1]], dtype=torch.float).to(device))
        delta = G - value
        value.backward()
        for p in self.v.parameters():
            p.grad[:] = p.grad * delta
        self.v_optim.step()
        # update theta values based on pi function
        self.pi_optim.zero_grad()
        ln_pi_val = torch.log(self.pi(torch.tensor([action[0], action[1], state[0], state[1]], dtype=torch.float).to(device)))
        ln_pi_val.backward()
        for p in self.pi.parameters():
            p.grad[:] = p.grad * delta * self.discount_rate ** t
        self.pi_optim.step()

    def respawn(self):
        """respawn the agent at starting state"""
        self.state = (3, 5)
 