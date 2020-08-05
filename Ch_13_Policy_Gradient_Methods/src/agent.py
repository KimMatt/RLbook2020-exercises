# Policy Gradient Method, Actor Critic
import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.gridworld import GridWorld
from src.approximator import NN

# commenting out because my gpu -> wsl integration is bad :(
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Agent():

    def __init__(self, episodes, alpha_theta, alpha_w):
        """Initialize agent"""
        self.gridworld = GridWorld()
        # policy func learning rate
        self.alpha_theta = alpha_theta
        # value func learning rate
        self.alpha_w = alpha_w
        self.dc_factor = 1.0
        self.decay_w = 0.9
        self.decay_theta = 0.9
        self.actions = [1,2,3,4]
        self.start_state = self.gridworld.start_state
        self.state = None
        self.v = NN(self.gridworld.height + self.gridworld.width, 1).to(device)
        self.v_optim = optim.RMSprop(self.v.parameters(), lr=self.alpha_w)
        self.pi = NN(self.gridworld.height + self.gridworld.width, 4, softmax_out=True).to(device)
        self.pi_optim = optim.RMSprop(self.pi.parameters(), lr=self.alpha_theta)


    def state_to_encoding(self, state):
        """Turn the given state tuple(x,y) into a one hot encoding representation."""
        y_encoding = [1 if state[1] == i else 0 for i in range(self.gridworld.height)]
        x_encoding = [1 if state[0] == i else 0 for i in range(self.gridworld.width)]
        position_encoding = torch.tensor(y_encoding + x_encoding, dtype=torch.float).to(device)
        return position_encoding


    def get_action(self):
        """pick an action according to stochastic policy

        Returns:
            chosen_action: (x,y)
        """
        actions = self.actions
        # with a running count, choose the action stochastically
        roll = np.random.random()
        pi_vals = []
        # calculate h val for each action based on softmax
        with torch.no_grad():
            state_encoding = self.state_to_encoding(self.state)
            pi_vals = self.pi(state_encoding)
            print(pi_vals)
        running_sum = 0.0
        for i, pi_val in enumerate(pi_vals):
            running_sum += pi_val
            if running_sum >= roll or i == len(actions) - 1:
                action_index = i
                break
        return actions[action_index], action_index


    def play_step(self):
        """Play a time_step of gridworld and return the (learned) reward."""
        state = self.state
        action, action_index = self.get_action()
        reward, next_state = self.gridworld.time_step(self.state, action)
        self.state = next_state
        return state, action_index, reward, next_state


    def backprop_value(self, state_encoding, advantage):
        """Give state and advantage for time step, backpropagate value's weights."""
        self.v.zero_grad()
        value = self.v(state_encoding)
        value.backward()
        for p in self.v.parameters():
            p.grad.data 
            p.data += self.alpha_w * p.grad.data * advantage


    def backprop_value_eg_trace(self, state_encoding, advantage):
        """Give state and advantage for time step, backpropagate value's weights."""
        # decay weights so far
        for p in self.v.parameters():
            if p.grad is not None:
                p.grad.data = p.grad.data * self.decay_w
        value = self.v(state_encoding)
        value.backward()
        for index, p in enumerate(self.v.parameters()):
            p.data += self.alpha_w * p.grad.data * advantage


    def backprop_pi(self, state_encoding, advantage, action_index):
        """Give state and advantage for time step, backpropagate pi's weights."""
        self.pi.zero_grad()
        pi_val = self.pi(state_encoding)[action_index]
        # epsilon to prevent log(0) making nans
        epsilon = 0.00001
        ln_pi_val = torch.log(pi_val + epsilon)
        ln_pi_val.backward()
        for p in self.pi.parameters():
            p.data += self.alpha_theta * p.grad.data * advantage 


    def backprop_pi_eg_trace(self, state_encoding, advantage, action_index):
        """Give state and advantage for time step, backpropagate pi's weights."""
        # decay weights so far
        for p in self.pi.parameters():
            if p.grad is not None:
                p.grad.data = p.grad.data * self.decay_theta
        pi_val = self.pi(state_encoding)[action_index]
        # epsilon to prevent log(0) making nans
        epsilon = 0.00001
        ln_pi_val = torch.log(pi_val + epsilon)
        ln_pi_val.backward()
        for p in self.pi.parameters():
            p.data += self.alpha_theta * p.grad.data * advantage 


    def reinforce_with_baseline(self, state, action_index, G):
        """Calculate gradients and update 
        them accordingly based on the given time step."""
        state_encoding = self.state_to_encoding(state)
        with torch.no_grad():
            value_state = self.v(state_encoding)
        advantage = G - value_state
        # update w values based on advantage
        self.backprop_value(state_encoding, advantage)
        # update theta values based on advantage
        self.backprop_pi(state_encoding, advantage, action_index)


    def actor_critic(self, state, next_state, action_index, r):
        """One step of actor_critic."""
        state_encoding = self.state_to_encoding(state)
        next_state_encoding = self.state_to_encoding(next_state)
        with torch.no_grad():
            value_state = self.v(state_encoding)
            value_next_state = self.v(next_state_encoding)
        advantage = r + value_next_state - value_state
        # update w values based on advantage
        self.backprop_value(state_encoding, advantage)
        # update theta values based on advantage
        self.backprop_pi(state_encoding, advantage, action_index)


    def actor_critic_eg_trace(self, state, next_state, action_index, r):
        """One step of actor_critic with eligibility traces"""
        state_encoding = self.state_to_encoding(state)
        next_state_encoding = self.state_to_encoding(next_state)
        with torch.no_grad():
            value_state = self.v(state_encoding)
            value_next_state = self.v(next_state_encoding)
        advantage = r + value_next_state - value_state
        # update w values based on advantage
        self.backprop_value_eg_trace(state_encoding, advantage)
        # update theta values based on advantage
        self.backprop_pi_eg_trace(state_encoding, advantage, action_index)

    def respawn(self):
        """Respawn the agent at starting state."""
        self.state = self.start_state
        # for when we are using eg traces
        self.v.zero_grad()
        self.pi.zero_grad()
 