import numpy as np
import math

from grid.gridworld import GridWorld

class Agent():

    def __init__(self, experimental):
        """Initialize agent

        Args:
            experimental(Boolean): whether or not this is the experimental agent.
        """
        self.experimental = experimental
        self.gridworld = GridWorld()
        self.model = {}
        self.k = 0.025
        self.alpha = 0.1
        self.epsilon = 0.4
        self.t = 0
        # state action pairs that have been encountered, mapped to the time step
        # they were last used
        # ((x,y),(x_d,y_d)) = t
        self.S_A = {}
        self.n = 5
        self.actions = []
        self.discount_rate = 0.95
        for x in range(-1,2):
            if x == 0:
                pass
            else:
                self.actions.append((x,0))
                self.actions.append((0,x))
        self.Q = {}
        for state in self.gridworld.tiles:
            for action in self.actions:
                self.Q[(state, action)] = 0
        self.state = (3,5)

    def get_max_action(self, state, experimental=False):
        """Return the max action from state and the q_value of the state action pair

        Args:
            state ((x,y)): tuple of x and y coordinates
            experimental (boolean): if we want to use the experimental version or
            not (we want to be able to use both during the experimental version)

        Returns:
            max_action ((x_d,y_D)) tuple of action
        """
        max_q_value = 0
        actions = self.get_possible_actions(state)
        for action in actions:
            q_value = self.Q[(state, action)]
            if experimental:
                t_last = 0 if self.S_A.get(
                    (state, action)) is None else self.S_A.get((state, action))
                t_passed = self.t - t_last
                q_value += self.k * math.sqrt(t_passed)
            if q_value >= max_q_value:
                max_q_value = q_value
                max_action = action
        return max_action, max_q_value

    def update_Q(self, state, action, reward, next_state):
        """Update q value for the given transition

        Args:
            state ((x,y)): current state
            action ((x_d,y_d)): action taken
            reward (int): reward given from transition
            next_state ((x,y)): resultant state of transition
        """
        _, max_q_value = self.get_max_action(next_state)
        plus_factor = 0
        # if not self.experimental:
        #     t_last = 0 if self.S_A.get(
        #         (state, action)) is None else self.S_A.get((state, action))
        #     t_passed = self.t - t_last
        #     plus_factor = self.k * math.sqrt(t_passed)
        # else:
        #     plus_factor = 0
        self.Q[(state, action)] += self.alpha * ((reward + plus_factor + (self.discount_rate *
                                                 max_q_value)) - self.Q[(state, action)])

    def plan(self):
        """take a plan step, one step tabular Q learning over random S,A
        """
        state, action = list(self.S_A.keys())[
            np.random.randint(0, len(self.S_A))]
        r, s_prime = self.model.get((state, action))
        self.update_Q(state, action, r, s_prime)

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
        """pick an action according to epsilon-greedy policy

        Returns:
            chosen_action: (x,y)
        """
        actions = self.get_possible_actions(self.state)
        if np.random.rand() > self.epsilon:
            action, _ = self.get_max_action(self.state, self.experimental)
        else:
            action = actions[np.random.randint(0, len(actions))]
        return action

    def play(self):
        """Play a time_step of gridworld and return the (learned) reward
        """
        action = self.get_action()
        reward, s_prime = self.gridworld.time_step(self.state, action)
        self.update_Q(self.state, action, reward, s_prime)
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
