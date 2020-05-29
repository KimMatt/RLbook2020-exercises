import numpy as np


class States():

    def __init__(self, b, num_states):
        """Initialize the states.

        Args:
            b (int): branching factor
            num_states(int): number of total states
        """
        self.b = b
        self.num_states = num_states
        self.state_map = {}
        for state in range(num_states):
            transition_info = []
            for action in range(2):
                possible_states = []
                for branch in range(b):
                    # (reward, s_prime)
                    possible_states.append(
                        (np.random.normal(scale=1.0, loc=0.0), np.random.randint(num_states)))
                transition_info.append(possible_states)
            self.state_map[state] = transition_info

    def time_step(self, state, action):
        """[summary]

        Args:
            state (int): current state
            action (int): action (0 or 1) chosen

        Returns:
            reward: reward received on transition
            s_prime: int id of resulting state, None for terminal
        """
        if np.random.rand() < 0.1:
            return 0, None
        reward, s_prime = self.state_map[state][action][np.random.randint(self.b)]
        return reward, s_prime
