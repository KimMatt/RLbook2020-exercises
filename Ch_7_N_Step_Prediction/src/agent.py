import numpy as np
from src.racetrack import RaceTrack

class Agent:

    position = None
    track = None
    velocity = None
    values = None

    def __init__(self, n):
        """Initialize our agent's parameters and data stores

        Args:
            n (int): n-step TD parameter for update step size
        """
        self.track = RaceTrack()
        self.n = n
        self.epsilon = 0.2
        self.alpha = 0.05
        self.discount_rate = 0.9
        self.actions = []
        for i in [1, 0, -1]:
            for j in [1, 0, -1]:
                self.actions.append(np.array([i, j]))
        self.values = {}
        for state in self.track.states.values():
            for i in range(6):
                for j in range(-5, 1):
                    self.values[str((state, np.array([i,j])))] = 0.0
        self.reset()

    def bind_velocity(self, velocity):
        """Bind our agent's velocities according to "speed limits"
        """
        if velocity[0] >= 5:
            velocity[0] = 4
        if velocity[0] < 0:
            velocity[0] = 0
        if velocity[1] <= -5:
            velocity[1] = -4
        if velocity[1] > 0:
            velocity[1] = 0
        return velocity

    def select_action(self):
        """Select an action using epsilon greedy policy

        Returns:
            [action]: select action
        """
        # if not exploiting select the max action
        if np.random.rand() > self.epsilon:
            max_action = None
            max_value = None
            for action in self.actions:
                # get resulting velocity
                velocity = np.array(self.bind_velocity(np.add(self.velocity, action)))
                action_val = self.values[str((self.position, velocity))]
                if max_value is None or action_val >= max_value:
                    max_value = action_val
                    max_action = action
            action = np.array(max_action)
        else:
            # if exploring select an action at random
            random = np.random.randint(0, len(self.actions))
            action = np.array(self.actions[random])
        return action

    def update_td(self):
        if self.t >= self.n - 1:
            G = np.sum([self.rewards[j] * (self.discount_rate ** i)
                        for i, j in enumerate([j for j in range(self.t - self.n + 1, self.t + 1)])])
            G += (self.discount_rate ** self.n) * self.values[self.states[self.t + 1]]
            # mathematically, the only difference between update_td and update_td_v_same is that
            # we are adding v_{t+n-1}(S_t+n) to G instead of V_{t}(S_t)
            update_state = self.states[self.t - self.n + 1]
            err = G - self.values[update_state]
            self.values[update_state] += self.alpha * err

    def update_td_v_same(self):
        self.update_values.append(self.values[self.states[self.t + 1]])
        if self.t >= self.n - 1:
            err = np.sum([(self.discount_rate ** i) * (self.rewards[j] + self.discount_rate * self.update_values[j+1]
                          - self.update_values[j]) for i,j in enumerate([k for k in range(self.t - self.n + 1, self.t + 1)])])
            update_state = self.states[self.t - self.n + 1]
            self.values[update_state] += self.alpha * err

    def play(self, mode=None):
        """Play a move on the racetrack, selecting an action
        Returns:
            [reward]: integer for reward from this time step
        """
        action = self.select_action()
        self.velocity = self.bind_velocity(np.add(self.velocity, action))
        reward, self.position, self.velocity = self.track.time_step(
            self.position, self.velocity)
        self.rewards.append(reward)
        self.states.append(str((self.position, self.velocity)))
        # update values V(s) <- V(s) + \alpha[R_n + \gamma G]
        if mode == 'v_same':
            self.update_td_v_same()
        else:
            self.update_td()
        self.t += 1
        return reward

    def reset(self):
        """Reset the agent's episode-dependent parameters
        """
        self.t = 0
        self.position = self.track.get_starting_position()
        self.velocity = np.array([0, 0])
        self.states = []
        self.rewards = []
        self.states.append(str((self.position, self.velocity)))
        self.update_values=[self.values[self.states[0]]]
