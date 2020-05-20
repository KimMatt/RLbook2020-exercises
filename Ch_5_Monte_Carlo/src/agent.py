import numpy as np

class Agent:

    position = None
    track = None
    velocity = None
    values = None

    def __init__(self, track):
        self.position = track.get_starting_position()
        self.track = track
        self.velocity = np.array([0,0])
        self.values = {}

    def bind_velocities(self):
        if self.velocity[0] >= 5:
            self.velocity[0] = 4
        if self.velocity[0] < 0:
            self.velocity[0] = 0
        if self.velocity[1] <= -5:
            self.velocity[1] = -4
        if self.velocity[1] > 0:
            self.velocity[1] = 0

    def play(self, action, greedy=False):
        """Play a timestep with velocity changes v1 v2

        Args:
            action ([list of ints]): [v1_increment, v2_increment]
        Returns:
            [reward]: integer for reward from this time step
        """
        # 0.1 percent chance of no change to velocity
        if np.random.rand() > 0.1:
            self.velocity = np.add(self.velocity, action)
        self.bind_velocities()
        self.position, reset = self.track.time_step(self.position, self.velocity)
        if reset:
            self.velocity = np.array([0, 0])
        if self.position[0] == self.track.f_line:
            return 1
        return -1

