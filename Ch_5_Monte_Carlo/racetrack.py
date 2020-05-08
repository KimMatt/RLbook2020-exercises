# racetrack.py
#
# implementation of racetrack problem from chapter 5
import pickle
import math
import numpy as np

class RaceTrack:

    states = None

    def __init__(self):
        race_map = open("./map.bin", "r")
        lines = race_map.readlines()
        self.states = {}
        # cheat
        # taking into account the /n character
        self.f_line = len(lines[0]) - 2
        self.start_positions = []

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char != "\n" and int(char) == 1:
                    self.states[str(np.array([x, y]))] = np.array([x,y])
                    if y == len(lines) - 1:
                        self.start_positions.append(np.array([x,y]))

    def get_starting_position(self):
        start_position = np.random.randint(0, len(self.start_positions))
        return self.start_positions[start_position]

    def check_f_line_legal_cross(self, start_pos, end_pos):
        if end_pos[0] >= self.f_line:
            # height
            y_dist = abs(start_pos[1] - end_pos[1])
            if y_dist == 0:
                y_intercept = end_pos[1]
            else:
                # width
                x_dist = abs(start_pos[0] - end_pos[0])
                # |\ the top angle
                angle = math.atan(x_dist/y_dist)
                # horizontal distance between finish line and start line
                x_intercept_dist = abs(self.f_line - start_pos[0])
                y_intercept_dist = x_intercept_dist / math.tan(angle)
                sign = 1 if end_pos[1] - start_pos[1] >= 0 else -1
                y_intercept = start_pos[1] + (sign * y_intercept_dist)
            if y_intercept >= 0 and y_intercept <= 5:
                return True, math.floor(y_intercept)
        return False, None

    def time_step(self, position, velocities):
        new_position = np.add(velocities, position)
        reset = False
        # to correct for overshooting the finish line
        legal_cross, y_intercept = self.check_f_line_legal_cross(position, new_position)
        if legal_cross:
            new_position[0] = self.f_line
            new_position[1] = y_intercept
        if self.states.get(str(np.array(new_position))) is None:
            new_position = self.get_starting_position()
            reset = True
        return new_position, reset



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

    def play(self, action):
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
        if self.position[0] >= self.track.f_line:
            return 1
        return -1


class Learner:

    values = None
    actions = []
    agent = None
    track = None
    return_count = None
    epsilon = 0.3
    discount_rate = 0.9

    def __init__(self):
        for i in [1, 0, -1]:
            for j in [1, 0, -1]:
                self.actions.append(np.array([i, j]))
        self.track = RaceTrack()
        self.values = {}
        for state in self.track.states.values():
            for action in self.actions:
                self.values[str((state, action))] = 0.0
        self.return_count = {}

    def select_action(self):
        # if not exploiting select the maximal action
        if np.random.rand() > (self.epsilon - (self.epsilon / len(self.actions))):
            max_action = None
            max_value = self.values.get(str((self.agent.position, self.actions[0])))
            for action in self.actions:
                action_val = self.values.get(str((self.agent.position, action)))
                if action_val >= max_value:
                    max_value = action_val
                    max_action = action
            return np.array(max_action)
        # if exploring select an action at random
        random = np.random.randint(0,len(self.actions))
        return np.array(self.actions[random])

    def play_round(self):
        self.agent = Agent(self.track)
        result = -1
        state_action_count = {}
        round_log = []
        while result != 1:
            action = self.select_action()
            a_s = (self.agent.position, action)
            round_log.append(a_s)
            if state_action_count.get(str(a_s)) is None:
                state_action_count[str(a_s)] = 0
            state_action_count[str(a_s)] += 1
            result = self.agent.play(action)
        g = 0
        for i in range(len(round_log) - 1, -1, -1):
            r = 1 if i == len(round_log) - 1 else -1
            g = self.discount_rate * g + r
            if state_action_count[str(round_log[i])] == 1:
                # if the count of how many samples have been used in the average is None
                if self.return_count.get(str(round_log[i])) is None:
                    self.return_count[str(round_log[i])] = 0
                # k += 1
                self.return_count[str(round_log[i])] += 1
                # g_t+1 = g_t + 1/k * (r_t+1)
                self.values[str(round_log[i])] += (1/self.return_count[str(round_log[i])]) * g
            state_action_count[str(round_log[i])] -= 1


if __name__ == "__main__":

    learner = Learner()

    for i in range(5000):
        print("ROUND {}".format(i))
        learner.play_round()

    pickle.dump(learner.values, open("./pickles/race_values.p", "wb"))
