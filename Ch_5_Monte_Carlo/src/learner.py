import numpy as np

from src.agent import Agent
from src.racetrack import RaceTrack

class Learner:

    values = None
    actions = []
    agent = None
    track = None
    return_count = None
    epsilon = 0.2
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
        if np.random.rand() > self.epsilon:
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
                self.values[str(round_log[i])] = (
                    1/self.return_count[str(round_log[i])]) * (g - self.values[str(round_log[i])])
            state_action_count[str(round_log[i])] -= 1
