import numpy as np
from src.windworld import WindWorld

class Agent:

    def __init__(self):
        """Initialize agent at given start_state to play on given gridworld

        Args:
            start_state ([list]): [x,y]
            gridworld ([Object]): GridWorld object
        """
        self.windworld = WindWorld()
        self.spawn()
        self.Q = {}
        self.alpha = 0.5
        self.t = 1
        self.epsilon = 0.1
        self.actions = []
        # initialize actions
        for x in range(-1,2):
            for y in range(-1,2):
                if x== 0 and y==0:
                    continue
                else:
                    self.actions.append(np.array([x,y]))
        # initialize Q values to 0
        for x in range(self.windworld.width):
            for y in range(self.windworld.height):
                for action in self.actions:
                    self.Q[str((x,y,action[0],action[1]))] = 0

    def spawn(self):
        """spawns the agent at a random start_state
        """
        # x = np.random.randint(0,self.windworld.width)
        # y = np.random.randint(0,self.windworld.height)
        # self.state = np.array([x,y])
        self.state = np.array([0,3])


    def select_action(self):
        """based on the agent's state and q values, select an action
        """
        # greedy
        if np.random.rand() > self.epsilon:
            max_q_value = self.Q[str(
                (self.state[0], self.state[1], self.actions[0][0], self.actions[0][1]))]
            max_action = self.actions[0]
            for action in self.actions:
                q_value = self.Q[str(
                    (self.state[0], self.state[1], action[0], action[1]))]
                if q_value >= max_q_value:
                    max_q_value = q_value
                    max_action = action
            action = max_action
        # not greedy
        else:
            action = self.actions[np.random.randint(0, len(self.actions))]
        # update t, and epsilon
        # self.t += 1
        # self.epsilon = 1/self.t
        return action

    def play(self, action):
        """from whatever present state, choose an action and take it.
            apply updates to Q
            Returns:
                next_action: next action selected
        """
        # play action
        reward, result_state = self.windworld.time_step(self.state, action)
        # update state
        current_state = np.copy(self.state)
        self.state = result_state
        # select next action
        next_action = self.select_action()
        # update action value
        current_qval = self.Q[str(
            (current_state[0], current_state[1], action[0], action[1]))]
        next_q_val = self.Q[str((result_state[0], result_state[1], next_action[0], next_action[1]))]
        self.Q[str((current_state[0], current_state[1], action[0], action[1]))] += reward + next_q_val - current_qval
        return reward, next_action
