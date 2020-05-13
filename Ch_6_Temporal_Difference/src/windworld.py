import numpy as np

class WindWorld:

    def __init__(self):
        """Initialize windworld
        """
        self.winds = []
        ww_map = open("./maps/windworld.bin", "r").readlines()
        self.height = len(ww_map)
        self.width = len(ww_map[0])
        self.rewards = np.ndarray(shape=(self.width, self.height))

        for row_ind, row in enumerate(ww_map):
            for item_ind, item in enumerate(row):
                if row_ind == len(ww_map) - 1:
                    self.winds.append(item)
                else:
                    self.rewards[item_ind][row_ind] = item - 1


    def time_step(self, state, action):
        """given state and action return the reward and resulting state

        Args:
            state ([list]): current state [x,y]
            action ([list]): direction given in the form of [x,y]
        Returns:
            state: resulting state
            reward: integer reward, 1 if terminal -1 if not
        """
        # calculate result state
        result_state = np.add(state, action)
        result_state[1] += self.winds[state[0]]
        if result_state[0] >= self.width:
            result_state[0] = self.width - 1
        if result_state[0] < 0:
            result_state[0] = 0
        if result_state[1] < 0:
            result_state[1] = 0
        if result_state[1] >= self.height:
            result_state[1] = self.height - 1
        reward = self.rewards[result_state[0]][result_state[1]]
        # return reward of result state
        return reward, result_state
