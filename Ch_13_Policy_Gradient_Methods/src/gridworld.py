# GridWorld.py

class GridWorld():

    def __init__(self):
        """Initialize gridworld, load map
        """
        #(x,y) -> reward
        self.tiles = {}
        gw_map = open("./maps/map.bin", "r").readlines()
        for row_ind, row in enumerate(gw_map):
            for col_ind, col in enumerate(row):
                if col != "\n" and col != " ":
                    self.tiles[(col_ind, row_ind)] = int(col)

    def time_step(self, state, action):
        """run a time step given state and action and give the resulting reward
        and state

        Args:
            state ((x,y)): given state
            action ((x_d, y_d)): given action
        """
        s_prime = (state[0] + action[0], state[1] + action[1])
        reward = -0.1 if self.tiles[s_prime] == 0 else 1
        return reward, s_prime
