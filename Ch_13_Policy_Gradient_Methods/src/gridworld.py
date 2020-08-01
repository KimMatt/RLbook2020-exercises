# GridWorld.py

class GridWorld():

    tile_to_reward = {0:-0.1, 2:-10, 1:0.1}
    action_to_deltas = {1: (-1,0),
                        2: (1,0),
                        3: (0,1),
                        4: (0,-1)}

    def __init__(self):
        """Initialize gridworld, load map."""
        self.tiles = {}
        self.start_state = (3,0)
        gw_map = open("./maps/cliff.bin", "r").readlines()
        for row_ind, row in enumerate(gw_map):
            for col_ind, col in enumerate(row):
                if col != "\n" and col != " ":
                    self.tiles[(col_ind, row_ind)] = int(col)
        self.height = len(gw_map)
        self.width = len(gw_map[0])

    def time_step(self, state, action):
        """
        Run a time step given state and action and give the resulting reward
        and state

        Args:
            state ((x,y)): given state
            action ((x_d, y_d)): given action
        """
        delta = self.action_to_deltas[action]
        s_prime = (state[0] + delta[0], state[1] + delta[1])
        # if it goes out of bounds
        if self.tiles.get(s_prime) is None:
            s_prime = state
        reward = self.tile_to_reward[self.tiles[s_prime]]
        if reward == -10:
            s_prime = self.start_state
        return reward, s_prime
