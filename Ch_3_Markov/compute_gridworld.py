# compute_gridworld.py
#
# Used to compute the optimal solutions for the gridworld example.
import numpy as np


def get_actions(state):
    """Returns a list of the neighboring coords to go to as actions

    Args:
        coord ([list]): [x,y]

    Returns:
        [list]: list of [x,y] states
    """
    directions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])

    if all(np.equal(state, [0, 1])):
        return np.array([[4, 1]])
    if all(np.equal(state, [0, 3])):
        return np.array([[2, 3]])

    actions = []

    for direction in directions:
        action = np.add(state, direction)
        if action[0] > 4 or action[0] < 0 or action[1] > 4 or action[1] < 0:
            continue
        actions.append(action)
    return np.array(actions)


def get_reward(state_one, state_two):
    """Returns the reward from going from state_one to state_two

    Args:
        state_one ([array]): [x,y]
        state_two ([array]): [x,y]

    Returns:
        [float]: reward
    """
    if all(np.equal(state_one, [0, 1])) and all(np.equal(state_two, [4, 1])):
        return 10.0
    if all(np.equal(state_one, [0, 3])) and all(np.equal(state_two, [2, 3])):
        return 5.0
    return 0.0


def get_coords():
    """Returns all possible states on the grid problem

    Returns:
        coords: np array of states
    """
    coords = []
    for x in range(0, 5):
        for y in range(0, 5):
            coords.append([x, y])
    return np.array(coords)



def calc_v_star():
    """Perform value iteration over the grid problem

    Returns:
        V: dictionary containing the (state) -> state value
    """
    V = {}
    coords = get_coords()
    d_factor = 0.9

    def get_V(coord):
        coord = np.array(coord)
        if V.get(str(coord)):
            return V.get(str(coord))
        return 0.0

    def get_total_V():
        total_v = 0.0
        for coord in coords:
            total_v += get_V(coord)
        return total_v

    def set_V(coord, value):
        coord = np.array(coord)
        V[str(coord)] = value

    diff = 1.0

    while diff > .00001:
        v = get_total_V()
        for coord in coords:
            actions = get_actions(coord)
            value = [get_reward(coord, action) + (d_factor * get_V(action)) for action in actions]
            value = np.max(value)
            set_V(coord, value)
        diff = abs(v - get_total_V())

    return V

if __name__ == "__main__":

    V = calc_v_star()
    coord = np.array([0, 1])
    print(V[str(coord)])
