# compute_gridworld.py
#
# Used to compute the optimal solutions for the gridworld example from chapter 4 (non discounted, episodic)
import numpy as np


def get_actions(state):
    """Returns a list of the neighboring coords to go to as actions
    Args:
        state ([list]): [x,y]
    Returns:
        [list]: list of [x,y] states
    """

    if all(np.equal(state, [4, 1])):
        return np.array([[3,0],[3,1],[3,2],[4,1]])

    directions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])

    actions = []

    for direction in directions:
        action = np.add(state, direction)
        if (action[0] > 3 or action[0] < 0 or action[1] > 3 or action[1] < 0) and not all(np.equal(action, [4, 1])):
            actions.append(state)
        else:
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
    return -1.0


def get_coords():
    coords = []
    for row in range(4):
        for col in range(4):
            if (row == 3 and col == 3) or (row == 0 and col == 0):
                continue
            coords.append([row, col])
    coords.append([4, 1])
    return np.array(coords)


def calc_v_star():

    A = {}
    coords = get_coords()

    def get_A(coord):
        coord = np.array(coord)
        if A.get(str(coord)):
            return A.get(str(coord))
        return 0.0

    def get_total_v():
        total_v = 0.0
        for coord in coords:
            total_v += get_A(coord)
        return total_v

    def set_A(coord, value):
        coord = np.array(coord)
        A[str(coord)] = value

    diff = 1.0

    while diff > .0000001:
        v = get_total_v()
        for coord in coords:
            actions = get_actions(coord)
            value = np.sum([0.25 * (get_reward(coord, action) + get_A(action))
                     for action in actions])
            set_A(coord, value)
        diff = abs(v - get_total_v())
    return A


if __name__ == "__main__":

    A = calc_v_star()
    coord = np.array([4, 1])
    print(A[str(coord)])
