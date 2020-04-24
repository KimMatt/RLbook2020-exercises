# compute_gridworld.py
#
# Used to compute the optimal solutions for the gridworld example.
import numpy as np

coords = []
for x in range (0,5):
    for y in range (0,5):
        coords.append([x,y])
coords = np.array(coords)

d_factor = 0.9

directions = np.array([[-1,0],[1,0],[0,1],[0,-1]])


def get_actions(state):
    """Returns a list of the neighboring coords to go to as actions

    Args:
        coord ([list]): [x,y]

    Returns:
        [list]: list of [x,y] states
    """
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


def calc_v_star(k_limit):

    A = {}

    def get_A(coord, k):
        k = float(k)
        coord = np.array(coord)
        if A.get(str([coord, k])):
            return A.get(str([coord,k]))
        return 0.0

    def set_A(coord, k, value):
        k = float(k)
        coord = np.array(coord)
        A[str([coord,k])] = value

    for k in range(k_limit-1,-1,-1):
        k = float(k)
        for coord in coords:
            actions = get_actions(coord)
            value = [get_reward(coord, action) + (d_factor**k * get_A(action, k+1.0)) for action in actions]
            value = np.max(value)
            set_A(coord, k, value)

    return A

if __name__ == "__main__":

    A = calc_v_star(100000)
    coord = np.array([0,1])
    coord_b = np.array([4,1])
    k = float(0)
    print(A[str([coord,k])])
    print(A[str([coord_b, float(1)])])
