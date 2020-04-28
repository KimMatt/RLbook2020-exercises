
import pickle
import numpy as np
import math

def get_actions(state):
    """Return possible actions based on state
    Args:
        state ([list]): [l1, l2] # of cars at location 1 and 2
    Returns:
        [list]: list of integer actions
    """
    # maximum # of cars that can be moved from l1 to l2
    max_action = min(5, state[0], 10 - state[1])
    # maximum # of cars that can be moved from l2 to l1
    min_action = -1 * min(5, state[1], 10 - state[0])
    return [i for i in range(min_action, max_action+1)]


POISS_P = {}
FACTORIAL = {}


def poiss_p(n, e):
    """Returns the probability of number n being drawn from poisson with expected value e
    Args:
        n ([int]): number
        e ([int]): expected value
    Returns:
        [float]: probability of n
    """
    # need to sample the # of returns and rentals here
    if POISS_P.get(str([n, e])):
        return POISS_P.get(str([n, e]))
    if FACTORIAL.get(n):
        factorial = FACTORIAL.get(n)
    else:
        factorial = math.factorial(n)
    return (e**n / factorial) * (math.e ** (-1*e))


def apply_action(state, action):
    """Get the resulting state of applying action to state

    Args:
        state ([type]): [description]
        action ([type]): [description]
    """
    l1 = state[0] - action
    l2 = state[1] + action
    if l1 > 10 or l2 > 10 or l1 < 0 or l2 < 0:
        raise Exception
    return np.array([l1, l2])


def get_possible_states(rental_limit):
    """Return possible states given maximum car rentals per locations
    """
    possible_states = []
    for i in range(rental_limit+1):
        for j in range(rental_limit+1):
            possible_states.append([i, j])
    return np.array(possible_states)


POSS_EVENTS = {}


def get_possible_events(curcar_ns, m1, m2):
    """Return a list of possible events for a single location

    Args:
        curcar_ns ([int]): current number of cars in the parking lot
        m1 ([int]): mean of the rental poisson
        m2 ([int]): mean of the return poisson
    Returns:
        possible_events ([list]): [([rentals, returns], prob)],...,[]]
    """
    possible_events = []
    if POSS_EVENTS.get(str((curcar_ns, m1, m2))):
        return POSS_EVENTS.get(str((curcar_ns, m1, m2)))
    for rentals in range(0, curcar_ns):
        for returns in range(0, 10-(curcar_ns-rentals)):
            probability = poiss_p(rentals, m1) * poiss_p(returns, m2)
            possible_events.append(([rentals, returns], probability))
    POSS_EVENTS[str((curcar_ns, m1, m2))] = possible_events
    return possible_events


def get_reward(mid_state, l1_rentals, l2_rentals):
    """Get reward not including cost from action

    Args:
        mid_state ([list of ints]): [l1, l2]
        l1_rentals ([int]): # of rentals at l1
        l2_rentals ([int]): # of rentals at l2

    Returns:
        reward: numerical reward
    """
    return 10 * (min(l1_rentals, mid_state[0]) + min(l2_rentals, mid_state[1]))


ALL_POSS_EVENTS = {}


def get_all_possible_events(mid_state, action):
    """Get all possible events to occur to state with their probabilities

    Args:
        state ([list]): [l1, l2] # of cars at l1 and l2
    Returns:
        possible_events ([list]): [[[l1, l2], reward, prob]],...,[]]
    """
    mid_state = np.array(mid_state)
    if ALL_POSS_EVENTS.get(str((mid_state, action))):
        possible_events = ALL_POSS_EVENTS.get(str((mid_state, action)))
    else:
        possible_events = []
        possible_events_l1 = get_possible_events(mid_state[0], 3.0, 4.0)
        possible_events_l2 = get_possible_events(mid_state[1], 3.0, 2.0)
        for l1_event in possible_events_l1:
            for l2_event in possible_events_l2:
                reward = get_reward(mid_state, l1_event[0][0], l2_event[0][0])
                probability = l1_event[1] * l2_event[1]
                result_state = mid_state[:]
                change = [l1_event[0][1] - l1_event[0]
                          [0], l2_event[0][1] - l2_event[0][0]]
                result_state = np.add(mid_state, change)
                possible_events.append([result_state, reward, probability])
        ALL_POSS_EVENTS[str((mid_state, action))] = np.array(possible_events)
    # add action penalty to reward
    action_cost = -2 * abs(action)
    np.add(possible_events, np.array(
        [[[0, 0], action_cost, 0] for i in range(len(possible_events))]))
    return possible_events


def construct_q_map(size):
    """Constructs the q_map to map [state,action] to info needed for their q function

    Args:
        possible_states ([list]): list of all possible [l1,l2] states for the problem

    Returns:
        q_map: [state,action] -> [[s', reward, prob]],...,[]]
    """
    possible_states = get_possible_states(size)
    q_map = {}
    for state in possible_states:
        for action in get_actions(state):
            mid_state = apply_action(state, action)
            possible_events = get_all_possible_events(mid_state, action)
            q_map[str([state, action])] = possible_events
    return q_map

if __name__ == "__main__":

    Q_MAP = construct_q_map(10)
    print("dumping")
    pickle.dump(Q_MAP, open("./pickles/jacks_qmap.p", "wb"))
