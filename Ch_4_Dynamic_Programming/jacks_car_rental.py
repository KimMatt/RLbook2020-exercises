# jacks_car_rental.py
#
# Implementation of exercise 4.5 of jack's car rental problem

import pickle
import numpy as np


def get_actions(state):
    """Return possible actions based on state
    Args:
        state ([list]): [l1, l2] # of cars at location 1 and 2
    Returns:
        [list]: list of integer actions
    """
    # maximum # of cars that can be moved from l1 to l2
    max_action = min(3, state[0], 12 - state[1])
    # maximum # of cars that can be moved from l2 to l1
    min_action = -1 * min(3, state[1], 12 - state[0])
    return [i for i in range(min_action, max_action+1)]


def get_possible_states(rental_limit):
    """Return possible states given maximum car rentals per locations
    """
    possible_states = []
    for i in range(rental_limit+1):
        for j in range(rental_limit+1):
            possible_states.append([i,j])
    return np.array(possible_states)

def get_V(state, v):
    state = np.array(state)
    if v.get(str(state)):
        return v.get(str(state))
    return 0.0

def set_V(state, value, v):
    state = np.array(state)
    v[str(state)] = value

def q(state, action, q_map, v):
    outcomes = q_map[str([state, action])]
    d_factor = 0.9
    q = 0.0
    for outcome in outcomes:
        q += outcome[2] * (outcome[1] + (d_factor * get_V(outcome[0], v)))
    return q

def set_PI(action, state, value, pi):
    action = int(action)
    state = np.array(state)
    pi[str((action, state))] = value

def get_PI(action, state, pi):
    action = int(action)
    state = np.array(state)
    return pi[str((action, state))]

def policy_eval(v, pi, q_map, possible_states):
    epsilon_reached = False
    epsilon = 0.001
    while not epsilon_reached:
        delta = 0.0
        for state in possible_states:
            old_v = get_V(state, v)
            set_V(state, np.sum([get_PI(action, state, pi) * q(state,
                                                               action, q_map, v) for action in get_actions(state)]), v)
            delta = max(delta, old_v - get_V(state, v))
        if delta < epsilon:
            epsilon_reached = True

def policy_imp(v, pi, q_map, possible_states):
    policy_stable = True
    for state in possible_states:
        old_pi = dict.copy(pi)
        actions = get_actions(state)
        q_optimal = q(state, actions[0], q_map, v)
        optimal_actions = []
        old_optimal_actions = []
        for action in actions:
            q_as = q(state, action, q_map, v)
            if get_PI(action, state, old_pi) > 0.0:
                old_optimal_actions.append(action)
            if q_as > q_optimal:
                optimal_actions = [action]
                q_optimal = q_as
            elif q_as == q_optimal:
                optimal_actions.append(action)
        for action in actions:
            if action in optimal_actions:
                set_PI(action, state, 1.0 / len(optimal_actions), pi)
            else:
                set_PI(action, state, 0.0, pi)
        if old_optimal_actions != optimal_actions:
            policy_stable = False
    return policy_stable


def initialize_pi(pi, possible_states):
    for state in possible_states:
        actions = get_actions(state)
        for action in actions:
            set_PI(action, state, 0.0, pi)
        set_PI(0, state, 1.0, pi)


if __name__ == "__main__":

    Q_MAP = pickle.load(open("./pickles/jacks_qmap.p", "rb"))
    V = {}
    PI = {}
    POSSIBLE_STATES = get_possible_states(12)
    initialize_pi(PI, POSSIBLE_STATES)
    POLICY_STABLE = False

    while not POLICY_STABLE:
        print("eval")
        policy_eval(V, PI, Q_MAP, POSSIBLE_STATES)
        print("improve")
        POLICY_STABLE = policy_imp(V, PI, Q_MAP, POSSIBLE_STATES)

    policy_eval(V, PI, Q_MAP, POSSIBLE_STATES)

    pickle.dump(V, open("./pickles/jacks_values.p", "wb"))
    pickle.dump(PI, open("./pickles/jacks_policies.p", "wb"))
