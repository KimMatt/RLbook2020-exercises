# gamblers_problem.py
#
# implementation of gambler's problem from chapter 4
import numpy as np
import pickle

def policy_eval(v, pi, p_w, p_l):
    epsilon = 0.001
    epsilon_reached = False
    while not epsilon_reached:
        delta = 0.0
        for state in range(1, 100):
            v_old = v[state]
            w_vals = np.sum([p_w * pi[(action, state)]*v[state + action] for action in range(1, min(state, 100 - state) + 1)])
            l_vals = np.sum([p_l * pi[(action, state)]*v[state - action] for action in range(1, min(state, 100 - state) + 1)])
            v[state] = w_vals + l_vals
            delta = max(delta, abs(v_old-v[state]))
        if delta < epsilon:
            epsilon_reached = True


def policy_imp(v, pi, p_w, p_l):
    policy_stable = True
    old_pi = dict.copy(pi)
    for state in range(1, 100):
        old_optimal_actions = []
        new_optimal_actions = []
        optimal_q = 0.0
        for action in range(1, min(state, 100-state)+1):
            # create a list of optimal actions from given policy
            if old_pi[(action, state)] > 0.0:
                old_optimal_actions.append(action)
            # calculate q(s,a)
            q = (p_w * v[state + action]) + (p_l * v[state - action])
            if q > optimal_q:
                new_optimal_actions = [action]
                optimal_q = q
            elif optimal_q == q:
                new_optimal_actions.append(action)
        for action in range(1, min(state, 100-state)+1):
            if action in new_optimal_actions:
                pi[(action, state)] = 1.0 / len(new_optimal_actions)
            else:
                pi[(action, state)] = 0.0
        if old_optimal_actions != new_optimal_actions:
            policy_stable = False
    return policy_stable


def value_iter(v, pi, p_w, p_l):
    epsilon = 0.2
    epsilon_reached = False
    for i in range(0,32):
        delta = 0.0
        v_total = 0
        v_old_total = sum(v)
        for state in range(1, 100):
            v_old = v[state]
            w_vals = np.array([p_w * v[state + action] for action in range(1, min(state, 100 - state) + 1)])
            l_vals = np.array([p_l * v[state - action] for action in range(1, min(state, 100 - state) + 1)])
            q_values = np.add(w_vals, l_vals)
            v[state] = np.max(q_values)
            delta = max(delta, abs(v_old - v[state]))
        if epsilon > delta:
            epsilon_reached = True

if __name__ == "__main__":
    V = [0 for i in range(101)]
    V[100] = 1
    PI = {}
    for s in range(1, 100):
        for a in range(1, min(s, 100-s)+1):
            PI[(a, s)] = 1.0/(min(s, 100-s))
    p_w = 0.51
    p_l = 0.49

    # value iteration
    """value_iter(V, PI, p_w, p_l)
    policy_imp(V, PI, p_w, p_l)"""

    # policy iteration
    POLICY_STABLE = False
    while not POLICY_STABLE:
        print("eval")
        policy_eval(V, PI, p_w, p_l)
        print("improve")
        POLICY_STABLE = policy_imp(V, PI, p_w, p_l)
    pickle.dump(V, open("./pickles/gamblers_v.p", "wb"))
    pickle.dump(PI, open("./pickles/gamblers_pi.p", "wb"))
