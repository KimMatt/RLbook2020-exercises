# gamblers_problem.py
#
# implementation of gambler's problem from chapter 4
import numpy as np
import pickle

def policy_eval(v, pi, p_w, p_l):
    epsilon = 0.1
    while True:
        delta = 0.0
        for state in range(1,100):
            v_old = v[state]
            w_vals = np.sum([p_w * pi[(action, state)]*v[state + action] for action in range(0, min(state, 100 - state) + 1)])
            l_vals = np.sum([p_l * pi[(action, state)]*v[state - action] for action in range(0, min(state, 100 - state) + 1)])
            v[state] = w_vals + l_vals
            delta = max(delta, abs(v_old-v[state]))
        if delta < epsilon:
            return v


def policy_imp(v, pi, p_w, p_l):
    policy_stable = True
    old_pi = dict.copy(pi)
    for state in range(1,100):
        old_optimal_actions = []
        new_optimal_actions = []
        optimal_q = -1
        for action in range(0, min(state, 100-state)+1):
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
        for action in range(0, min(state, 100-state)+1):
            if action in new_optimal_actions:
                pi[(action, state)] = 1.0 / len(new_optimal_actions)
            else:
                pi[(action, state)] = 0.0
        if old_optimal_actions != new_optimal_actions:
            policy_stable = False
    return policy_stable


if __name__ == "__main__":
    v = [0 for i in range(101)]
    v[100] = 1
    pi = {}
    for s in range(1, 100):
        for a in range(0, min(s,100-s)+1):
            pi[(a, s)] = 1.0/(s+1)
    p_w = 0.4
    p_l = 0.6

    policy_stable = False
    while not policy_stable:
        print("evaluating policy")
        policy_eval(v, pi, p_w, p_l)
        print("stabilizing policy")
        policy_stable = policy_imp(v, pi, p_w, p_l)
    pickle.dump(v, open("gamblers_v.p", "wb"))
    pickle.dump(pi, open("gamblers_pi.p", "wb"))
