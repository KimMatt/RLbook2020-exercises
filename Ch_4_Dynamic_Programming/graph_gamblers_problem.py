# graph_gamblers_problem.py
#
# code to graph the results of running gamblers problem

import matplotlib
import pandas as pd
import pickle


if __name__ == "__main__":
    value_function = pickle.load(open("./pickles/gamblers_v.p", "rb"))
    policy = pickle.load(open("./pickles/gamblers_pi.p", "rb"))
    max_actions = []
    for state in range(1,100):
        max_action = 1
        max_policy = 0.0
        for action in range(1, min(state, 100-state)+1):
            if policy.get((action,state)) > max_policy:
                max_action = action
                max_policy = policy.get((action, state))
        max_actions.append(max_action)

    values = pd.DataFrame({"values": value_function})
    graph = values.plot(kind="line", title="gamblers value to state")
    graph.set_xlabel("Capital")
    graph.set_ylabel("Value estimates")
    f = graph.get_figure()
    f.savefig("./figs/gamblers_values_051.png")

    policies = pd.DataFrame({"optimal_action": max_actions})
    graph = policies.plot(kind="line", title="optimal action to state")
    graph.set_xlabel("Capital")
    graph.set_ylabel("Final policy (stake)")
    f = graph.get_figure()
    f.savefig("./figs/gamblers_policies_051.png")
