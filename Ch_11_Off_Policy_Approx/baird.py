# baird's counter example implementation
import os
import numpy as np
import torch
import matplotlib
import pandas as pd

# whether or not we treat this as episodic or continuing task
episodic = True

# state vectors as per baird's example
state_vectors = np.array([[2,0,0,0,0,0,0,1],
                         [0,2,0,0,0,0,0,1],
                         [0,0,2,0,0,0,0,1],
                         [0,0,0,2,0,0,0,1],
                         [0,0,0,0,2,0,0,1],
                         [0,0,0,0,0,2,0,1],
                         [0,0,0,0,0,0,1,2]])

# initialize weights and average reward
weights = np.ones(9)
r_t = 1

# record of weight values to graph later
weights_record = [[val] for val in weights]

# b policy probabilities
dashed = 6.0/7.0

alpha = 0.1
gamma = 0.99

def step():
    """ Return action, next state and reward"""
    r = np.random.rand()
    if r < dashed:
        return 0, np.random.randint(6), 0
    else:
        return 1, 6, 0

def calc_max_q(state):
    """ Return q value of max action"""
    state_vector = state_vectors[state]
    s_as = np.array([np.concatenate((state_vector, [0])), 
                    np.concatenate((state_vector, [1]))])
    results = np.dot(s_as, weights.T)
    return max(results)

def update_weights_record():
    for i in range(len(weights_record)):
        weights_record[i].append(weights[i])

iterations = 2000
state = np.random.randint(7)

for i in range(iterations):
    action, next_state, reward = step()
    q_a_vect = np.concatenate((state_vectors[state], [action]))
    # Q(S, A) = Q(S, A) + alpha[R +  maxa Q(S', a) - Q(S, A)]
    if episodic:
        delta = (reward - r_t) + calc_max_q(next_state) - np.dot(q_a_vect, weights.T)
        r_t += alpha * delta
    else:
        delta = gamma * calc_max_q(next_state) - np.dot(q_a_vect, weights.T)
    # wt += alpha * delta * gradient(q(s,a,w)) 
    weights += alpha * delta * q_a_vect
    state = next_state
    update_weights_record()

graph = pd.DataFrame({"w" + str(i): weights_record[i] for i in range(len(weights_record))}).plot(
        kind="line", title="Exercise 11.3")
graph.set_xlabel("time_step")
f = graph.get_figure()
try:
    f.savefig("figs/ex11_2.png")
except:
    os.mkdir("figs")
    f.savefig("figs/ex11_2.png")

s_as = []
for state in range(7):
    state_vector = state_vectors[state]
    s_as.append([np.concatenate((state_vector, [0]))])
    s_as.append([np.concatenate((state_vector, [1]))])
results = np.dot(s_as, weights.T)
print("values: {}".format(results))
print("weights: {}".format(weights))