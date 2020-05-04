# graph_jacks_car_rental.py
#
# contains code for visualizing the results of solving jack's car rental problem

import matplotlib
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

values = pickle.load(open("./pickles/jacks_values.p", "rb"))
policy = pickle.load(open("./pickles/jacks_policies.p", "rb"))

l1 = []
l2 = []
v = []
opt_actions = []

for i in range(0, 13):
    for j in range(0, 13):
        l1.append(i)
        l2.append(j)
        v.append(values.get((str(np.array([i, j])))))
        # maximum # of cars that can be moved from l1 to l2
        max_action = min(3, i, 12 - j)
        # maximum # of cars that can be moved from l2 to l1
        min_action = -1 * min(3, j, 12 - i)
        actions = [i for i in range(min_action, max_action+1)]
        optimal_action = actions[0]
        optimal_value = policy[str((actions[0], np.array([i, j])))]
        for action in actions:
            if policy[str((action, np.array([i, j])))] > optimal_value:
                optimal_action = action
                optimal_value = policy[str((action, np.array([i, j])))]
        opt_actions.append(optimal_action)



fig = plt.figure()

ax = Axes3D(fig)

df = pd.DataFrame({'x': l1, 'y': l2, 'z': v}, index=range(len(l1)))

ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.jet, linewidth=0.2)
plt.show()

df = pd.DataFrame({'x': l1, 'y': l2, 'z': opt_actions}, index=range(len(l1)))

ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.jet, linewidth=0.2)
plt.show()

