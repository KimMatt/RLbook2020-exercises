import numpy as np
import multiprocessing as m
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from experiments.run_nArmBandit import run
from src.nArmBandit import Bandit

from src.experiment import ExperimentLog, Trial, Experiment


# trial function
def run_bandit(args = None):
    # get params
    bandit = args['bandit']
    iterations = args['iterations']

    # update, get scores
    scores = [0] * (iterations)
    for i in range(1, iterations):
        bandit.update()
        avg = bandit.get_average()
        scores[i] = avg 
    
    return scores


if __name__=='__main__':

    # Build Bandits
    # - bandit params
    n_arms = 10
    epsilons = [0, 0.1, 0.01]
    bandits = [Bandit(n_arms, e) for e in epsilons]
    # - set rewards
    rewards_control = [np.random.normal() for i in range(n_arms)]
    for bandit in bandits:
        bandit.set_rewards(rewards_control)

    # Build Trials
    niters = 10000
    trial_args = [{'bandit': bandit, 'iterations': niters} for bandit in bandits]
    trials = [Trial(run_bandit, trial_arg, 'Trial') for trial_arg in trial_args]

    # Experiment
    experiment = Experiment(trials, 'Test Epsilons {0, 0.1, 0.01}')
    logs = experiment.run_parallel(3)

    # Results -----------
    # - get scores
    avg_logs = [0] * len(bandits)
    for i, log in enumerate(logs):
        scores = log.log
        avg = [np.sum(x) / len(scores) for x in zip(*scores)]
        avg_logs[i] = avg

    # - plot beginning
    f = plt.figure()

    ax1 = f.add_subplot(1, 2, 1)
    ax2 = f.add_subplot(1, 2, 2)
    for avg, epsilon in zip(avg_logs, epsilons):
        ax1.plot(avg[:200], label = 'e: ' + str(epsilon))
        ax2.plot(avg, label = 'e: ' + str(epsilon))
    
    plt.legend()
    plt.show()

    