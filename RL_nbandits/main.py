import numpy as np
import multiprocessing as m
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from experiments.run_nArmBandit import run
from src.nArmBandit import Bandit

from src.experiment import ExperimentLog, Trial, Experiment

if __name__=='__main__':

    # temp function
    def trial_sum(args):
        return np.sum(args)

    # Run Trial
    ExperimentLog('Log')
    trial_odds = Trial(trial_sum, [1, 3, 5, 7], 'Odds')
    trial_evens = Trial(trial_sum, [2, 4, 6, 8], 'Evens')

    # Experiment
    experiment = Experiment([trial_odds, trial_evens], title='Sums')
    experiment.run_parallel(iterations=3)
    answer = experiment.experiment_logs

    print('Summation Answers ---- ')
    print(answer[0].log)
    print(answer[1].log)

    experiment.produce_plot(show=True)