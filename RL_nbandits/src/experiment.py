# experiment.py
# A combination of Classes to use to run an experiment.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool


class ExperimentLog:

    def __init__(self, label):
        """Initialize a new instance of Experiment Log

        Args:
            label ([string]): Label for experiment
        """
        self.label = label
        self.log = []

    def append(self, item):
        self.log.append(item)


class Trial:

    def __init__(self, trial_function, parameters, title):
        """Creates a Trial object to run trial function with given parameters

        Args:
            trial_function ([function]): a function that is expected to return
                a single scalar and only has one kwargs as its argument
            parameters ([dictionary]): a dictionary of kwargs to pass to trial function
            title ([string]): a title for the Trial
        """
        self.trial_function = trial_function
        self.parameters = parameters
        self.title = title

    def run(self, iterations):
        """Runs the trial for given iterations

        Args:
            iterations ([int]): number of iterations to run the trial for

        Returns:
            experiment_log ([ExperimentLog]): A log of the experiment to be used
                to create plots of the result
        """
        experiment_log = ExperimentLog(self.title)
        experiment_log.log = [self.trial_function(self.parameters) for i in range(iterations)]
        return experiment_log


class Experiment:

    @staticmethod
    def run_trial(kwargs):
        trial = kwargs.get("trial")
        iterations = kwargs.get("iterations")
        return trial.run(iterations)

    def __init__(self, trials, title):
        """Initialize the Experiment object

        Args:
            trials ([Trial]): List of trials to run
            title ([string]): Title of experiment
        """
        self.trials = trials
        self.title = title
        self.experiment_logs = []

    def run_parallel(self, iterations):
        """Run the trials and save their results

        Args:
            iterations ([int]): number of iterations to run for every trial
        """
        with Pool(len(self.trials)) as p:
            pool_args = [{"trial": trial, "iterations": iterations}
                         for trial in self.trials]
            experiment_logs = p.map(Experiment.run_trial, pool_args)
        self.experiment_logs = experiment_logs
        return experiment_logs

    def produce_plot(self, y_label=None, show=False):
        """Produces and saves a plot resulting from the experiment

        Args:
            y_label ([string]): optionally add a y label
        """
        data = pd.DataFrame({e_log.label: e_log.log for e_log in self.experiment_logs})
        graph = data.plot(kind="line", title=self.title)
        graph.set_xlabel("iterations")
        f = graph.get_figure()
        if y_label:
            graph.set_ylabel(y_label)

        try:
            f.savefig("figs/" + self.title + ".svg")
        except:
            os.mkdir("figs")
            f.savefig("figs/" + self.title + ".svg")

        if show:
            plt.show()
