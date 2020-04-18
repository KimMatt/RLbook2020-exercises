# experiment.py
# class Experiment used to run experiments
import numpy as np
import pandas as pd
from multiprocessing import Pool


class ExperimentLog:

    def __init__(self, unit, dtype, label):
        """Initialize a new instance of Experiment Log

        Args:
            label ([string]): Label for experiment
            unit ([string]): x or y label for unit
            dtype ([class]): datatype to pass to pandas like float or int
        """
        self.label = label
        self.unit = unit
        self.dtype = dtype
        self.log = []

    def append(self, item):
        self.log.append(item)


class Trial:

    def __init__(self, trial_function, parameters, title, dtype, unit):
        """Creates a Trial object to run trial function with given parameters

        Args:
            trial_function ([function]): a function that is expected to return
                a single scalar
            parameters ([dictionary]): a dictionary of kwargs to pass to trial function
            title ([string]): a title for the Trial
        """
        self.trial_function = trial_function
        self.parameters = parameters
        self.title = title
        self.dtype = dtype
        self.unit = unit

    def run(self, iterations):
        """Runs the trial for given iterations

        Args:
            iterations ([int]): number of iterations to run the trial for

        Returns:
            experiment_log ([ExperimentLog]): A log of the experiment to be used
                to create plots of the result
        """
        experiment_log = ExperimentLog(self.unit, self.dtype, self.title)
        for i in range(iterations):
            experiment_log.append(self.trial_function(**parameters))
        return experiment_log


class Experiment:

    @staticmethod
    def run_trial(**kwargs):
        trial = kwargs.get("trial")
        iterations = kwargs.get("iterations")
        trial.run(iterations)

    def __init__(self, trials, title, iterations):
        """Initialize the Experiment object

        Args:
            trials ([Trial]): List of trials to run
            title ([string]): Title of experiment
        """
        self.trials = trials
        self.title = title
        self.iterations = iterations

    def run_parallel(self, iterations):
        """Run the trials and save their results

        Args:
            iterations ([int]): number of iterations to run for every trial
        """
        with Pool(len(self.trials)) as p:
            pool_args = [{"trial": trial, "iterations": self.iterations}
                         for trial in self.trials]
            experiment_logs = p.map(Experiment.run_trial, pool_args)
        self.experiment_logs = experiment_logs

    def produce_plot(self, y_label=None):
        """Produces and saves a plot resulting from the experiment

        Args:
            y_label ([string]): optionally add a y label
        """
        data = pd.DataFrame({e_log.title: e_log.log for e_log in self.experiment_logs})
        graph = data.plot(kind="line", title=self.title).get_figure()
        graph.set_xlabel("iterations")
        if y_label:
            graph.set_ylabel(y_label)
        graph.save_fig(self.title + ".svg")
