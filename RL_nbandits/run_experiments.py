# Fill in this file
from src.agent import SimpleAgent
from src.bandit import NBandits
from src.experiment import Experiment, Trial


if __name__ == "__main__":

    def run_iteration(kwargs):
        agent = kwargs["agent"]
        if agent.total_plays % 100:
            agent.bandit.random_walk()
        return agent.play()

    bandit_1 = NBandits(10)
    bandit_2 = NBandits(10)
    agent_1 = SimpleAgent(0.1, 0, bandit_1)
    agent_2 = SimpleAgent(0.1, 0, bandit_2, method="constant", constant=0.1)

    trial_one = Trial(run_iteration, {"agent": agent_1}, "sample average method")
    trial_two = Trial(run_iteration, {"agent": agent_2}, "constant update method")

    experiment = Experiment([trial_one, trial_two], "Exercise 2.4")
    experiment.run_parallel(2000)

    experiment.produce_plot("reward")
