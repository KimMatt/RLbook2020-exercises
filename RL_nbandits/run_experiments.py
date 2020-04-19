# Fill in this file
from src.agent import SimpleAgent
from src.bandit import NBandits
from src.experiment import Experiment, Trial


if __name__ == "__main__":

    # Exercise 2.4

    def run_iteration_trial_one(nothing):
        bandit = NBandits(10)
        agent = SimpleAgent(0.1, 0, bandit, method="means")
        for i in range(5000):
            if agent.total_plays % 1 == 0:
                agent.bandit.alternating_dependent_random_walk()
            agent.play()
        return agent.total_rewards


    def run_iteration_trial_two(nothing):
        bandit = NBandits(10)
        agent = SimpleAgent(0.1, 0, bandit, method="constant", constant=0.1)
        for i in range(5000):
            if agent.total_plays % 1 == 0:
                agent.bandit.alternating_dependent_random_walk()
            agent.play()
        return agent.total_rewards

    trial_one = Trial(run_iteration_trial_one, None, "sample update method")
    trial_two = Trial(run_iteration_trial_two, None, "constant update method")

    experiment = Experiment([trial_one, trial_two], title="Exercise 2.4")
    results = experiment.run_parallel(100)

    el_log_one = results[0]
    el_log_two = results[1]
    samp_avg_wins = 0
    const_wins = 0
    ties = 0
    for i in range(1000):
        if el_log_one.log[i] > el_log_two.log[i]:
            samp_avg_wins += 1
        elif el_log_one.log[i] < el_log_two.log[i]:
            const_wins += 1
        else:
            ties += 1

    print("sample wins: {} const wins: {} ties: {}".format(samp_avg_wins, const_wins, ties))

    experiment.produce_plot(y_label="total rewards per 1000 iterations")



