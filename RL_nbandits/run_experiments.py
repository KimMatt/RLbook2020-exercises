# Fill in this file
import numpy as np 
from src.agent import SimpleAgent
from src.bandit import NBandits
from src.nArmBandit import Bandit
from src.experiment import Experiment, Trial



if __name__ == "__main__":

    # Exercise 2.1 ----------------------------------------

    # trial function
    def run_bandit(args):
        bandit_list = args['bandits']
        rewards = [] 
        for bandit in bandit_list:
            bandit.update()
            rewards.append(np.sum(bandit.total_rewards))

        reward = np.mean(rewards)
        return reward

    # - bandit params
    n_arms = 10
    e0, e01, e001 = [0]*10, [0.1]*10, [0.01]*10
    epsilons = [e0, e01, e001]
    rewards_control = [np.random.normal() for i in range(n_arms)]
    bandits = [[Bandit(n_arms, e, rewards=rewards_control) for e in ep] for ep in epsilons]

   
    # Build Trials
    trial_args = [{'bandits': bandit_list} for bandit_list in bandits]
    trials = [Trial(run_bandit, trial_arg, 'Trials: ' + str(e[0])) 
                for trial_arg, e in zip(trial_args, epsilons)]
    print(trials)
    # Experiment
    experiment = Experiment(trials, 'Exercise 2.1')
    experiment.run_parallel(2000)
    experiment.produce_plot(show=True)


    # Exercise 2.4 ----------------------------------------

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



