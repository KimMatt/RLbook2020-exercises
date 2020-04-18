import pandas as pd 
import numpy as np 
import multiprocessing as m
from collections import defaultdict
from src.nArmBandit import Bandit


# Iterate trials and updates in Pool
def run_in_pool(args):

    # Args
    trials = args['trials']
    iterations = args['iterations']
    e = args['epsilon']
    arms = args['arms']

    # Iterate Trials
    trial_avgs, trial_totals = [0]*trials, [0]*trials
    for j in range(trials):

        # Build bandit
        bandit = Bandit(arms, e)
        rewards = [np.random.normal() for i in range(arms)]
        bandit.set_rewards(rewards)
        
        # Store scores
        avg_list, total_list = [0] * iterations, [0] * iterations

        # Iterate bandit updates
        for i in range(iterations):
            # run update
            bandit.update()
            avg = bandit.get_average()
            total = np.sum(bandit.total_rewards)

            # update score lists
            avg_list[i] = avg
            total_list[i] = total

        # Trial lists
        trial_avgs[j] = avg_list 
        trial_totals[j] = total_list

    # Calculate Scores
    trial_avgs = [np.sum(a) / 10 for a in zip(*trial_avgs)]
    trial_totals = [np.sum(t) / 10 for t in zip(*trial_totals)]

    return trial_avgs, trial_totals


# Run Parallel Trials
def get_scores_parallel(trials=10, iterations=2000, 
                        arms=10, epsilon = [0, 0.1, 0.01]):
    # Set Up
    threads = len(epsilon)
    print('Running', threads, 'Bandits')


    # Build Pool list of dicts
    pool_args = [{'trials': trials, 'iterations': iterations, 
                         'arms': arms, 'epsilon': e} 
                            for e in epsilon]

    # Run in pool
    averages, totals = defaultdict(float), defaultdict(float)
    p = m.Pool(threads)
    results = p.map(run_in_pool, pool_args)

    # Scoring dictionaries
    for score, e in zip(results, epsilon):
        averages['Avg: ' + str(e)] = score[0]
        totals['Totals: ' + str(e)] = score[1]

    # Scoring DataFrames
    df_avg = pd.DataFrame(averages)
    df_totals = pd.DataFrame(totals)
   
    return df_avg, df_totals


# Run
def run():
    # Run trials
    df_avgs, df_totals = get_scores_parallel(iterations=10000, epsilon=[0, 0.1, 0.01])

    # Plot Averages
    df_avgs.plot()
    plt.title('Average Reward for 10, 10-Arm Bandits', fontsize=18)
    plt.xlabel('Average of Average Rewards')
    plt.ylabel('Iteration')
    plt.savefig('plots/nArmBandit_AverageReward.png')
    plt.show()

    # Plot Totals
    df_totals.plot()
    plt.title('Total Reward for 10, 10-Arm Bandits', fontsize=18)
    plt.xlabel('Total of Average Rewards')
    plt.ylabel('Iteration')
    plt.savefig('plots/nArmBandit_TotalReward.png')
    plt.show()