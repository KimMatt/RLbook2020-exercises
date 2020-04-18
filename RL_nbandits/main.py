import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from experiments.nArmBandit import get_scores_parallel, run_in_pool
from src.nArmBandit import Bandit
import multiprocessing as m 

if __name__=='__main__':
    
    
    
    df_avgs, df_totals = get_scores_parallel(iterations=10000, epsilon=[0, 0.1, 0.01])

    df_avgs.plot()
    plt.title('Average Reward for 10, 10-Arm Bandits', fontsize=18)
    plt.xlabel('Average of Average Rewards')
    plt.ylabel('Iteration')
    plt.savefig('plots/nArmBandit_AverageReward.png')
    plt.show()

    df_totals.plot()
    plt.title('Total Reward for 10, 10-Arm Bandits', fontsize=18)
    plt.xlabel('Total of Average Rewards')
    plt.ylabel('Iteration')
    plt.savefig('plots/nArmBandit_TotalReward.png')
    plt.show()




