from src.logger import Logger
from src.tictactoe import TicTacToe
from src.agent import Agent, ExpertPlayer
from src.experiment import (train, test, run_in_pool,
                            run_parallel, ensemble_agents,
                            compare_agents, build_meta_agent)


# Demonstration of meta agent improvement and comparison of agent types
if __name__ == "__main__":

    # Build meta agent ----------
    meta_agent, meta_scores = build_meta_agent(get_scores=True)

    # Comparison -----------
    # - builds players
    a1 = Agent(1, 0.5)
    a2 = Agent(1, 0.5, symmetric_aware = True)
    a3 = Agent(1, 0.5)

    # - build self-play opponent
    a4 = Agent(2, 0.5)
    a4.set_policy(a3.get_policy())

    # - build agent lists
    player_list = [a1, a2, a3]
    opponent_list = [Agent(2, 0.5, dummy=True), Agent(2, 0.5, dummy=True), a4]

    # run comparisons
    print(' ')
    print('Running Comparisons...')
    print(' ')
    comparison_agent_list = compare_agents(steps = 1, train_iterations = 10000,
                                           player_list = player_list,
                                           opponent_list = opponent_list,
                                           get_scores = False)

    # Outputs -------
    # - Meta Agent
    print('META AGENT TEST ---- ')
    test(10000, meta_agent, Agent(2, 0.5, dummy=True))
    print(' ')

    # - Comparison Agents
    scores = run_parallel(player_list = comparison_agent_list,
                          train_or_test = 'test',
                          print=False)
    print('COMPARISON TESTS SCORES vs Random Agents (50K iterations) ----- ')
    print('Control Agent:', scores[0])
    print('Symmetrically Aware Agent:', scores[1])
    print('Self-Play Agent:', scores[2])
