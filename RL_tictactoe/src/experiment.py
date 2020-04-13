import math
import numpy as np
import copy

from .logger import Logger
from .tictactoe import TicTacToe

def backprop_agents(game_winner, a1, a2, alpha):
    # set reward values based on game outcome
    if game_winner is None:
        a1_reward = 0.5
        a2_reward = 0.5
    else:
        a1_reward = 1.0 if game_winner == 1 else 0.0
        a2_reward = 1.0 if game_winner == 2 else 0.0
    # update policies
    a1.back_propagate_policies(alpha, a1_reward)
    a2.back_propagate_policies(alpha, a2_reward)


def run_games(iterations, a1, a2, logger, **kwargs):
    """
    Args:
        a1 ([Agent]): agent 1 object to play
        a2 ([Agent]): agent 2 object to be played against
        iterations ([int]): number of games to run
        logger ([Logger]): logger object to keep track of wins/losses/ties
        kwargs:
            training ([Boolean]): whether or not this is a training run
                if true then will backpropagate
            alpha ([Float]): hyperparameter for update values
            decrease_factor ([Float]): hyperparameter for alpha decrease over time
            decrease_rate ([int]): every decrease_rate games the alpha will decrease by 1 - decrease_factor
    """
    if kwargs.get('training'):
        training = kwargs.get('training')
        alpha = kwargs.get('alpha') if kwargs.get(
            'alpha') is not None else 0.2
        decrease_factor = kwargs.get('decrease_factor') if kwargs.get(
            'decrease_factor') is not None else 0.9
        decrease_rate = kwargs.get('decrease_rate') if kwargs.get(
            'decrease_rate') is not None else 50
    else:
        training = False

    games_left = iterations
    while games_left > 0:
        game = TicTacToe(logger)
        a1.enter(game)
        a2.enter(game)
        a1_turn = True if games_left % 2 == 0 else False
        while game.in_progress:
            if a1_turn:
                a1.play()
            else:
                a2.play()
            a1_turn = not a1_turn
        if training:
            backprop_agents(game.winner, a1, a2, alpha)
            # Decrease alpha over time
            if games_left % (math.ceil(games_left/decrease_rate)) == 0:
                alpha = alpha * decrease_factor
        games_left -= 1


def train(iterations, a1, a2, random=False, return_both=False):

    if random:
        np.random.seed()

    # Hyperparameters
    alpha = 0.2
    decrease_factor = 0.9
    decrease_rate = 50
    logger = Logger()
    run_games(iterations, a1, a2, logger, training=True,
              alpha=alpha, decrease_factor=decrease_factor, decrease_rate=decrease_rate)
    if return_both:
        print("TRAIN VS. DUMMY: agent 1 wins: {}, agent 2 wins: {}, ties: {}".format(
            (logger.agent_1_wins / iterations), (logger.agent_2_wins / iterations),
            (logger.ties / iterations)))
        return [a1, a2]

    if a2.dummy:
        print("TRAIN VS. DUMMY: agent 1 wins: {}, agent 2 wins: {}, ties: {}".format(
            (logger.agent_1_wins / iterations), (logger.agent_2_wins / iterations),
            (logger.ties / iterations)))
    else:
        print("TRAIN VS. LEARNING AGENT: agent 1 wins: {}, agent 2 wins: {}, ties: {}".format(
            (logger.agent_1_wins / iterations), (logger.agent_2_wins / iterations),
            (logger.ties / iterations)))
    # returns the better agent
    # if logger.agent_1_wins > logger.agent_2_wins:
    #     return a1
    # return a2
    return a1

def test(iterations, a1, a2, random=False, return_both=False):

    # get exploration value
    a1_exploration = a1.exploration 
    a1.set_exploration(0)

    if random:
        np.random.seed()
        
    logger = Logger()
    run_games(iterations, a1, a2, logger)

    wins = logger.agent_1_wins / iterations
    losses = logger.agent_2_wins / iterations
    ties = logger.ties / iterations

    # set exploration back
    a1.set_exploration(a1_exploration)

    if return_both:
        print("TEST VS. DUMMY: agent 1 wins: {}, agent 2 wins: {}, ties: {}".format(
            (wins), (losses), (ties)))
        return [a1, a2]

    if a2.dummy:
        print("TEST VS. DUMMY: agent 1 wins: {}, agent 2 wins: {}, ties: {}".format(
            (wins), (losses), (ties))) 
    else:
        print("TEST VS. LEARNING AGENT: agent 1 wins: {}, agent 2 wins: {}, ties: {}".format(
            (wins), (losses), (ties)))

    return {'wins': wins, 'losses':losses, 'ties':ties}