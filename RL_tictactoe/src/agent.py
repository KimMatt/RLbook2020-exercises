import numpy as np
import random
import math

from .logger import Logger
from .tictactoe import TicTacToe


class Agent:

    player_n = None
    game = None
    # Whether or not this agent treats symmetrical game states identically.
    symmetric_aware = None
    mirror = None
    exploration = None


    def __init__(self, player_n, exploration, symmetric_aware=False, dummy=False):
        """ Initialize the Agent object with given settings
        Args:
            player_n ([int]): The player #, 1 or 2
            exploration ([float]): Percentage chance to explore non optimal options randomly
            symmetric_aware (bool, optional): Set to true if you want policy updates to update symmetrical
                states as well. Defaults to False.
            dummy (bool, optional): Set to true if you don't want this agent to learn. Defaults to False.
        """
        self.exploration = exploration
        self.player_n = player_n
        self.symmetric_aware = symmetric_aware
        self.game_logs = []
        self.policy = {}
        self.dummy = dummy


    def normalized_game_state(self, game_state):
        """Normalizes the game state, making all elements /in [-1,0,1] representation where
        -1 is the opposing player and 1 is the current player.

        Args:
            game_state ([array]): representation of the tic tac toe board
            player_n ([int]): the player id (1 or 2)

        Returns:
            [array]: normalized game state
        """
        normalized_game_state = game_state[:]
        for element, i in enumerate(game_state):
            if element == self.player_n:
                normalized_game_state[i] = 1
            elif element != 0:
                normalized_game_state[i] = -1
        return normalized_game_state


    def set_policy(self, policy):
        # In order to allow for shared policies between agents.
        self.policy = policy


    def update_policy(self, game_state, value):
        # Normalize the game state
        game_state = self.normalized_game_state(game_state)
        self.policy[str(game_state)] = value

        if self.symmetric_aware:
            transform_maps = [{0: 2, 1: 1, 2: 0, 3: 5, 4: 4, 5: 3, 6: 8, 7: 7, 8: 6},  # x mirror
                              {0: 6, 1: 7, 2: 8, 3: 3, 4: 4, 5: 5,
                                  6: 0, 7: 1, 8: 2},  # y mirror
                              {0: 8, 1: 5, 2: 2, 3: 7, 4: 4, 5: 1,
                                  6: 6, 7: 3, 8: 0},  # xy mirror
                              {0: 0, 1: 3, 2: 6, 3: 1, 4: 4, 5: 7,
                               6: 2, 7: 5, 8: 8},  # yx mirror
                              {0: 2, 1: 5, 2: 8, 3: 1, 4: 4, 5: 7,
                               6: 0, 7: 3, 8: 6},  # rotate 90
                              {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3,
                               6: 2, 7: 1, 8: 0},  # rotate 180
                              {0: 6, 1: 3, 2: 0, 3: 7, 4: 4, 5: 1, 6: 8, 7: 5, 8: 2}]  # rotate 270

            for transform_map in transform_maps:
                transformed_game_state = [0 for i in range(0, 9)]
                for i, element in enumerate(game_state):
                    transformed_game_state[transform_map[i]] = element
                self.policy[str(transformed_game_state)] = value


    def set_exploration(self, exploration):
        self.exploration = exploration


    def enter(self, game):
        self.game = game


    def state_p(self, game_state):
        # return the probability of the game state to win according to policy
        state_p_value = self.policy.get(
            str(self.normalized_game_state(game_state)))
        if state_p_value is None:
            self.update_policy(game_state[:], 0.5)
            return 0.5
        return state_p_value


    def play(self):
        # Compute the likelihoods for each move
        possible_move_infos = []
        for move in self.game.possible_moves:
            possible_game_state = self.game.game_state[:]
            possible_game_state[move] = self.player_n
            # possible_move_infos = [(percentage_chance_to_win, move, game_state_as_a_result)]
            possible_move_infos.append(
                (self.state_p(possible_game_state), move, possible_game_state))

        # Based on our exploration value, see if we explore or exploit
        explore = np.random.uniform(0, 1) <= self.exploration
        possible_move_infos.sort()

        # Choose a move
        target_move = None

        # Calculate how many non optimal moves we have
        optimal_p = possible_move_infos[-1][0]
        non_optimal_move_counts = 0
        for i, _ in enumerate(possible_move_infos):
            if possible_move_infos[i][0] == optimal_p:
                break
            non_optimal_move_counts += 1

        # If we have 0 non optimal moves, then just choose a random out of the optimals
        if non_optimal_move_counts == 0:
            target_move = np.random.randint(0, len(possible_move_infos))
        # If explore choose one of the non optimals
        elif explore:
            target_move = np.random.randint(0, non_optimal_move_counts)
        # If exploit choose one of the optimals
        else:
            target_move = np.random.randint(
                non_optimal_move_counts, len(possible_move_infos))
        self.game.play_move(self.player_n, possible_move_infos[target_move][1])
        self.game_logs.append(possible_move_infos[target_move])


    def back_propagate_policies(self, alpha, reward):
        if not self.dummy:
            # update final move policy value
            self.update_policy(
                self.game_logs[-1][2][:], reward)
            # back propagate policy values
            for i in range(len(self.game_logs) - 2, 0, -1):
                # state probability = state probability + alpha(state probs next - current)
                current_p = self.state_p(self.game_logs[i][2])
                next_p = self.state_p(self.game_logs[i+1][2])
                update_p = current_p + (alpha * (next_p - current_p))
                self.update_policy(self.game_logs[i][2][:], update_p)
            # Clear game logs in case we want to play another game with this agent
            self.game_logs = []
