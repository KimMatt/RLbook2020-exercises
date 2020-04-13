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


    def __init__(self, player_n, exploration=None, symmetric_aware=False, dummy=False):
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
        for i, element in enumerate(game_state):
            if element == self.player_n:
                normalized_game_state[i] = 1
            elif element != 0:
                normalized_game_state[i] = -1
        return normalized_game_state



    # Get the policy map of the agent
    def get_policy(self):
        return self.policy


    def set_policy(self, policy):
        # In order to allow for shared policies between agents.
        self.policy = policy


    def get_last_move(self):
        if self.game_logs:
            return self.game_logs[-1][1]


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


    def get_possible_move_infos(self):
        possible_move_infos = []
        for move in self.game.possible_moves:
            possible_game_state = self.game.game_state[:]
            possible_game_state[move] = self.player_n
            # possible_move_infos = [(percentage_chance_to_win, move, game_state_as_a_result)]
            possible_move_infos.append(
                (self.state_p(possible_game_state), move, possible_game_state))
        return possible_move_infos


    def play(self):
        # Compute the likelihoods for each move
        possible_move_infos = self.get_possible_move_infos()
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


class ExpertPlayer(Agent):

    def __init__(self, player_n):
        super().__init__(player_n)
        self.dummy = True


    def winning_moves(self, player_n):
        """Returns list of moves [int] that would result
        in player_n winning if they took that position.
        Args:
            player_n (int): Player to check for
        Returns:
            moves (list): List of winning moves
        """

        possible_moves = self.game.possible_moves

        moves = []

        for move in possible_moves:
            next_state = self.game.game_state[:]
            next_state[move] = player_n
            if self.game.winning_state(next_state, player_n, move):
                moves.append(move)

        return moves


    def fork_moves(self, player_n):
        """Returns list of moves [int] where role has
        two opportunities to win (two non-blocked lines of 2) if
        they took that position.
        Args:
            player_n (int): Player to check for
        Returns:
            moves (list): List of fork moves
        """
        moves = []
        possible_moves = self.game.possible_moves[:]
        # Note: This is used to test different positions so it may not be role's
        # actual turn so role-checking is turned off
        for move in possible_moves:
            next_state = self.game.game_state[:]
            next_state[move] = player_n
            remaining_moves = possible_moves[:]
            remaining_moves.remove(move)
            winning_count = 0
            for move_2 in remaining_moves:
                test_state = next_state[:]
                test_state[move_2] = player_n
                if self.game.winning_state(test_state, player_n, move_2):
                    winning_count += 1
            if winning_count >= 2:
                moves.append(move)
        return moves


    def opposite_corners(self, player_n, opponent_n):
        """Returns list of moves [int] opposite to an opponent's corner
        Args:
            player_n (int): Player to check for
        Returns:
            moves (list): List of opposite corner moves
        """
        moves = []
        opposite_corners = {0: 8, 2: 6, 8: 0, 6: 2}
        for k, v in opposite_corners.items():
            if self.game.game_state[k] == opponent_n and self.game.game_state[v] == 0:
                moves.append(v)
        return moves


    def play(self):
        available_moves = self.game.possible_moves[:]

        def get_move(self):

            corners = [0,2,6,8]
            center = 4

            opponent_n = 1 if self.player_n == 2 else 2

            winning_positions = self.winning_moves(self.player_n)
            blocking_positions = self.winning_moves(opponent_n)
            fork_positions = self.fork_moves(self.player_n)
            opponent_forks = self.fork_moves(opponent_n)
            opposite_corners = self.opposite_corners(self.player_n, opponent_n)
            available_corners = list(set(corners).intersection(set(available_moves)))

            if self.game.moves == 0:
                # 1. If first move of the game, play a corner or center
                corners_and_center = corners + [center]
                return corners_and_center[random.randint(0, 4)]
            if winning_positions:
                # 2. Check for winning moves
                return winning_positions[random.randint(0,len(winning_positions)-1)]
            if blocking_positions:
                # 3. Check for blocking moves
                return blocking_positions[random.randint(0,len(blocking_positions)-1)]
            if fork_positions:
                # 4. Check for fork positions
                return fork_positions[random.randint(0, len(fork_positions)-1)]
            if opponent_forks:
                # 5. Prevent opponent from using a fork position
                return opponent_forks[random.randint(0, len(opponent_forks)-1)]
            if center in available_moves:
                # 6. Try to play center
                return center
            if opposite_corners:
                # 7. Try to play a corner opposite to opponent
                return opposite_corners[random.randint(0, len(opposite_corners)-1)]
            if available_corners:
                # 8. Try to play any corner
                return available_corners[random.randint(0, len(available_corners)-1)]
            # 9. Play anywhere else - i.e. a middle position on a side
            return available_moves[random.randint(0,len(available_moves)-1)]

        # play the move
        move_location = get_move(self)
        self.game.play_move(self.player_n, move_location)

