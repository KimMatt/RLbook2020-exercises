import numpy as np


class InvalidMoveException(Exception):

    def __init__(self):
        pass


class TicTacToe:

    in_progress = False
    game_state = None
    logger = None
    winner = None
    possible_moves = None


    def __init__(self, logger):
        self.logger = logger
        self.in_progress = True
        # 0 for blank, 1 for x, 2 for o
        self.game_state = [0 for i in range(0, 9)]
        self.possible_moves = [0,1,2,3,4,5,6,7,8]


    @staticmethod
    def winning_state(game_state, player_sign, space):
        # Check if the space + player sign is a winning move in given state
        rows = [[1, 4, 7], [0, 3, 6], [2, 5, 8], # verticals
                [0, 1, 2], [3, 4, 5], [6, 7, 8], # horizontals
                [0, 4, 8], [2, 4, 6]] # diagonals
        for row in rows:
            if space in row:
                if all([game_state[each] == player_sign for each in row]):
                    return True
        return False


    def play_move(self, player_n, space):
        # Plays the given move by player in given space
        # Returns whether the move is a winning move or not.
        if self.game_state[space] == 0:
            self.possible_moves.remove(space)
            self.game_state[space] = player_n
            if TicTacToe.winning_state(self.game_state, player_n, space):
                self.in_progress = False
                self.logger.log_agent_win(player_n)
                self.winner = player_n
                return True
            elif not self.possible_moves:
                self.in_progress = False
                self.logger.log_tie()
            return False
        else:
            raise InvalidMoveException
