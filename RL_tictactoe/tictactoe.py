
import numpy as np

""""winning_policies = [[1,1,1,0,0,0,0,0,0], [0,0,0,1,1,1,0,0,0],
[0,0,0,0,0,0,1,1,1], [1,0,0,1,0,0,1,0,0], [0,1,0,0,1,0,0,1,0],
[0,0,1,0,0,1,0,0,1], [1,0,0,0,1,0,0,0,1], [0,0,1,0,1,0,1,0,0]]
""""
class InvalidMoveException(Exception):

    def __init__(self):
        pass


class TicTacToe:

    # 0 for blank, 1 for x, 2 for o
    in_progress = False
    game_state = None


    def __init__(self):
        self.in_progress = True
        self.game_state = [0 for i in range(0, 9)]


    @staticmethod
    def winning_state(game_state, player_sign):

        columns = [[1, 4, 7], [0, 3, 6], [2, 5, 8]]
        rows = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        for row in rows:
            if space in row:
                row.remove(space)
                if all([game_state[each] == player_sign for each in row]):
                    return True
        for column in columns:
            if space in column:
                column.remove(space)
                if all([game_state[each] == player_sign for each in column]):
                    return True
        return False


    def _winning_move(self, space):
        # Check if it is a winning move
        # calculate vertical matches
        # calculate horizontal matches
        player_sign = self.game_state[space]
        return TicTacToe.winning_state(self.game_state, player_sign)


    def play_move(self, player_n, space):
        # Plays the given move by player in given space
        # Returns whether the move is a winning move or not.
        if self.game_state[space] == 0:
            self.game_state[space] = player_n
            if self._winning_move(space):
                self.in_progress = False
                return True
            return False
        else:
            raise InvalidMoveException


class Agent:

    player_n = None
    game = None

    exploration = 0.2
    exploitation = 1.0 - exploration

    policy = {}
    game_logs = []

    def __init__(self, player_n, game):
        self.player_n = player_n
        self.game = game


    def state_p(self, game_state):
        # return the probability of the game state to win according to policy
        if policy.get(game_state) is None:
            if TicTacToe.winning_state(game_state, self.player_n):
                return 1.0
            return 0.5
        else:
            return policy.get(game_state)


    def play(self):
        # state is an array representing the current array mode
        # Marker is either 1 or 2
        # With a single geometric sample with the probability of exploration
        # see if we land "heads" on exploration
        game_state = self.game.game_state[:]
        possible_moves = []
        for i in range(0,len(game_state)):
            if game_state[i] == 0:
                # we have a possible move
                possible_game_state = game_state[:]
                possible_game_state[i] = self.player_n
                possible_moves.append((self.state_p(possible_game_state), i))
        possible_moves.sort()
        explore = np.random.geometric(p=self.exploration) == 1

        if explore and len(possible_moves) > 1:
            # try a new approach
            p_less_moves = possible_moves[:]
            max_p = possible_moves[-1][0]
            range_to_max = 0
            for i in range(0,len(p_less_moves)):
                if possible_moves[i][0] == range_to_max:
                    break
                range_to_max += 1
            if range_to_max == 0:
                self.game_logs.append(possible_moves[0][0])
                self.game.play_move(self.player_n, possible_moves[0][1])
            else:
                exploration_target = np.random.randint(0, range_to_max)
                self.game_logs.append(possible_moves[exploration_target][1])
                self.game.play_move(self.player_n, possible_moves[exploration_target][1])
        else:
            # exploitation, try the approach with highest probability
            move_space = possible_moves[-1][1]
            self.game_logs.append(possible_moves[-1][1])
            self.game.play_move(self.player_n, move_space)


    def update_policies(self, alpha):
        for i in range(len(self.policy_logs) -2, -1, -1):
        # update policies
        # state probability = state probability + alpha(state probs next - current)
            self.policy_logs[i-1] = self.policy_logs[i-1] + \
            alpha * (self.policy_logs[i] - self.policy_logs[i-1])


if __name__ == "__main__":

    def self_play(agent1, agent2):

        iterations = 100
        # Hyperparameters
        alpha = 0.2

        for i in range(0,iterations):
            game = TicTacToe()
            while game.in_progress:
                # play game
                agent1.play()
                agent2.play()
            # update policies
            agent1.update_policies(alpha)
            agent2.update_policies(alpha)

    TICTACTOE = TicTacToe()
    AGENT_1 = Agent(1, TICTACTOE)
    AGENT_2 = Agent(2, TICTACTOE)
    self_play(AGENT_1, AGENT_2)

