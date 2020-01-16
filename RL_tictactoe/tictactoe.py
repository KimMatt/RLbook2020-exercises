
winning_policies = [[1,1,1,0,0,0,0,0,0], [0,0,0,1,1,1,0,0,0],
[0,0,0,0,0,0,1,1,1], [1,0,0,1,0,0,1,0,0], [0,1,0,0,1,0,0,1,0],
[0,0,1,0,0,1,0,0,1], [1,0,0,0,1,0,0,0,1], [0,0,1,0,1,0,1,0,0]]

class InvalidMoveException(Exception):

    def __init__():
        pass


class TicTacToe:

    # 0 for blank, 1 for x, 2 for o
    in_progress = False
    game_state = [0 for i in range (0,9)]


    def __init__():
        in_progress = True
        pass


    def _winning_move(space):
        # Check if it is a winning move
        # calculate vertical matches
        # calculate horizontal matches
        player_sign = game_state[space]
        columns = [(1,4,7), (0,3,6), (2,5,8)]
        rows = [(0,1,2), (3,4,5), (6,7,8)]
        for row in rows:
            if space in row:
                row.remove(space)
                if all([game_state[each] == player_sign for each in row]):
                    return True
        for column in columns:
            if space in column:
                column.remove(space)
                if all(game_state[each] == player_sign for each in column):
                    return True


    def play_move(player_n, space):
        # Plays the given move by player in given space
        # Returns whether the move is a winning move or not.
        if game_state[space] == 0:
            game_state[space] = player_n
            if _winning_move():
                in_progress = False
                return True
            else:
                return False
        else:
            raise InvalidMoveException


class Agent:

    player_n = None
    game = None

    exploration = 0.2
    exploitation = 0.8

    policy = {}
    policy_logs = []

    def __init__(player_n, game):
        self.player_n = player_n
        self.game = game

    def play(self):
        # state is an array representing the current array mode
        # Marker is either 1 or 2


    def update_policies(self, alpha):
        for i in range(len(self.policy_logs) -2, -1, -1):
        # update policies
        # state probability = state probability + alpha(state probs next - current)
            policy_logs[i-1] = policy_logs[i-1] + alpha * (policy_logs[i] - policy_logs[i-1])




if __name__ == "__main__":


    def update_policies(agent):
         for policy in agent1.policy_logs
            # update policies
            # state probability = state probability + alpha(state probs next - current)


    def self_play(agent1, agent2):

        iterations = 100
        # Hyperparameters
        alpha = 0.2

        for i in range(0,iterations)
            game = TicTacToe()
            while game.in_progress:
                # play game
                agent1.play()
                agent2.play()
            # update policies
            agent1.update_policies(alpha)
            agent2.update_policies(alpha)

    game = TacTacToe()
    agent1 = Agent(1, game)
    agent2 = Agent(2, game)
    self_play(agent1, agent2)

