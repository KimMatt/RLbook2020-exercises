import numpy as np
import math
# TODO: Update game to have loss
# GOAL: All wins/ties against dummy

class InvalidMoveException(Exception):

    def __init__(self):
        pass


class Logger:

    agent_1_wins = 0
    agent_2_wins = 0
    ties = 0


    def __init__(self):
        pass


    def log_agent_win(self, player_n):
        if player_n == 1:
            self.agent_1_wins += 1
        else:
            self.agent_2_wins += 1


    def log_tie(self):
        self.ties += 1


class TicTacToe:

    # 0 for blank, 1 for x, 2 for o
    in_progress = False
    game_state = None
    logger = None
    winner = None


    def __init__(self, logger):
        self.logger = logger
        self.in_progress = True
        self.game_state = [0 for i in range(0, 9)]


    @staticmethod
    def winning_state(game_state, player_sign, space):
        # Check if the space + player sign is a winning move in given state
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


    def play_move(self, player_n, space):
        # Plays the given move by player in given space
        # Returns whether the move is a winning move or not.
        if self.game_state[space] == 0:
            self.game_state[space] = player_n
            if TicTacToe.winning_state(self.game_state, player_n, space):
                self.in_progress = False
                self.logger.log_agent_win(player_n)
                self.winner = player_n
                return True
            return False
        else:
            raise InvalidMoveException


    def mark_tie(self):
        # Mark a tie in the game.
        self.logger.log_tie()
        self.in_progress = False


class Agent:

    player_n = None
    game = None

    exploration = None
    exploitation = None

    policy = {}
    game_logs = []


    def __init__(self, player_n, exploration):
        self.exploration = exploration
        self.exploitation = 1.0 - exploration
        self.player_n = player_n


    def set_exploration(self, exploration):
        self.exploration = exploration
        self.exploitation = 1.0 - exploration


    def enter(self, game):
        self.game = game


    def state_p(self, game_state, space):
        # return the probability of the game state to win according to policy
        if self.policy.get(str(game_state)) is None:
            if TicTacToe.winning_state(game_state, self.player_n, space):
                self.policy[str(game_state)] = 1.0
                return 1.0
            self.policy[str(game_state)] = 0.5
            return 0.5
        else:
            return self.policy.get(str(game_state))


    def play(self):
        # state is an array representing the current array mode
        # Marker is either 1 or 2
        game_state = self.game.game_state[:]
        # To contain tuples of (probability, move location, game state)
        possible_moves = []
        # Can store this as a list in Game to save computation time
        for i in range(0, len(game_state)):
            if game_state[i] == 0:
                # we have a possible move
                possible_game_state = game_state[:]
                possible_game_state[i] = self.player_n
                possible_moves.append((self.state_p(possible_game_state, i), i, possible_game_state))
        possible_moves.sort()

        if not possible_moves:
            self.game.mark_tie()
        else:
            # With a single geometric sample with the probability of exploration
            # see if we land "heads" on exploration
            explore = np.random.geometric(p=self.exploration) == 1

            if explore and len(possible_moves) > 1:
                # try a new approach
                p_less_moves = possible_moves[:]
                max_p = possible_moves[-1][0]
                range_to_max = 0
                for i in range(0,len(p_less_moves)):
                    if possible_moves[i][0] == max_p:
                        break
                    range_to_max += 1
                if range_to_max == 0:
                    self.game_logs.append(possible_moves[0])
                    self.game.play_move(self.player_n, possible_moves[0][1])
                else:
                    exploration_target = np.random.randint(0, range_to_max)
                    self.game_logs.append(possible_moves[exploration_target])
                    self.game.play_move(self.player_n, possible_moves[exploration_target][1])
            else:
                # exploitation (also includes 1 choice case)
                # try the approach with highest probability
                self.game_logs.append(possible_moves[-1])
                self.game.play_move(self.player_n, possible_moves[-1][1])


    def update_policies(self, alpha):
        for i in range(len(self.game_logs) -2, -1, -1):
        # update policies
        # state probability = state probability + alpha(state probs next - current)
            self.policy[str(self.game_logs[i][2])] = self.state_p(self.game_logs[i][2],
                self.game_logs[i][1]) + alpha * (self.state_p(self.game_logs[i+1][2],
                self.game_logs[i+1][1]) - self.state_p(self.game_logs[i][2],
                self.game_logs[i][1]))
        self.game_logs = []


if __name__ == "__main__":


    def intelligent_play(agent1, agent2):
        iterations = 50000
        games_count = iterations
        # Hyperparameters
        alpha = 0.5
        decrease_factor = 0.99
        logger = Logger()

        while iterations > 0:
            game = TicTacToe(logger)
            agent1.enter(game)
            agent2.enter(game)
            while game.in_progress:
                if iterations % 2 == 0:
                    # play game
                    agent1.play()
                    if game.in_progress:
                        agent2.play()
                else:
                    agent2.play()
                    if game.in_progress:
                        agent1.play()
            # update policies
            agent1.update_policies(alpha)
            agent2.update_policies(alpha)
            # Decrease alpha over time
            if iterations % (math.ceil(iterations/50)) == 0:
                alpha = alpha * decrease_factor
            iterations -= 1
        print("TRAIN VS. LEARNING AGENT: agent 1 wins: {}, agent 2 wins: {}, ties: {}".format(
            (logger.agent_1_wins / games_count), (logger.agent_2_wins / games_count),
            (logger.ties / games_count)))

        if logger.agent_1_wins > logger.agent_2_wins:
            return agent1
        return agent2


    def train_vs_dummy(agent1, agent2):
        iterations = 50000
        games_count = iterations
        # Hyperparameters
        alpha = 0.5
        decrease_factor = 0.99
        logger = Logger()

        while iterations > 0:
            game = TicTacToe(logger)
            agent1.enter(game)
            agent2.enter(game)
            while game.in_progress:
                if iterations % 2 == 0:
                    # play game
                    agent1.play()
                    if game.in_progress:
                        agent2.play()
                else:
                    agent2.play()
                    if game.in_progress:
                        agent1.play()
            # update policies
            agent1.update_policies(alpha)
            # Decrease alpha over time
            if iterations % (math.ceil(iterations/50)) == 0:
                alpha = alpha * decrease_factor
            iterations -= 1
        print("TRAIN VS DUMMY: agent 1 wins: {}, agent 2 wins: {}, ties: {}".format(
            (logger.agent_1_wins / games_count), (logger.agent_2_wins / games_count),
            (logger.ties / games_count)))

        return agent1


    def dummy_tournament(agent1, agent2):
        iterations = 5000
        games_count = iterations
        logger = Logger()

        while iterations > 0:
            game = TicTacToe(logger)
            agent1.enter(game)
            agent2.enter(game)
            while game.in_progress:
                if iterations % 2 == 0:
                    # play game
                    agent1.play()
                    if game.in_progress:
                        agent2.play()
                else:
                    agent2.play()
                    if game.in_progress:
                        agent1.play()
            iterations -= 1
        print("VS. DUMMY: agent 1 wins: {}, agent 2 wins: {}, ties: {}".format( (
            logger.agent_1_wins / games_count ), ( logger.agent_2_wins / games_count ),
            ( logger.ties / games_count )))

    AGENT_1 = Agent(1, 0)
    AGENT_2 = Agent(2, 0.8)
    train_vs_dummy(AGENT_1, AGENT_2)
    AGENT_1.set_exploration(0)
    AGENT_3 = Agent(2, 0.5)
    dummy_tournament(AGENT_1, AGENT_3)

    AGENT_1 = Agent(1, 0.8)
    AGENT_2 = Agent(2, 0.8)
    WINNER = intelligent_play(AGENT_1, AGENT_2)
    WINNER.player_n = 1
    WINNER.set_exploration(0)
    AGENT_3 = Agent(2, 0.5)
    dummy_tournament(WINNER, AGENT_3)