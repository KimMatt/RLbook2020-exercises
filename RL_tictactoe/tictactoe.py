import numpy as np
import math
# TODO: Batching
# TODO: Have policies stored independent of player #
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
    possible_moves = None


    def __init__(self, logger):
        self.logger = logger
        self.in_progress = True
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


class Agent:

    player_n = None
    game = None
    # Whether or not this agent treats symmetrical game states identically.
    symmetric_aware = None
    mirror = None
    exploration = None


    @staticmethod
    def mirrored_game_state(game_state):
        mirrored_game_state = game_state[:]
        for element, i in enumerate(game_state):
            if element == 1:
                game_state[i] = 2
            elif element == 2:
                game_state[i] = 1
        return mirrored_game_state


    def __init__(self, player_n, exploration, symmetric_aware=False, mirror=False, dummy=False):
        self.exploration = exploration
        self.player_n = player_n
        self.symmetric_aware = symmetric_aware
        # If the policies read/updated need to be done in a mirror fashion.
        self.mirror = mirror
        self.game_logs = []
        self.policy = {}
        self.dummy = dummy


    def set_policy(self, policy):
        # In order to allow for shared policies between agents.
        self.policy = policy


    def update_policy(self, game_state, last_move, value):
        if self.mirror:
            game_state = Agent.mirrored_game_state(game_state)
        # Helper functions for symmetric awareness
        def _swap(i,i_prime, game_state):
            temp = game_state[i]
            game_state[i] = game_state[i_prime]
            game_state[i_prime] = temp

        def _swap_x(game_state):
            _swap(0,2, game_state)
            _swap(3,5, game_state)
            _swap(6,8, game_state)
            return game_state

        def _swap_y(game_state):
            _swap(0,6, game_state)
            _swap(1,7, game_state)
            _swap(2,8, game_state)
            return game_state

        def _swap_diagonal(game_state):
            _swap(0,8, game_state)
            _swap(3,7, game_state)
            _swap(1,5, game_state)
            return game_state

        if self.symmetric_aware:
            x_swapped_game_state = _swap_x(game_state[:])
            y_swapped_game_state = _swap_y(game_state[:])
            diagonal_swapped_game_state = _swap_diagonal(_swap_x(game_state[:]))
            self.policy[str((x_swapped_game_state, last_move))] = value
            self.policy[str((y_swapped_game_state, last_move))] = value
            self.policy[str((diagonal_swapped_game_state, last_move))] = value
        self.policy[str((game_state, last_move))] = value


    def set_exploration(self, exploration):
        self.exploration = exploration


    def enter(self, game):
        self.game = game


    def state_p(self, game_state, last_move):
        # TODO: Make this always from the perspective of player 1
        if self.mirror:
            game_state = Agent.mirrored_game_state(game_state)
        # return the probability of the game state to win according to policy
        state_p_value = self.policy.get(str((game_state, last_move)))
        if state_p_value is None:
            self.update_policy(game_state[:], last_move, 0.5)
            return 0.5
        return state_p_value


    def play(self):
        # Compute the likelihoods for each move
        possible_move_infos = []
        for move in self.game.possible_moves:
            possible_game_state = self.game.game_state[:]
            possible_game_state[move] = self.player_n
            # possible_move_infos = [(percentage_chance_to_win, move, game_state_as_a_result)]
            possible_move_infos.append((self.state_p(possible_game_state, move), move, possible_game_state))

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
            target_move = np.random.randint(0, non_optimal_move_counts + 1) # +1 because end of range is exclusive
        # If exploit choose one of the optimals
        else:
            target_move = np.random.randint(non_optimal_move_counts, len(possible_move_infos))
        self.game.play_move(self.player_n, possible_move_infos[target_move][1])
        self.game_logs.append(possible_move_infos[target_move])


    def back_propagate_policies(self, alpha, reward):
        if not self.dummy:
            # update final move policy value
            self.update_policy(
                self.game_logs[-1][2][:], self.game_logs[-1][1], reward)
            # back propagate policy values
            for i in range(len(self.game_logs) -2, 0, -1):
            # state probability = state probability + alpha(state probs next - current)
                current_p = self.state_p(self.game_logs[i][2], self.game_logs[i][1])
                next_p = self.state_p(self.game_logs[i+1][2], self.game_logs[i+1][1])
                update_p = current_p + (alpha * (next_p - current_p))
                self.update_policy(self.game_logs[i][2][:], self.game_logs[i][1], update_p)
            # Clear game logs in case we want to play another game with this agent
            self.game_logs = []


if __name__ == "__main__":

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
        """
        if kwargs.get('training'):
            training = kwargs.get('training')
            alpha = kwargs.get('alpha') if kwargs.get('alpha') is not None else 0.2
            decrease_factor = kwargs.get('decrease_factor') if kwargs.get('decrease_factor') is None else 0.9
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
                if games_left % (math.ceil(games_left/50)) == 0:
                    alpha = alpha * decrease_factor
            games_left -= 1


    def train(iterations, a1, a2):
        # Hyperparameters
        alpha = 0.2
        decrease_factor = 0.9
        logger = Logger()
        run_games(iterations, a1, a2, logger, training=True,
                  alpha=alpha, decrease_factor=decrease_factor)
        if a2.dummy:
            print("TRAIN VS. DUMMY: agent 1 wins: {}, agent 2 wins: {}, ties: {}".format(
                (logger.agent_1_wins / iterations), (logger.agent_2_wins / iterations),
                (logger.ties / iterations)))
        else:
            print("TRAIN VS. LEARNING AGENT: agent 1 wins: {}, agent 2 wins: {}, ties: {}".format(
                (logger.agent_1_wins / iterations), (logger.agent_2_wins / iterations),
                (logger.ties / iterations)))
        # returns the better agent
        if logger.agent_1_wins > logger.agent_2_wins:
            return a1
        return a2


    def test(iterations, a1, a2):
        logger = Logger()
        run_games(iterations, a1, a2, logger)
        print("TEST VS. DUMMY: agent 1 wins: {}, agent 2 wins: {}, ties: {}".format( (
            logger.agent_1_wins / iterations ), ( logger.agent_2_wins / iterations ),
            (logger.ties / iterations)))

    AGENT_1 = Agent(1, 0.8)
    AGENT_2 = Agent(2, 0.8, dummy=True)
    train(50000, AGENT_1, AGENT_2)
    AGENT_1.set_exploration(0)
    AGENT_3 = Agent(2, 0.5, dummy=True)
    test(50000, AGENT_1, AGENT_3)

    AGENT_1 = Agent(1, 0.8)
    AGENT_2 = Agent(2, 0.8)
    WINNER = train(50000, AGENT_1, AGENT_2)
    WINNER.player_n = 1
    WINNER.set_exploration(0)
    AGENT_3 = Agent(2, 0.5, dummy=True)
    test(50000, WINNER, AGENT_3)

    print("Now with symmetric awareness")
    AGENT_1 = Agent(1, 0.8, symmetric_aware=True)
    AGENT_2 = Agent(2, 0.8)
    WINNER = train(50000, AGENT_1, AGENT_2)
    WINNER.player_n = 1
    WINNER.set_exploration(0)
    AGENT_3 = Agent(2, 0.5, dummy=True)
    test(50000, WINNER, AGENT_3)

    print("Now with self play")
    AGENT_1 = Agent(1, 0.8)
    AGENT_2 = Agent(2, 0.8, mirror=True)
    AGENT_2.set_policy(AGENT_1.policy)
    WINNER = train(50000, AGENT_1, AGENT_2)
    WINNER.player_n = 1
    WINNER.set_exploration(0)
    AGENT_3 = Agent(2, 0.5, dummy=True)
    test(50000, WINNER, AGENT_3)

    print("Now with self play & symmetric awareness")
    AGENT_1 = Agent(1, 0.8)
    AGENT_2 = Agent(2, 0.8)
    AGENT_2.set_policy(AGENT_1.policy)
    WINNER = train(50000, AGENT_1, AGENT_2)
    WINNER.player_n = 1
    WINNER.set_exploration(0)
    AGENT_3 = Agent(2, 0.5, dummy=True)
    test(50000, WINNER, AGENT_3)
