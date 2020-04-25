
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
