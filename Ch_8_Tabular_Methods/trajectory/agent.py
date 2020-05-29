import numpy as np
from trajectory.states import States

class Agent():

    def __init__(self, states, trajectory_sampling):
        # whether or not we are using trajectory sampling or uniform
        self.trajectory_sampling = trajectory_sampling
        # our environment
        self.states = states
        self.num_states = states.num_states
        # initialize Q values
        self.Q = {}
        for state in range(self.num_states):
            for action in range(2):
                self.Q[(state, action)] = 0
        # start state
        self.state = 0
        # keep track of model
        self.model = {}
        # hyperparameters
        self.epsilon = 0.1
        self.discount_rate = 1.0
        self.alpha = 0.1
        self.n = 25
        # tracker variables for planning
        self.plan_state = 0
        # for graphing later
        self.start_s_values = []

    def update_model(self, state, action, reward, s_prime):
        """Update the stochastic model

        Args:
            state (int): state
            action (int): action
            reward (float): reward from observed transition
            s_prime (int): resultant state from observed transition
        """
        if self.model.get((state, action)) is None:
            self.model[(state, action)] = {}
        if self.model[(state,action)].get((reward, s_prime)) is None:
            self.model[(state, action)][(reward, s_prime)] = 1
        else:
            self.model[(state, action)][(reward, s_prime)] += 1

    def sample_model(self, state, action):
        """Generate a sample from the model, stochastic.

        Args:
            state ([int]): state to sample transition from
            action ([int]): action to sample transition from

        Returns:
            r, s_prime: return and s_prime
        """
        if state is None:
            return 0
        if self.model.get((state, action)) is None:
            return None
        # construct probability value to result
        model_samples = self.model.get((state, action))
        total_samples = len(model_samples)
        observed_probabilities = []
        prob_sum = 0
        for r_s_pair in model_samples.keys():
            percentage_observed = (model_samples[r_s_pair]/total_samples)
            prob_sum += percentage_observed
            observed_probabilities.append((prob_sum ,r_s_pair))
        sample_statistic = np.random.rand()
        for transition in observed_probabilities:
            if sample_statistic <= transition[0]:
                return transition[1]

    def get_expected_value(self, state):
        """calculate the expected value on epsilon-greedy policy of given state

        Args:
            state (int): given state

        Returns:
            [int]: (1 - epsilon * 1/2) * Q(state, max_action) + (epsilon * 1/2) * Q(state, lesser_action)
        """
        if state is None:
            return 0
        if self.Q[(state, 0)] > self.Q[(state, 1)]:
            return ((1.0 - (self.epsilon/2.0)) * self.Q[(state, 0)]) + ((self.epsilon/2.0) * self.Q[(state, 1)])
        return ((1.0 - (self.epsilon/2.0)) * self.Q[(state, 1)]) + ((self.epsilon/2.0) * self.Q[(state, 0)])

    def get_max_action(self, state):
        """Simply return the maximum action according to current q values

        Args:
            state (int): state to base action selection on

        Returns:
            [int]: action
        """
        if self.Q[(state, 0)] > self.Q[(state, 1)]:
            return 0
        return 1

    def select_action(self, state):
        """Select action based on epsilon-greedy policy

        Args:
            state (int): state to base action selection on

        Returns:
            [int]: action selected
        """
        if np.random.rand() > self.epsilon:
            return self.get_max_action(state)
        return np.random.randint(2)

    def calculate_true_v(self):
        """Calculate the true value of the start state given a greedy policy that operates
            on current Q-values and append it to the list of values.
        """
        transitions = [(0, 0)]
        unprocessed_transitions = [(0, 0)]
        state_value_map = {}
        accuracy = 200
        # first get the sequence of states that would be encountered
        for i in range(accuracy):
            new_transitions = []
            while len(unprocessed_transitions) > 0:
                transition = unprocessed_transitions.pop(0)
                cur_state = transition[1]
                action = self.get_max_action(cur_state)
                # a list of transitions that would be encountered on a greedy policy
                new_transitions += self.states.state_map[cur_state][action]
                transitions += self.states.state_map[cur_state][action]
            unprocessed_transitions = new_transitions
        # then calculate true v going backwards
        for i in range(len(transitions)-1,-1,-1):
            cur_state = transitions[i][1]
            action = self.get_max_action(cur_state)
            next_transitions = self.states.state_map[cur_state][action]
            # only base cases should be just reward + 0
            reward_sum = sum([transition[0] if state_value_map.get(transition[1]) is None
                                else transition[0] + state_value_map[transition[1]]
                                for transition in next_transitions])
            # 0.9 * rewards + 0.1 * finish = 0.9 * rewards + 0.1 * 0
            state_value_map[cur_state] = 0.9 * (1/len(next_transitions)) * reward_sum
        self.start_s_values.append(state_value_map[0])

    def get_possible_model_action(self, state):
        """Get possible actions according to what our model has seen

        Args:
            state (int): state to get possible actions from

        Returns:
            action (int): either the only observed action or an action chosen according to policy
        """
        if self.sample_model(state, 1) is not None and self.sample_model(state, 0) is not None:
            return self.select_action(state)
        if self.sample_model(state, 1) is not None:
            return 1
        if self.sample_model(state, 0) is not None:
            return 0
        return None

    def update_expected_Q(self, state, action, reward, s_prime):
        """Update Q-value for state and action based on expected value of epsilon-greedy policy

        Args:
            state (int): state to update
            action (int): action to update
            reward (int): reward observed
            s_prime (int): state transition observed
        """
        expected_value = self.get_expected_value(s_prime)
        self.Q[(state, action)] += (self.alpha *
                                    ((reward + (self.discount_rate * expected_value)) -
                                        self.Q[(state, action)]))

    def plan(self):
        """planning step
        """
        if self.trajectory_sampling:
            action = self.select_action(self.plan_state)
            if self.sample_model(self.plan_state, action) is None:
                # return to beginning
                self.plan_state = 0
                action = self.get_possible_model_action(self.plan_state)
            reward, s_prime = self.sample_model(self.plan_state, action)
            self.update_expected_Q(self.plan_state, action, reward, s_prime)
            self.plan_state = s_prime
            if s_prime is None:
                self.plan_state = 0
        else:
            state, action = list(self.model.keys())[self.plan_state]
            reward, s_prime = self.sample_model(state, action)
            self.update_expected_Q(state, action, reward, s_prime)
            self.plan_state += 1
            if self.plan_state >= len(self.model):
                self.plan_state = 0
        self.calculate_true_v()

    def play(self):
        """play a time step of traversing the sub problem

        Returns:
            state: resulting state
        """
        action = self.select_action(self.state)
        reward, s_prime = self.states.time_step(self.state, action)
        self.update_model(self.state, action, reward, s_prime)
        self.update_expected_Q(self.state, action, reward, s_prime)
        self.state = s_prime
        self.calculate_true_v()
        for i in range(self.n):
            self.plan()
        # reset if we're at the end
        if self.state is None:
            self.state = 0
