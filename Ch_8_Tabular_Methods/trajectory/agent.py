import numpy as np
from trajectory.states import States

class Agent():

    def __init__(self, states, trajectory_sampling):
        # whether or not we are using trajectory sampling or uniform
        self.trajectory_sampling = trajectory_sampling
        self.num_states = states.num_states
        self.states = states
        # initialize Q values
        self.Q = {}
        for state in range(self.num_states):
            for action in range(2):
                self.Q[(state, action)] = 0
        self.Q[(None, 0)] = 0
        self.Q[(None, 1)] = 0
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
        self.s_a_plan = (0, 0)
        # for graphing later
        self.start_s_values = []

    def get_expected_value(self, state):
        if self.Q[(state, 0)] > self.Q[(state, 1)]:
            return ((1.0 - (self.epsilon/2.0)) * self.Q[(state, 0)]) + ((self.epsilon/2.0) * self.Q[(state, 1)])
        return ((1.0 - (self.epsilon/2.0)) * self.Q[(state, 1)]) + ((self.epsilon/2.0) * self.Q[(state, 0)])

    def get_max_action(self, state):
        if self.Q[(state, 0)] > self.Q[(state, 1)]:
            return 0
        return 1

    def select_action(self, state):
        if np.random.rand() > self.epsilon:
            return self.get_max_action(state)
        return np.random.randint(2)

    def calculate_true_v(self):
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
                # 0.9 * rewards + 0.1 * finish
                # a list of (r, s_primes)
                new_transitions += self.states.state_map[cur_state][action]
                transitions += self.states.state_map[cur_state][action]
            unprocessed_transitions = new_transitions
        # then calculate true v going backwards
        for i in range(len(transitions)-1,-1,-1):
            cur_state = transitions[i][1]
            action = self.get_max_action(cur_state)
            next_transitions = self.states.state_map[cur_state][action]
            reward_sum = 0
            for t in next_transitions:
                # only base case should be 0
                state_value = 0 if state_value_map.get(t[1]) is None else state_value_map[t[1]]
                reward_sum += t[0] + state_value
            state_value_map[cur_state] = 0.9 * (1/len(next_transitions)) * reward_sum
        self.start_s_values.append(state_value_map[0])

    def get_possible_action(self, state):
        """Get possible actions according to what has been sampled before

        Args:
            state (int): state to get actions leading out of

        Returns:
            action (int): either the only observed action or an action chosen according to policy
        """
        if self.model.get((state, 1)) is not None and self.model.get((state, 0)) is not None:
            return self.select_action(state)
        if self.model.get((state, 1)) is not None:
            return 1
        if self.model.get((state, 0)) is not None:
            return 0
        return None

    def increment_s_a_plan(self):
        """in the case of uniform planning, update the s_a_pair to the next in the list
        """
        if self.s_a_plan[1] == 1:
            if self.s_a_plan[0] >= self.num_states - 1:
                self.s_a_plan = (0,0)
            else:
                self.s_a_plan = (self.s_a_plan[0] + 1, 0)
        else:
            self.s_a_plan = (self.s_a_plan[0], 1)

    def plan(self):
        """planning step
        """
        if self.trajectory_sampling:
            action = self.get_possible_action(self.plan_state)
            if action is None:
                # return to beginning
                self.plan_state = 0
                action = self.get_possible_action(self.plan_state)
            r, s_prime = self.model[(self.plan_state, action)]
            expected_value = self.get_expected_value(s_prime)
            self.Q[(self.plan_state, action)] += (self.alpha * \
                ((r + (self.discount_rate * expected_value)) -
                self.Q[(self.plan_state, action)]))
            self.plan_state = s_prime
            if s_prime is None:
                self.plan_state = 0
        else:
            state, action = list(self.model.keys())[self.plan_state]
            r, s_prime = self.model[(state, action)]
            expected_value = self.get_expected_value(s_prime)
            self.Q[(state, action)] += (self.alpha * \
                ((r + (self.discount_rate * expected_value)) -
                 self.Q[(state, action)]))
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
        r, s_prime = self.states.time_step(self.state, action)
        self.model[(self.state, action)] = (r, s_prime)
        expected_value = self.get_expected_value(s_prime)
        self.Q[(self.state, action)] += self.alpha * \
            ((r + (self.discount_rate * expected_value)) -
             self.Q[(self.state, action)])
        self.state = s_prime
        self.calculate_true_v()
        for i in range(self.n):
            self.plan()
        # reset if we're at the end
        if self.state is None:
            self.state = 0
