

diff = 1

state_rewards = [0, 0, 1]
state_values = [1/3, 2/3, 2/3]
avg_r = 1/3
alpha = 1
beta = 1

while diff > 0:
    # update each state value to be
    # state_reward - avg_r + next state_value - current_state value
    diff = 0

    for index, value in enumerate(state_values):
        next_state_value = state_values[index + 1] if index + \
            1 < len(state_values) else state_values[0]
        print(avg_r)
        delta = state_rewards[index] - avg_r + \
            next_state_value - value
        print("delta:{}".format(delta))
        avg_r += beta * delta
        diff += delta
        state_values[index] += alpha * delta
        print("delta:{}".format(delta))
        # state_values[index] += alpha * delta

    print(state_values)
    print(avg_r)
