# RL Tic Tac Toe

This project contains a tic tac toe agent that uses temporal difference to learn. Multi threaded training, ensembled agents, symmetric awareness, interface for a human to vs. the agent. It has an 'expert' agent taken from [billtub's gamelearner](https://github.com/billtubbs/game-learner) and a random agent. The temporal difference update rule is described below.

If you would like to vs. our meta agent then you may play it by running `python game.py` on python 3.5.2. Please remember to `pip install requirements.txt` before running the game.

The game will show you what the agent is "thinking" by coloring available blocks according to its policy map values before it makes its move. Red being the lowest value, green the greatest.

![](./output/screenshots/game.png)


## Update rule for a temporal difference model

`V(s) = V(s) + a * (V(s') - V(s))`

Where `V(s)` = estimated value of state

`s` = state

`s'` = next state

`a` = step size parameter AKA rate of learning

The step size parameter is generally reduced over time.

This is considered a temporal difference model update rule because its updates are based on the estimates of a state from two different times.
