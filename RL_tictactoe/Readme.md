# Tic Tac Toe

## Update rule for a temporal difference model

`V(s) = V(s) + a * (V(s') - V(s))`

Where `V(s)` = estimated value of state

`s` = state

`s'` = next state

`a` = step size parameter AKA rate of learning

The step size parameter is generally reduced over time.

This is considered a temporal difference model update rule because its updates are based on the estimates of a state from two different times.

## Extended Tic Tac Toe Temporal Difference Example

A simplified Tic Tac Toe where ties are also considered losses. Implementation using the above update rule in `tictactoe.py`

If you would like to vs. this agent then you may play it by running `python game.py` on python 3.5.2
