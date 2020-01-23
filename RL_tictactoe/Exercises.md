# Chapter 1: The Reinforcement Learning Problem

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

## Exercises

### Exercise 1.1: Self-Play
Suppose, instead of playing against a random opponent, the reinforcement learning algorithm described above played against itself. What do you think would happen in this case? Would it learn a different way of playing?

I would expect an agent to perform better against a random agent after it trains with a learning agent.

```
TRAIN VS DUMMY: agent 1 wins: 0.30496, agent 2 wins: 0.26278, ties: 0.43226
VS. DUMMY: agent 1 wins: 0.8374, agent 2 wins: 0.138, ties: 0.0246
TRAIN VS. LEARNING AGENT: agent 1 wins: 0.26078, agent 2 wins: 0.3288, ties: 0.41042
VS. DUMMY: agent 1 wins: 0.77, agent 2 wins: 0.2214, ties: 0.0086
```

Both agents played fully greedy when vs. dummy.

Contrary to my hypothesis, however, the agent that trained with a random agent performed better. This makes me think that the agent that trained against another learning agent "overfitted" to learning tactics against more intelligent agents. But when faced with a completely random agent, it did not perform as well as an agent that learned tactics which work against a more random agent.

This experiment was not self playing.


### Exercise 1.3: Greedy Play
Suppose the reinforcement learning player was
greedy, that is, it always played the move that brought it to the position that
it rated the best. Would it learn to play better, or worse, than a nongreedy
player? What problems might occur?

It would not learn as well as a non greedy player. Especially at the beginning of the learning phase, it would be more beneficial to have a high exploration vs. exploitation value. Once the agent learns a few ways to win, it will stick to those potentially suboptimal moves and not explore and learn potentially optimal moves.

```
TRAIN VS DUMMY: agent 1 wins: 0.73524, agent 2 wins: 0.22646, ties: 0.0383
VS. DUMMY: agent 1 wins: 0.5544, agent 2 wins: 0.335, ties: 0.1106
TRAIN VS DUMMY: agent 1 wins: 0.1462, agent 2 wins: 0.28714, ties: 0.56666
VS. DUMMY: agent 1 wins: 0.725, agent 2 wins: 0.2248, ties: 0.0502
```

The first trial was with a fully greedy agent 1 while the second trial was with a fully explorative agent. The results align with my hypothesis so I'm going to stick to my guns and believe my logic is sound :)

Both agents played fully greedy when versing the dummy.