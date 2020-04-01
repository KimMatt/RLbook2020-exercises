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

I would expect the self play agent to perform better. I would expect it to learn a more intelligent way of playing and for it to learn faster. It would learn faster because it shares a policy map with its opponent. Thus, both agents would learn from each other's mistakes and wins.

```
TRAIN VS. DUMMY: agent 1 wins: 0.73694, agent 2 wins: 0.22716, ties: 0.0359
VS. DUMMY: agent 1 wins: 0.5602, agent 2 wins: 0.3292, ties: 0.1106
TRAIN VS. LEARNING AGENT: agent 1 wins: 0.28776, agent 2 wins: 0.28792, ties: 0.42432
VS. DUMMY: agent 1 wins: 0.7424, agent 2 wins: 0.2464, ties: 0.0112
Now with self play
TRAIN VS. LEARNING AGENT: agent 1 wins: 0.28552, agent 2 wins: 0.27866, ties: 0.43582
VS. DUMMY: agent 1 wins: 0.793, agent 2 wins: 0.172, ties: 0.035
```

Both agents played fully greedy when vs. the dummy agent. The agent that trained with self play performed almost 5% better than the agent that trained against an seperate learning agent and more than 20% better than the agent which trained against a random opponent.

Perhaps this is closer to reality. Consider when a human player sees another player win, they may try to copy that winning strategy the next time they play.

### Exercise 1.2: Symmetries
Many tic-tac-toe positions appear different but are really the same because of symmetries. How might we amend the reinforcement learning algorithm described above to take advantage of this? In what ways would this improve it? Now think again. Suppose the opponent did not take advantage of symmetries. In that case, should we? Is it true, then, that symmetrically equivalent positions should necessarily have the same value?

The policies should map all symmetrical states to the same estimated value when updating them.

This could improve our RL method by allowing agents to learn faster. An agent would not have to relearn the same estimates for two states that would otherwise be the same.

If the opponent does not take advantage of the symmetries, they would learn at a slower rate. Also, if a symmetrical-aware agent learns how to defeat the opponent with one strategy, it can easily defeat it in the same way, but with a symmetrical version. Thus, we should treat the symmetrical states equivalently.

```
TRAIN VS. LEARNING AGENT: agent 1 wins: 0.28762, agent 2 wins: 0.29042, ties: 0.42196
VS. DUMMY: agent 1 wins: 0.7108, agent 2 wins: 0.2758, ties: 0.0134
Now with symmetric awareness
TRAIN VS. LEARNING AGENT: agent 1 wins: 0.28752, agent 2 wins: 0.27084, ties: 0.44164
VS. DUMMY: agent 1 wins: 0.7794, agent 2 wins: 0.2156, ties: 0.005
```

The symmetrically aware agent performed ~5% better than the non symmetrically aware agent.


### Exercise 1.3: Greedy Play
Suppose the reinforcement learning player was
greedy, that is, it always played the move that brought it to the position that
it rated the best. Would it learn to play better, or worse, than a nongreedy
player? What problems might occur?

It would learn to play worse than a non greedy player, especially at the beginning of the learning phase. It's beneficial to have a high exploration vs. exploitation value when initially learning, because the agent really knows nothing at the beginning... Once an exploitive agent learns a few ways to win, it will stick to those potentially suboptimal moves and not explore to learn the actual optimal moves.

```
TRAIN VS DUMMY: agent 1 wins: 0.73524, agent 2 wins: 0.22646, ties: 0.0383
VS. DUMMY: agent 1 wins: 0.5544, agent 2 wins: 0.335, ties: 0.1106
TRAIN VS DUMMY: agent 1 wins: 0.1462, agent 2 wins: 0.28714, ties: 0.56666
VS. DUMMY: agent 1 wins: 0.725, agent 2 wins: 0.2248, ties: 0.0502
```

The first trial was with a fully greedy agent while the second trial was with a fully explorative agent. The results align with my hypothesis so I'm going to stick to my guns and believe my logic is sound :)

Both agents played fully greedy when versing the dummy.

###Exercise 1.4: Learning from Exploration 
Suppose learning updates occurred after all moves, including exploratory moves. If the step-size parameter is appropriately reduced over time, then the state values would converge to a set of probabilities. What are the two sets of probabilities computed when we do, and when we do not, learn from exploratory moves? Assuming that we do continue to make exploratory moves, which set of probabilities might be better to learn? Which would result in more wins?

I'm not sure if this is asking if the learning updates occur after all moves of a game or after each move? Is the expected a

###Exercise 1.5: Other Improvements 
Can you think of other ways to improve the reinforcement learning player? Can you think of any better way to solve the tic-tac-toe problem as posed?

Currently, ties are treated the same as losses. However, a tie would be considered more of a win than a complete loss. And a good tic tac toe player should be able to tie more often than lose. Treating the tie in a way such that it is slightly better than a loss could improve the reinforcement learning player.