## Exercise 4.1
*If π is the equiprobable random policy, what is qπ(11, down)?
What is qπ(7, down)?*

$qπ(11, down) = \sum_{s'}p(s',r|s=11,a=down) * [r_{11,s'} + vπ(s')]$

$qπ(11, down) = 1 * [0 + 0]$

$qπ(11, down) = 0$

and...

$qπ(7, down) = 1 * [-1 + -14]$

$qπ(7, down) = -15$

## Exercise 4.2

*Suppose a new state 15 is added to the gridworld just below
state 13, and its actions, left, up, right, and down, take the agent to states
12, 13, 14, and 15, respectively. Assume that the transitions from the original
states are unchanged. What, then, is vπ(15) for the equiprobable random
policy? Now suppose the dynamics of state 13 are also changed, such that
action down from state 13 takes the agent to the new state 15. What is vπ(15)
for the equiprobable random policy in this case?*

## Exercise 4.3

*What are the equations analogous to (4.3), (4.4), and (4.5) for
the action-value function qπ and its successive approximation by a sequence of
functions q0, q1, q2, . . . ?*

## Exercise 4.4

*In some undiscounted episodic tasks there may be policies
for which eventual termination is not guaranteed. For example, in the grid
problem above it is possible to go back and forth between two states forever.
In a task that is otherwise perfectly sensible, vπ(s) may be negative infinity
for some policies and states, in which case the algorithm for iterative policy
evaluation given in Figure 4.1 will not terminate. As a purely practical matter,
how might we amend this algorithm to assure termination even in this case?
Assume that eventual termination is guaranteed under the optimal policy.*

## Exercise 4.5 (programming)

*Write a program for policy iteration and
re-solve Jack’s car rental problem with the following changes. One of Jack’s
employees at the first location rides a bus home each night and lives near
the second location. She is happy to shuttle one car to the second location
for free. Each additional car still costs $2, as do all cars moved in the other
direction. In addition, Jack has limited parking space at each location. If
more than 10 cars are kept overnight at a location (after any moving of cars),
then an additional cost of $4 must be incurred to use a second parking lot
(independent of how many cars are kept there). These sorts of nonlinearities
and arbitrary dynamics often occur in real problems and cannot easily be
handled by optimization methods other than dynamic programming. To check
your program, first replicate the results given for the original problem. If your
computer is too slow for the full problem, cut all the numbers of cars in half.*

## Exercise 4.6

*How would policy iteration be defined for action values? Give
a complete algorithm for computing q∗, analogous to Figure 4.3 for computing
v∗. Please pay special attention to this exercise, because the ideas involved
will be used throughout the rest of the book.*

## Exercise 4.7

*Suppose you are restricted to considering only policies that are
-soft, meaning that the probability of selecting each action in each state, s,
is at least /|A(s)|. Describe qualitatively the changes that would be required
in each of the steps 3, 2, and 1, in that order, of the policy iteration algorithm
for v∗ (Figure 4.3).*

## Exercise 4.8

*Why does the optimal policy for the gambler’s problem have
such a curious form? In particular, for capital of 50 it bets it all on one flip,
but for capital of 51 it does not. Why is this a good policy?*

## Exercise 4.9 (programming)

*Implement value iteration for the gambler’s
problem and solve it for ph = 0.25 and ph = 0.55. In programming, you may
find it convenient to introduce two dummy states corresponding to termination
with capital of 0 and 100, giving them values of 0 and 1 respectively. Show
your results graphically, as in Figure 4.6. Are your results stable as θ → 0?*

## Exercise 4.10

*What is the analog of the value iteration backup (4.10) for
action values, qk+1(s, a)?*