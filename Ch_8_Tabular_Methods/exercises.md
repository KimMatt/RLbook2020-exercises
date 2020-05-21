## Exercise 8.1
*The nonplanning method looks particularly poor in Figure 8.3 because it is a one-step method; a method using multi-step bootstrapping would do better. Do you
think one of the multi-step bootstrapping methods from Chapter 7 could do as well as
the Dyna method? Explain why or why not.*

In terms of pure episode to step ratio, a multi-step boostrapping method may not do as well as the dyna method, especially since the agent starts at random positions.

Although many states are updated with a multi-step boostrapping method, the result of which are updated and which direction the maximum Q values are updated to are based on the random path that's been taken. In the case of a Dyna method with n=50, after a single step a large area around the single learned path is updated. For almost 2/3 of the grid we have learned optimal paths to the goal.

However, there may be cases where this is not true depending on the state space. Dyna is great for this example due to the random nature of the agent's start point, and due to the states being in a grid. There may still yet be some specific problems that are exceptions. Perhaps, when it is very difficult to build a model from samples.

## Exercise 8.2
*Why did the Dyna agent with exploration bonus, Dyna-Q+, perform
better in the first phase as well as in the second phase of the blocking and shortcut
experiments?*

It performed better because the added reward of exploration (since reward give 0 base) helped it find the most optimal path in fewer episodes by having it perform almost a methodic "sweep" of S,A pairs close to the reward.

I would estimate, however, that in the long run, without the environment change, that the Dyna-Q+ would eventually lose its lead on the Dyna-Q, since the optimal path has already been found and Dyna-Q+ is simpy exporing more.

## Exercise 8.3
*Careful inspection of Figure 8.5 reveals that the difference between Dyna-Q+ and Dyna-Q narrowed slightly over the first part of the experiment. What is the reason
for this?*

This is because once Dyna-Q+ and Dyna-Q learn policies, exploratory moves would only cause Dyna-Q+ to lose its 'lead' on Dyna-Q. We see this starting to happen in the first stage.

## Exercise 8.4 (programming)
*The exploration bonus described above actually changes
the estimated values of states and actions. Is this necessary? Suppose the bonus 
p⌧ was used not in updates, but solely in action selection. That is, suppose the action selected was always that for which Q(St, a) + 
p⌧ (St, a) was maximal. Carry out a
gridworld experiment that tests and illustrates the strengths and weaknesses of this
alternate approach.*
