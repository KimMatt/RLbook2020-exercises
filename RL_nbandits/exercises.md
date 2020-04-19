# Chapter 2

## Exercise 2.1
*In the comparison shown in Figure 2.1, which method will
perform best in the long run in terms of cumulative reward and cumulative
probability of selecting the best action? How much better will it be? Express
your answer quantitatively.*


## Exercise 2.2
*Give pseudocode for a complete algorithm for the n-armed
bandit problem. Use greedy action selection and incremental computation of
action values with α = 1/k step-size parameter. Assume a function bandit(a) that takes an action and returns a reward. Use arrays and variables; do not subscript anything by the time index t (for examples of this style of pseudocode, see Figures 4.1 and 4.3). Indicate how the action values are initialized
and updated after each reward. Indicate how the step-size parameters are set
for each action as a function of how many times it has been tried.*


```
Initialize policy map [] of length n to val_init (default 0)
Set alpha = 1.0
Set k map = [1.0 array of length n]
Set exploration = some constant between 0 and 1
Repeat while playing:
    random = random value between 0 and 1 sampled from a uniform distribution
    If random <= explore:
        arm = random integer between 0 and n
    else:
        arm = index of max value from policy map
    reward = bandit(arm)
    Set k maps[arm] += 1.0
    policy_map[arm] += (1.0/k maps[arm]) * (reward - policy_map[arm])
```

## Exercise 2.3
*If the step-size parameters, αk, are not constant, then the estimate Qk is a weighted average of previously received rewards with a weighting
different from that given by (2.6). What is the weighting on each prior reward
for the general case, analogous to (2.6), in terms of αk?*

## Exercise 2.4
*Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for nonstationary
problems. Use a modified version of the 10-armed testbed in which all the
q(a) start out equal and then take independent random walks. Prepare plots
like Figure 2.1 for an action-value method using sample averages, incrementally computed by α = 1/k, and another action-value method using a constant
step-size parameter, α = 0.1. Use ε = 0.1 and, if necessary, runs longer than
1000 plays.*

![](figs/Exercise_2.4.png)

These are the cumulative rewards of running the described experiment 100 times with 1000 iterations each.

```sample wins: 36 const wins: 60 ties: 4```

The reason we ran so many iterations of the experiment was because when we ran them independently, the results were inconsistent. However over a large number of trials the sample methods did worse than the constant method.

During the creation of this experiment it was discovered that the sample method actually does better than the constant method when every arm takes a random walk in the same direction. In other words, when the entire n-armed bandits takes a random walk. Here the bandit would move all distributions in the same direction with the same unit.

![](figs/Exercise_2.4_1.png)

```sample wins: 75 const wins: 25 ties: 0```

The sample method will perform better than a constant method on a non stationary problem where the entire environment changes in the same direction.

This is because if all the arms are moving in the same direction, then whatever is optimal will remain optimal. This problem is really only technically stationary. Perhaps stationary should be defined in such a way that the optimal actions must change.

We were curious what would happen if we made a bandit which had a random walk with dependent parts that did not all move in the same direction, but had some sort of alternating rule where if one went up, the other went down. So we did one with an alternating random walk where each arm would do the opposite direction of the previous.

![](figs/Exercise_2.4_2.png)

```sample wins: 31 const wins: 63 ties: 6```

The result was consistent with the first experiment. Thus, we can see that the constant update method performs better for a nonstationary problem where the optimal arm is subject to change.

With a constant method new rewards are weighed in with equal magnitude to the policy. In sampling methods, the weight of new rewards are diminished as time increases because k grows larger with each time step. When the optimal arm changes it's difficult for the sampling method to take enough steps to make enough change to its policy to reflect this. As time goes on, the sampling method actually get worse at this. This is because it's slowing down the rate of change to its policies while the optimal arm's rate of change is random.

This lead to a curiosity of what would happen if the weight of old rewards diminished over time?

For this, we want a monotically increasing function. I chose to use `f(x) = tanh(x)` multiplied by a constant scalar.

This resulted in a competitive result to the constant method. The "increasing method" did not do as well against constant at the beginning. However, when I extended the length of a trial from 1000 to 5000 it became competitive; sometimes it would do better, equal, or worse. Overall, they always had a majority tied.

![](figs/Exercise_2.4_3.png)

```increasing wins: 14 const wins: 12 ties: 74```


## Exercise 2.5
*The results shown in Figure 2.2 should be quite reliable because they are averages over 2000 individual, randomly chosen 10-armed bandit tasks. Why, then, are there oscillations and spikes in the early part of
the curve for the optimistic method? What might make this method perform
particularly better or worse, on average, on particular early plays?*

## Exercise 2.6

*Suppose you face a binary bandit task whose true action values change randomly from play to play. Specifically, suppose that for any play the true values of actions 1 and 2 are respectively 0.1 and 0.2 with probability 0.5
(case A), and 0.9 and 0.8 with probability 0.5 (case B). If you are not able to
tell which case you face at any play, what is the best expectation of success
you can achieve and how should you behave to achieve it? Now suppose that
on each play you are told if you are facing case A or case B (although you still
don’t know the true action values). This is an associative search task. What
is the best expectation of success you can achieve in this task, and how should
you behave to achieve it?*