## Exercise 10.1
*We have not explicitly considered or given pseudocode for any Monte Carlo
methods in this chapter. What would they be like? Why is it reasonable not to give
pseudocode for them? How would they perform on the Mountain Car task?*

The Monte Carlo methods would be like the TD(n) methods except where n is equivalent to the length of the entire episode. It's reasonable not to give pseudocde for them because it would be the same as a TD(n) where n = T-t.

They would probably not very well as we see the performance starts to get worse as n gets too large.

## Exercise 10.2
*Give pseudocode for semi-gradient one-step Expected Sarsa for control.*


```
Input: a differentiable action-value function parameterization q: S x A
-> R
Algorithm parameters: step size alpha > 0, small epsilon > 0
Initialize value-function weights w arbitrarily
pi(a|S) based on current q values that operates on epsilon-greedy

Loop for each episode:
    S, A <- initial state and action of episode (e.g epsilon-greedy)
    Loop for each step of episode:
        Take action A, observe R, S'
        If S' is terminal:
            w <- w + alpha[R - Q(S, A)] * gradient of Q w.r.t w
            Go to next episode
        Ut = R + sum of pi(a|S') * q(S',a) for all actions a
        w <- w + alpha[R - Ut] * gradient of Q w.r.t w
        Choose A' as a function of Q(S',,w) according to epsilon greedy policy
        S <- S'
        A <- A'
```


## Exercise 10.3
*Why do the results shown in Figure 10.4 have higher standard errors at
large n than at small n?*

Because when n is too large it makes it harder to learn as effectively.
We learn "strokes" of rewards when the agent is exploring, so one pattern of actions
may gain reward estimates from another pattern of actions that is unrelated.

## Exercise 10.4
*Give pseudocode for a differential version of semi-gradient Q-learning.*

```
Input: a differentiable action-value function parameterization q : S x A x R -> R
Algorithm parameters: step sizes alpha, beta > 0
Initialize value-function weights w in R arbitrarily
Initialize average reward estimate R average arbitrarily

Initialize state S and action A

Loop for each step:
    Take action A, Observe R, S'
    Choose A' as a function of q(S',,w) (epsilon greedy)
    delta <- R - R average + max_a q(S',a, w) - q(S,A,w)
    R average <- + beta * delta
    w <- w + alpha * delta * derivateive of q(S,A,w) w.r.t w
    S <- S'
    A <- A'

```

## Exercise 10.5
*What equations are needed (beyond 10.10) to specify the di↵erential
version of TD(0)?*

The update rule for the reward rate seems to be the only equation needed beyond 10.10 since the bellman value function is only used to get 10.10.

## Exercise 10.6
*Suppose there is an MDP that under any policy produces the deterministic
sequence of rewards +1, 0, +1, 0, +1, 0,... going on forever. Technically, this violates
ergodicity; there is no stationary limiting distribution µ⇡ and the limit (10.7) does not
exist. Nevertheless, the average reward (10.6) is well defined. What is it? Now consider
two states in this MDP. From A, the reward sequence is exactly as described above,
starting with a +1, whereas, from B, the reward sequence starts with a 0 and then
continues with +1, 0, +1, 0,.... We would like to compute the di↵erential values of A and
B. Unfortunately, the differential return (10.9) is not well defined when starting from these states as the implicit limit does not exist. To repair this, one could alternatively define the differential value of a state as (10.13) Under this definition, what are the differential values of states A and B?*

The average reward according to 10.6 would be 0.5.

$v_{\pi}(s) = \lim_{\gamma \to 1}\lim_{h\to \infty}\sum_{t=0}^{h}\gamma^t(E_{\pi}[R_{t+1}|S_0=s] - r(\pi))$

First let's consider $v_{\pi}(A)$

We know the reward will either alternate between 0 - 0.5 and 1 - 0.5 depending on if t is an odd or even timestep.

$v_{\pi}(A) = \lim_{\gamma \to 1}\lim_{h\to \infty}\sum_{t=0}^{h}\gamma^t \frac{-1^t}{2}$

$= \frac{1}{2} \lim_{\gamma \to 1}\lim_{h\to \infty}\sum_{t=0}^{h}(-\gamma)^t$

This is a geometric series.

$= \frac{1}{2} \lim_{\gamma \to 1}\lim_{h\to \infty}\frac{1 + \gamma^h}{1 + \gamma}$

Since we assume $\gamma$ is less than 1.

$= \frac{1}{2} \lim_{\gamma \to 1}\frac{1}{1 + \gamma}$

$= \frac{1}{2} * \frac{1}{2}$

$v_{\pi}(A) = \frac{1}{4}$

For $V(B)$ it is almost identical except:

$v_{\pi}(B) = \lim_{\gamma \to 1}\lim_{h\to \infty}\sum_{t=0}^{h}\gamma^t \frac{-1^{(t+1)}}{2}$

$v_{\pi}(B) = \lim_{\gamma \to 1}\lim_{h\to \infty}\sum_{t=0}^{h} (-1) *\gamma^t \frac{-1^t}{2}$

$v_{\pi}(B) = (-1) * \lim_{\gamma \to 1}\lim_{h\to \infty}\sum_{t=0}^{h}\gamma^t \frac{-1^t}{2}$

$v_{\pi}(B) = -\frac{1}{4}$


## Exercise 10.7
*Consider a Markov reward process consisting of a ring of three states A, B,
and C, with state transitions going deterministically around the ring. A reward of +1 is
received upon arrival in A and otherwise the reward is 0. What are the di↵erential values
of the three states, using (10.13)?*

Skip

## Exercise 10.8
*The pseudocode in the box on page 251 updates R¯t+1 using t as an error
rather than simply Rt+1  R¯t+1. Both errors work, but using t is better. To see why,
consider the ring MRP of three states from Exercise 10.7. The estimate of the average
reward should tend towards its true value of 1
3 . Suppose it was already there and was held
stuck there. What would the sequence of Rt  R¯t errors be? What would the sequence of
t errors be (using Equation 10.10)? Which error sequence would produce a more stable
estimate of the average reward if the estimate were allowed to change in response to the
errors? Why?*

In the case of R - R¯ the three updates would be -1/3, -1/3, 2/3.

In the case of $/delta$ the updates would depend on the initial values. Assuming an initial value of 0 for all states, our updates would be:

$\delta(A) = 0 - 1/3 + 0 - 0 = -1/3$
$v(A) = 0 - 1/3 + 0 = -1/3$ (according to our differential value definition for TD(0))
$\delta(B) = 0 - 1/3 + 0 - 0 = -1/3$
$v(B) = 0 - 1/3 + 0 = -1/3$
$\delta(C) = 1 - 1/3 + 0 + 1/3 = 1$
$v(C) = 1 - 1/3 - 1/3 = 1/3$

Now let's consider if the estimate were allowed to change in response to the errors:

R - R¯
Step 1: R¯ = 0
Step 2: 0 - 0 = 0 thus R¯ = 0
Step 3: 1 - 0 = 1 thus R¯ += 1 = 1

$/delta$ case:
Step 1: R¯ = 0
Step 2:
$delta(B) = 0 - 0 + 0 - 0 = 0$
$v(B) = 0 - 0 + 0 = 0$
R¯ = 0
Step 3:
$delta(C) = 1 - 0 + 0 + 1/3 = 4/3$
v(C) = 1 - 0 -1/3 = 1/3

Both methods fluctuate our reward. After running a script of the two methods, I found that using $delta$ allowed our changes to converge more quickly, while using R - R¯ seemed to go on forever, actually driving our values to insanely large numbers.

## Exercise 10.9
*In the differential semi-gradient n-step Sarsa algorithm, the step-size
parameter on the average reward, , needs to be quite small so that R¯ becomes a good
long-term estimate of the average reward. Unfortunately, R¯ will then be biased by its
initial value for many steps, which may make learning inecient. Alternatively, one could
use a sample average of the observed rewards for R¯. That would initially adapt rapidly
but in the long run would also adapt slowly. As the policy slowly changed, R¯ would also
change; the potential for such long-term nonstationarity makes sample-average methods
ill-suited. In fact, the step-size parameter on the average reward is a perfect place to use
the unbiased constant-step-size trick from Exercise 2.7. Describe the specific changes
needed to the boxed algorithm for di↵erential semi-gradient n-step Sarsa to use this
trick.*

We would simply add the trace variable and update $\Beta$ at each step as follows:

```
o += o + beta *(1 - o)
beta = const_beta / o
```

We would also initialize o to 0 and const_beta to a value between 0 and 1.