

### Defining a Policy
- $A$ = set of all possible actions
- $S$ = set of all possible states
- $\theta$ = parameters that define the conditional distribution


### Trajectories
- $\tau$ = trajectory = a set of Random Variables over $T$ future actions and $T+1$ future states
- $p(\tau|\theta)$ = joint probability of a trajectory ($\tau$) given the parameters ($\theta$)


How do we evaluate the reward associated with that trajectory:
- $r_t = r(s_t,a_t)$ = reward = expected reward for taking action $a_t$ in state $s_t$
- $R(\tau) = \sum_{t=1}^T r(s_t,a_t)$ = Return = total reward for a trajectory $\tau$

Notice the Return ($R(\tau)$) is also a random variable because it is a function of $2T$ random variables (which make up the trajectory $\tau$).

### Objective in Reinforcement Learning

Because of this we can optimize the expected return ($\mathbb{E}[R(\tau)]$) w.r.t the parameters ($\theta$):

$$ J(\theta) = \mathbb{E}_{\tau \sim p(\tau|\theta)}[R(\tau)] $$

By ancestral sampling over the trajectory we can rewrite this as:

$$ J(\theta) = \mathbb{E}_{\tau \sim p(\tau|\theta)}\sum_{t=1}^T r(s_t, a_t) $$

Using this and discounting future returns we can define a State Value Function ($V^{\pi}(s)$) which represents the expected discounted return from a given state $s$ under a certain policy $\pi$:

$$V^{\pi}(s) = \mathbb{E}_{\tau \sim p(\tau|\theta)}\sum_{t=1}^{\infty} \gamma^{t-1} r(s_t)$$

Where $\gamma$ is the discount factor. Computing the value function is generally impractical because we often do not have the model of the environment and thus we don't know the rewards of all possible states without taking an action to actually get us there.

Instead we can use the Action Value Function ($Q^{\pi}(s,a)$) which represents the expected discounted return from a given state $s$ and action $a$ under a certain policy $\pi$:

$$Q^{\pi}(s,a) = \mathbb{E}_{\tau \sim p(\tau|\theta)}\sum_{t=1}^{\infty} \gamma^{t-1} r(s_t,a_t)$$

Notice how this relates to the value function:

$$V^{\pi}(s) = \mathbb{E}_{a \sim \pi(a|s)}[Q^{\pi}(s,a)]$$

Thus the optimal action is:

$$a^*(s) = \arg\max_{a} Q^{\pi}(s,a)$$

### Optimal Bellman Equation

The optimal policy ($\pi^*$) is the one that maximizes the expected discounted return ($V^{\pi}(s)$), and the optimal action-value function ($Q^*$) is the action-value function that for $\pi^*$

The optimal bellman equation gives a recursive relationship between the optimal action-value function and the optimal value function:

$$Q^*(s,a) = \mathbb{E}_{s' \sim p(s'|s,a)}[r(s,a) + \gamma \max_{a'} Q^*(s',a')]$$
