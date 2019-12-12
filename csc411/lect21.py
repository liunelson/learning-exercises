# %%[markdown]
# # Notes from [CSC 411](http://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/)
# --- 

# %%[markdown] 
# ## Lecture 21 - Reinforcement Learning (Part I-II)
#
# Learning problems:
# 
# 1. supervised: 
#    given inputs and some known outputs, predict remaining outputs; 
# 2. reinforcement: 
#    given inputs and no correction, choose outputs (actions) to maximize reward; 
# 3. unsupervised: 
#    given only inputs, predict outputs (e.g. clustering).
# 
# 
# Challenges of reinforcement learning (RL): 
# - continuous stream of inputs;
# - effects of an action depend on the state of the agent;
# - reward depends on the state and actions;
# - only know reward given specific action, not generally;
# - possible delay between action and reward;
# - e.g. tic-tac-toe.
# 
# 
# Markov decision process (MDP): 
# - framework to describe RL; 
# - defined by the tuple $(\mathcal{S, A, P, R}, \gamma)$; 
#    - $\mathcal{S}$, state space;
#    - $\mathcal{A} = \{a_1, \ldots, a_{|\mathcal{A}|}\}$, action space;   
#    - $\mathcal{P}$, transition probability;
#    - $\mathcal{R}$, immediate reward distribution;
#    - $\gamma \in [0, 1)$, discount factor
# - the agent has a state $S \in \mathcal{S}$ within the environment;
# - at every time step $t$, 
#    - the agent in state $S_t$;
#    - taking action $A_t$ moves it to a new state $S_{t+1} \sim \mathcal{P}(\cdot|S_t, A_t)$;
#    - some reward $R_{t+1} \sim \mathcal{R(\cdot|S_t, A_t, S_{t+1})}$ is received;
# - action selection mechanism is described by a *policy* $\pi$;
#    - $\pi$ maps states to actions;
#    - deterministic: $A_t = \pi(S_t)$
#    - stochastic: $A_t \sim \pi(\cdot|S_t)$
# - goal: find $\pi$ such that the agent's long-term reward is maximized;
# - long-term reward: 
#    - total reward, $R_0 + R_1 + R_2 + \ldots$;
#    - discounted reward, $R_0 + \gamma R_1 + \gamma^2 R_2 + \ldots$;
#    - discount factor $\gamma$ determines how near- or far-sighted the agent is;
# - transition probability, $\mathcal{P}(S_{t+1} = s^\prime|S_t = s, A_t = a)$
# - Markov property: the future depends on the past only through the current state;
# - *(state-)value function* is the expected discounted reward;
#    - for evaluating desirability of states);
#    - $V^{(\pi)}(s) = \mathrm{E}_\pi \left[ \sum\limits_{t \geq 0} \gamma^t R_t \: | \: S_0 = s \right]$
#    - (state-value function for policy $\pi$ given starting state $s$)
# - *action-value function*
#    - $Q^{(\pi)}(s, a) = \mathrm{E}_\pi \left[ \sum\limits_{t \geq 0} \gamma^t R_t \: | \: S_0 = s, A_0 = a \right]$
# - optimal value function: 
#   $Q^\ast (s, a) = \sup\limits_\pi Q^{(\pi)}(s, a)$
# - optimal policy:   
#   $\pi^\ast(s) \leftarrow \arg \max\limits_{a} Q^\ast(s, a)$;
# 
# 
# Example: tic-tac-toe
# - the game is described as  
#    - state: positions of $X$'s and $O$'s on the board;
#    - action: location of new $X, O$'s;
#    - policy: mapping states to actions;
#    - reward: win, lose, tie;
#    - value function: predict future reward based on current state;
# - since the state space is small, value function can be just a table;
# 
# 
# *Bellman Equation*: 
# - $\begin{align} Q^{(\pi)}(s, a) &= \mathrm{E}_\pi \left[ \sum\limits_{t \geq 0} \gamma^t R_t \: | \: S_0 = s, A_0 = a \right] \\ &= \mathrm{E}\left[ R(S_0,A_0) + \gamma Q^{(\pi)}(S_1, \pi(S_1)) \: | \: S_0 = s, A_0 = a \right] \\ &= r(s, a) + \gamma \int_\mathcal{S} \mathcal{P}(\mathrm{d}s^\prime | s, a) Q^{(\pi)}(s^\prime, \pi(s^\prime)) \\ &= (T^{(\pi)} Q^{(\pi)})(s, a) \end{align}$ 
# - $T$ is the *Bellman operator*;
# - $r(s,a) = \mathrm{E}[\mathcal{R}(\cdot|s, a)]$;
# - the *Bellman optimality operator*, $(T^{\ast} Q)(s, a) = r(s, a) + \gamma \int_\mathcal{S} \mathcal{P}(\mathrm{d}s^\prime | s, a) \max\limits_{a^\prime \in \mathcal{A}} Q^{(\pi)}(s^\prime, \pi(s^\prime))$;
# - note that 
#     - $\begin{align} Q^{(\pi)} &= T^{(\pi)}  Q^{(\pi)} \\ Q^\ast &= T^{\ast}  Q^{\ast} \end{align}$
#     - these are fixed-point equations;
#     - they have unique solutions;
# 
#  
# Policy evaluation:
# - given $\pi$, find $V^{(\pi)}$ or $Q^{(\pi)}$; 
# - assuming $\mathcal{P}$ and $r(s,a)$ are known;
# - if the state-action space $\mathcal{S} \times \mathcal{A}$ is small enough,
#   then solve the linear system of equations
#   $Q(s, a) = r(s,a) + \gamma \sum\limits_{s^\prime \in \mathcal{S}} \mathcal{P}(s^\prime|s,a) Q(s^\prime, \pi(s^\prime)) \: \forall \: (s,a) \in \mathcal{S} \times \mathcal{A}$ 
# - *planning* problem: find optimal $\pi$; 
# - *value iteration* (VI): $Q_{k+1} \leftarrow T^\ast Q_k$;
#
#
# $\ldots$

# %%[markdown] 
# ## Lecture 23 - Algorithmic Fairness
# 
# Notation: 
# - $X$: input to classifier;
# - $S$: sensitive feature (e.g. age, sex, etc.);
# - $Z$: latent representation;
# - $Y$: prediction;
# - $T$: true label;
# 
# 
# Criteria for fair classification:
# - demographic parity: $Y \perp S$;
# - equalized odds: $Y \perp S | T$;
# - equal opportunity: $Y \perp S | T = t$;
# - equal (weak) calibration: $T \perp S | Y$;  
# - equal (strong) calibration: $T \perp S | Y$ and $Y = \mathrm{Pr}(T = 1)$;
# - fair subgroup accuracy: $\mathbb{1}[T = Y] \perp S$; 
# - $\perp$ denotes stochastic independence;
# - many are incompatible;
# 
# 
# Fair representation: 
# - goal: find fair representation $Z$ that removes any info about sensitive feature $S$;
# - desired traits: 
#     1. retain info about $X$ (high mutual info between $X$ and $Z$);
#     2. obfuscate $S$ (low mutual info between $S$ and $Z$);  
#     3. allow high classification accuracy (high mutual info between $T$ and $Z$);

# %%
