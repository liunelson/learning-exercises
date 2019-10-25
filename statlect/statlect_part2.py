#%% [markdown]
# # Notes and exercises from [Statlect](https://www.statlect.com)
# --- 

#%% [markdown]
# # Part 2 - [Fundamentals of probability](https://www.statlect.com/fundamentals-of-probability/)

#%% [markdown]
# ## [Probability](https://www.statlect.com/fundamentals-of-probability/probability)
#
# Let $\Omega$ be the set of all outcomes.
# 
# 1. *Mutually exclusive* outcomes: only one element of $\Omega$ will happens.
# 2. *Exhaustive* outcomes: at least one element of $\Omega$ will happen.
# 
#
# If $\Omega$ has these two properties, it is known as a *sample space* or 
# the space of all possible outcomes, where: 
# - each element $\omega \in \Omega$ is a *sample point*;
# - the element that has happened is called a *realized* outcome; 
# - a subset $E \subseteq \Omega$ is called an *event* (TBD).
# 
# 
# Since every set is a subset of itself, $\Omega$ is also known as the *sure* event. 
# 
# Similarly, $\varnothing \subseteq \Omega$ is the *impossible* event.  
#
# *Space of events*: a set that is a collection of events. 
#
# e.g. Given the sample space of a 6-sided dice roll  
# $\Omega = \{1, 2, 3, 4, 5, 6\}$ with two events $E_1 = \{1, 3, 5\}$ 
# and $E_2 = \{2, 4, 6\}$, 
# a space of events is $\mathcal{F} = \{E_1, E_2, \Omega, \varnothing\}$.
#

#%% [markdown]
#
# Given a space of events (*sigma-algebra*) $\mathcal{F}$ with elements $E$ 
# on the sample space $\Omega$, 
# a function $P: \mathcal{F} \rightarrow [0, 1]$ is a *probability measure* 
# if and only if: 
#
# 1. $P(\Omega) = 1$;
#
# 2. $P\left( \bigcup\limits_{i = 1}^\infty E_i \right) = \sum\limits_{i = 1}^\infty P(E_i)$ 
# for any sequence $\{E_1, E_2, \ldots E_i, \ldots\}$ 
# of mutually exclusive events (i.e. $E_i \cap E_j = \varnothing$ if $i \neq j$).
#
#
# Property 3 is known as *countable/sigma additivity*.
#
# $P(E)$ is known as the *probability* of event $E$.
#
# Corollary: 
# - $P(\varnothing) = 0$
# - $P(E^\complement) = 1 - P(E)$
# - $P(E \cup F) = P(E) + P(F) - P(E \cap F)$ for $E,F$ not necessarily disjoint
# - $P(E) \leq P(F)$ if $E \subseteq F$ 
# (monotonicity)
# 
# Monotonicity of probability is equivalent to: 
# if an event occurs less often, then its probability is smaller)
#
#
# More rigorously, a *space of events* is required to be a *sigma-algebra*, 
# i.e. have the following properties: 
# 1. $\Omega \in \mathcal{F}$ (whole set); 
# 2. $E \in \mathcal{F} \; \Rightarrow \; E^\complement \in \mathcal{F}$ (closure under complementation);
# 3. if $\{ E_1, E_2, \ldots E_i, \ldots \}$ is a sequence of subsets of \Omega 
# belonging to $\mathcal{F}$, then $\bigcup\limits_{i = 1}^\infty E_i \in \mathcal{F}$ 
# (closure under countable union).
# 
#
# Note: A subset of $\Omega$ that belongs to the sigma-algebra $\mathcal{F}$ is called *measurable*.


#%% [markdown]
# ## [Conditional Probability](https://www.statlect.com/fundamentals-of-probability/conditional-probability)
#
# Where $\Omega$ be the sample space, $P(E)$ denotes the probability of events $E \subseteq \Omega$. 
# If there is new information — that the realized outcome will belong to some set $I \subseteq \Omega$ — 
# then we define $P(E|I)$ as the *conditional probability* of $E$ given $I$. 
# 
# Definition...
#
# Suppose a sample space with finitely many sample points 
# $\Omega = \{ \omega_1, \ldots, \omega_n \; | \; n \in \mathbb{N} \}$
# with equal probability
# $P(\{ \omega_i \}) = \frac{1}{n} \; \forall \; i \leq n$.
# 
# Thus, the probability of a generic event $E \subseteq \Omega$ is just 
# $ P(E) = \frac{\mathrm{card(E)}}{\mathrm{card}(\Omega)}$
# where $\mathrm{card(S)}$ denotes the cardinality (number of elements) 
# of a set $S$.
# 
# This is just the ratio of the number of cases favourable to $E$ to 
# the number of all possible cases.
#
# Given $I \subseteq \Omega$, the outcomes in $E \cap I^\complement$ are no longer possible 
# and thus, 
# $ P(E|I) = \frac{\mathrm{card}(E \cap I)}{\mathrm{card}(I)} = \frac{P(E \cap I)}{P(I)}$
# 
# e.g. In a 6-sided dice roll, $\Omega = \{ 1, \ldots, 6\}$;  
# for the event 'any odd number', $E = \{ 1, 3, 5 \}$; 
# given information 'a number greater than 3', $I = \{ 4, 5, 6\}$; 
# then, $P(E|I) = \frac{P(E \cap I)}{P(I)} = \frac{P(\{5\})}{P(\{4\}) + P(\{5\}) + P(\{6\})} = \frac{1/6}{1/6 + 1/6 + 1/6} = \frac{1}{3}$
#
# Problem: what happens when $I$ is a zero-probability event ($P(I) = 0$)?
# 
# Axiomatically, conditional probability $P(E|I)$ must have the following properties: 
# 
# 1. $P$ is a probability measure;
#
# 2. certainty, $P(I|I) = 1$; 
#
# 3. impossibility, 
# $P(E|I) = 0$ for any $E \subseteq I^\complement$;
# 
# 4. constant likelihood ratios on $I$, 
# $\frac{P(F|I)}{P(E|I)} = \frac{P(F)}{P(E)}$ if $E \subseteq I$, $F \subseteq I$, $P(E) > 0$.
# 
#
# Point 4 means that, given the same information, 
# the ratio of the probability of two events remains the same.
# 
# ### Law of total probability 
# 
# Let $I_1, \ldots, I_n$ be $n$ events such that 
# 
# 1. $I_i \cap I_j = \varnothing \; \forall \; i \neq j$, 
#  
# 2. $\Omega = \bigcup\limits_{i = 1}^{n} I_i$, 
#
# 3. $P(I_i) > 0$ for any i. 
# 
# Such $I_i$ is a *partition* of $\Omega$ 
# and the Law of Total Probability states that, for any event $E$, 
# 
# $P(E) = \sum\limits_{i = 1}^n P(E|I_i) P(I_i)$
# 

#%% [markdown]
# ## [Independent Events](https://www.statlect.com/fundamentals-of-probability/independent-events)
#
# Two events $E, F$ are *independent* 
# if and only if $P(E \cap F) = P(E)P(F)$.
# 
# i.e. Two events are independent if the occurrence of either makes 
# neither more nor less probable.
#
# Corollary: $P(F|E) = P(F)$ and $P(E|F) = P(E)$.
# 
# Let $E_1, \ldots, E_n$ be $n$ events. 
# They are *jointly/mutually* independent if and only if 
# 
# $P\left( \bigcap\limits_{j = 1}^k E_{i_j} \right) = \prod\limits_{j = 1}^k P(E_{i_j})$
# 
# for any sub-collection of $k \leq n$ events $E_{i_1}, \ldots, E_{i_k}$.
# 
# Note: 
# Even if $E_i$ is independent of $E_j$ for any $i \neq j$, 
# they are *not* altogether jointly independent.
# However, the *converse is true*.
#
# ### Example
# 
# Consider an urn with 4 labeled balls $B_1, B_2, B_3, B_4$ 
# which are drawn at random.
# 
# Define 3 events: 
# $\begin{align} E = \{B_1,B_2\} \\ E = \{B_2,B_3\} \\ E = \{B_2,B_4\}\end{align}$
# 
# It follows that all the pairs of events are independent: 
#
# $\begin{align} P(E \cap F) = P(E)P(F) \\ P(E \cap G) = P(E)P(G) \\ P(F \cap G) = P(F)P(G) \end{align}$
# 
# However, they are not jointly independent; in fact: 
# 
# $P(E \cap F \cap G) \neq P(E)P(F)P(G)$
# 

#%% [markdown]
# ## [Zero-Probability Events](https://www.statlect.com/fundamentals-of-probability/zero-probability-events)
#
# An event $E$ is a *zero-probability event* if and only if 
# $P(E) = 0$.
# 
# Zero-probability events are *not* impossible events ($\varnothing$); 
# they can happen all the time in models where the sample space $\Omega$ 
# is not countable.
# 
# ### Almost Sure and Almost Surely
# 
# Let $\Phi$ be some property that a sample point $\omega \in \Omega$ 
# can either satisfy or not. 
# Let $F = \{ \omega \in \Omega \; : \; \omega \mathrm{satisfies} \Phi\}$.
# 
# Property $\Phi$ is said to be *almost sure* 
# (i.e. holds either *almost surely*)
# if there exists an event $E$ 
# such that $P(E) = 0$ and $F^\complement \subseteq E$.
# 
# From the monotonicity and complement properties: 
# - $F^\complement \subseteq E \; \Rightarrow \; 0 \leq P(F^\complement) \leq P(E) = 0$.
# - $P(F) = 1 - P(F^\complement) = 1 - 0 = 1$
# - An almost-sure event is one that happens with probability $1$.
# 
#
# Example: 
#
# 1. Consider a sample space $\Omega = [0, 1]$ 
# with probabilities $P([a, b]) = b- a$ where $[a, b] \subseteq [0, 1]$.
# 
# 2. Consider the event $E = \{ \omega \in \Omega \; | \; \omega \in \mathbb{G} \}$.
#  
# 3. Since $E$ is a countable set, 
# $E = \bigcup\limits_{i = 1}^\infty \{\omega_i\} \; \Rightarrow \; P(E) = \sum\limits_{i = 1}^\infty P(\{\omega_i\}) = 0$.
# 
# 4. Thus, $E$ is a zero-probability event.
# 
# 5. Corollary: the irrational set in $\Omega$ is an almost-sure event  
# since $F = E^\complement \; \Rightarrow \; P(F) = P(E^\complement) = 1 - P(E) = 1$.
# 

#%% [markdown]
# ## [Bayes' Rule](https://www.statlect.com/fundamentals-of-probability/Bayes-rule)
#
# Thomas Bayes' Rule:
# 
# - Let $A$ and $B$ be two events with probabilities $P(A) > 0$ and $P(B) > 0$.   
# - Proposition: $P(A|B) = \frac{P(B|A) P(B)}{P(B)}$
# 
#
# ### Example
# 
# - Given a defect-detecting robot.
# - If an item is defective, the robot detects it with 98% probability. 
# - If not defective, the robot will signal any defect with 99% probability.
# - At random, 0.1% of items are defective.
# - What is the probability that an item signaled as defective to be actually defective?
# 
#
# - Let $\mathrm{RD}$ mean that the robot signals defect,  
# $\mathrm{ID}$ mean that the item is defective, $\mathrm{IN}$ otherwise.
# - $\begin{align} P(\mathrm{RD}|\mathrm{ID}) &= 0.98 \\ P(\mathrm{RD}|\mathrm{IN}) &= 1 - 0.99 = 0.01 \\ P(\mathrm{ID}) &= 0.001 \\ P(\mathrm{IN}) &= 1 - 0.001 = 0.999 \\ P(\mathrm{ID}|\mathrm{RD}) &= ?\end{align}$
# - Note that $\mathrm{ID} \cap \mathrm{IN} = \varnothing$.
# - From the Law of Total Probability: 
# $\begin{align} P(\mathrm{RD}) &= P(\mathrm{RD}|\mathrm{ID})P(\mathrm{ID}) + P(\mathrm{RD}|\mathrm{IN})P(\mathrm{IN}) \\ &= 0.98 \times 0.001 + 0.01 \times 0.999 \\ &= 0.01097\end{align}$
# - Thus from Bayes' rule: 
# $\begin{align} P(\mathrm{ID}|\mathrm{RD}) &= \frac{P(\mathrm{RD}|\mathrm{ID}) P(\mathrm{ID})}{P(\mathrm{RD})} \\ &= \frac{0.98 \times 0.001}{0.01097} \\ &\approx 0.08933 \end{align}$
# - The robot is conditionally accurate (98%) but it isn't right very often (9%)!
# 
# 
# Definitions: 
# - $P(A)$ = *prior* probability;
# - $P(B|A)$ = *conditional* probability or *likelihood*;
# - $P(B)$ = *marginal* probability;
# - $P(A|B)$ = *posterior* probability.
# 
#   
# ### Example 
# 
# - Given two urns. 
# - The 1st urn has 50 red and 50 blue balls.
# - The 2nd urn has 30 red and 70 blue balls.
# - Probability of urn choosing is 50-50.
# - If a red ball is chosen, what is the probability that the 1st urn was chosen?
# 
#
# - Denoted urn choices by $U_1$ and $U_2$; ball choices by $R$ and $B$.
# - $P(U_1) = P(U_2) = 0.5$
# - $P(R|U_1) = 0.5$ and $P(B|U_1) = 0.5$
# - $P(R|U_2) = 0.3$ and $P(B|U_2) = 0.7$
# - Question becomes $P(U_1|R) = ?$
# 
# - $\begin{align} P(R) &= P(R|U_1) P(U_1) + P(R|U_2) P(U_2) \\ &= 0.5 \times 0.5 + 0.3 \times 0.5 \\ &= 0.4\end{align}$
# - $\begin{align} P(U_1|R) &= \frac{P(R|U_1) P(U_1)}{P(R)} \\ &= \frac{0.5 \times 0.5}{0.4} \\ &= 0.625 \end{align}$
# - If a red ball is chosen, there is 62.5% probability that the 1st urn was chosen.

#%% [markdown]
# ## [Random Variables](https://www.statlect.com/fundamentals-of-probability/random-variables)
#
# Definition: 
# A *random variable* $X$ is a function that maps the sample space $\Omega$ to $\mathbb{R}$, 
# i.e. $X: \Omega \rightarrow \mathbb{R}$.
# 
# Notation:
# - Formally, $X$ depends on $\omega$ but it is often omitted.
# - Given $A \subseteq \mathbb{R}$, $P_X(A) = P(X \in A) = P(\{ \omega \in \Omega \; : \; X(\omega) \in A \})$
# - $P_X$ means a probability measure on the set of $\mathbb{R}$ induced by random variable $X$.
# 
# Example: 
# - Consider a coin flip, $T$ = tail and $H$ = head.
# - Assuming fair, $P(\{T\}) = P(\{H\}) = 0.5$.
# - A random variable may be the dollar-value winning: $X(\omega) = \begin{cases} 1, &\; \omega = T \\ -1, &\; \omega = H \end{cases}$
# - Then, $\begin{align} P_X(1) = P(X = 1) = P(\{T\}) = 0.5 \\ P_X(-2) = P(X = -2) = P(\varnothing) = 0 \end{align}$  
# 
# 
# ### Discrete Random Variables
#
# Discrete versus continuous.
# 
# Definition: A random variable $X$ is *discrete* if 
#
# 1. its support $R_X$ is a countable set; 
#
# 2. there is a function $p_X: \mathbb{R} \rightarrow [0, 1]$ (*probability mass function*) of $X$ such that, $\forall \; x \in \mathbb{R}$, 
# $p_X(x) = \begin{cases} P(X = x) &\; x \in R_X \\ 0 &\; x \notin R_X \end{cases}$
# 
# 
# Two properties of a PMF:
# 
# 1. Non-negativity, $p_X(x) \geq 0 \; \forall \; x \in \mathbb{R}$.
# 2. Normalization, $\sum\limits_{x \in R_X} p_X(x) = 1$.
#
# 
# Example: *Bernoulli random variable* 
# - Only has two values ($1$ and $0$) with probability $q$ and $1 - q$ respectively.
# - Its support is $R_X = \{0, 1\}$.
# - Its probability mass function is $p_X(x) = \begin{cases} q & \textrm{if } x = 1 \\ 1-q & \textrm{if } x = 0 \\ 0 & \textrm{otherwise} \end{cases}$.
# 
#
# Definition: A random variable is *continuous* if 
# 
# 1. its support $R_X$ is uncountable;
# 
# 2. there is a function $f_X: \mathbb{R} \rightarrow [0, \infty)$ (*probability density function* or PDF) 
# of $X$ such that, $\forall \; [a, b] \subseteq \mathbb{R}$, 
# $P_X([a, b]) = P(X \in [a, b]) = \int_a^b f_X(x) \mathrm{d}x $
# 
#
# Example: *Uniform random variable* (on the interval $[0, 1]$) 
# - It can take any value in the interval $[0, 1]$. 
# - All sub-intervals of equal length are equally likely.
# - Its support is $R_X = [0, 1]$.
# - Its PDF is $f_X(x) = \begin{cases} 1 & \textrm{if } x \in [0, 1] \\ 0 & \textrm{otherwise} \end{cases}$.
# - Thus, $P(X \in [0.25, 0.75]) = \int_{0.25}^{0.75} f_X(x) \mathrm{d}x = 0.5$.
# 
# 
# In general, random variables $X$ of neither sort are often characterized as follows:
# - *Cumulative distribution function* (CDF) of $X$ is a function $F_X: \mathbb{R} \rightarrow [0,1]$ such that 
# $F_X(x) = P(X \leq x) \; \forall \; x \in \mathbb{R}$.
#
# - The probability that $X$ belongs to an interval $[a,b] \subseteq \mathbb{R}$ as 
# $P(a < X \leq b) = F_X(b) - F_X(b)$.
# 
#
# If $X$ is continuous, 
# 
# - then $F_X(x) = \int_{-\infty}^\infty f_X(t) \mathrm{d}t$ and 
# $f_X(x) = \frac{\mathrm{d} F_X(x)}{\mathrm{d}x}$.
# 
# - then the event $\{ \omega : X(\omega) = x\}$, where X takes on a specific value $x$, 
# is a zero-probability event, $P(X = x) = 0$.
# 

#%% [markdown]
# ## [Random Vectors](https://www.statlect.com/fundamentals-of-probability/expected-value)
# 
# A *random vector* is a function from the sample space $\Omega$ to the set of $K$-dimensional real vectors $\mathbb{R}^K$.
# 
# The real vector $\boldsymbol{X}(\omega)$ is called a *realization* of the random vector.
# 
# The set of all possible realizations is called the *support* of $\boldsymbol{X}$ and is denoted by $R_X$.
# 
# Example:
# 
# - Consider a two-coin toss; tail = $T$ and head = $H$.
# - The sample space is $\Omega = \{TT, TH, HT, HH \}$.
# - All four outcomes have equal probability of $1/4$.
# - Take $\boldsymbol{X}$ as the winnings on each toss (1 if $T$, 0 otherwise):
# $X(\omega) = \begin{cases} [1 1] & \textrm{if } \omega = TT \\ [1 -1] & \textrm{if } \omega = TH \\ [-1 1] & \textrm{if } \omega = HT \\ [-1 -1] & \textrm{if } \omega = HH \end{cases}$
# 
# - The probability of winning 1 on both tosses is: 
# $P(\boldsymbol{X} = [1 1]) = P(\{ \omega \in \Omega \: : \; \boldsymbol{X} = [1 1]\}) = P(\{TT\}) = \frac{1}{4}$  
# - The probability of losing 1 on the second toss is: 
# $P(X_2 = -1) = P(\{TH, HH\}) = P(\{TH\}) + P(\{HH\}) = \frac{1}{2}$
# 
# 
# As before, a random vector $\boldsymbol{X}$ is discrete if and only if:  
# - its support $R_X$ is a countable set;
# - there is a function $p_X: \mathbb{R}^K \rightarrow [0, 1]$ called the *joint* PMF of $\boldsymbol{X}$ 
# such that for any $\boldsymbol{x} \in \mathbb{R}^K$: 
# $p_X(\boldsymbol{x}) = \begin{cases} P(\boldsymbol{X} = \boldsymbol{x}) & \textrm{if } \boldsymbol{x} \in R_X \\ 0 & \textrm{otherwise} \end{cases}$
# 
#
# Notation: $p_X(\boldsymbol{x}) = p_X(x_1, \ldots, x_K) = p_{X_1, \ldots, X_K}(x_1, \ldots, x_K)$
# 
# Equivalently, for a *continuous random vector*:
# - its support $R_X$ is an uncountable set;
# - there is a function $f_X: \mathbb{R}^K \rightarrow [0, \infty)$ called the joint PDF of $\boldsymbol{X}$ 
# such that for any set $A \subseteq \mathbb{R}^K$, where $A = [a_1, b_1] \times \ldots \times [a_K, b_K]$, 
# the probability that $\boldsymbol{X}$ belongs to $A$ is 
# $P(\boldsymbol{X} \in A) = \int_{a_1}^{b_1}\ldots\int_{a_K}^{b_K} f_X(x_1, \ldots, x_K) \mathrm{d}x_1 \ldots \mathrm{d}x_K$
# 
# 
# *Marginal* CDF of $X_i$ is the CDF of $i$-th component of $\boldsymbol{X}$.
# It can be derived from the joint CDF as follows: 
# - $p_{X_i}(a) = \sum\limits_{(x_1, \ldots, x_K) \in R_X \; : \; x_i = a } p_X(x_1, \ldots, x_K)$
# - the sum is over the set $\{ (x_1, \ldots, x_K) \in R_X \; : \; x_i = a \}$
# 
# 
# As before, if $\boldsymbol{X}$ is continuous, 
# $F_X(\boldsymbol{x}) = \int_{-\infty}^{x_1} \ldots \int_{-\infty}^{x_K} f_X(t_1, \ldots, t_K) \mathrm{d}t_1 \ldots \mathrm{d}t_K$
# 
# Similarly, $\frac{\partial^K F_X(\boldsymbol{x})}{\partial x_1 \ldots \partial x_K} = f_X(\boldsymbol{x})$
#      


#%% [markdown]
# ## [Expected Value](https://www.statlect.com/fundamentals-of-probability/expected-value)
#
# The *expected value* (or *expectation*) of a random value $X$ is denoted $E[X]$ and 
# is defined as: 
# $\mathrm{E}[X] = \begin{cases} \sum\limits_{x \in R_X} x \: p_X(x) & \textrm{if discrete} \\ \int\limits_{-\infty}^\infty x \: f_X(x) \mathrm{d}x & \textrm{if continuous} \end{cases}$
# 
# This definition requires that *absolute summability/integrability*:
# either $\sum\limits_{x \in R_X} |x| \: p_X(x)$ 
# or $\int\limits_{-\infty}^\infty |x| \: f_X(x) \mathrm{d}x$ 
# to be $< \infty$. Otherwise, $\mathrm{E}[X]$ is said to be not well-defined or does not exist.
# 
# Properties: 
# - $\mathrm{E}[a + bX] = a + b \mathrm{E}[X]$
# - $\mathrm{E}[X + Y] = \mathrm{E}[X] + \mathrm{E}[Y]$
# - Expected value of a random vector/matrix is done entry-wise.
# - If $X, Y$ are integrable and $X \leq Y$ almost surely, 
# then $\mathrm{E}[X] \leq \mathrm{E}[Y]$.

#%% [markdown]
# ## [Variance](https://www.statlect.com/fundamentals-of-probability/variance)
# 
# Definition: 
# The *variance* of a random variable $X$ is denoted by $\mathrm{var}[X]$ and is defined as 
# $\mathrm{var}[X] = \mathrm{E}[ (X - \mathrm{E}[X])^2 ] = \mathrm{E}[X^2] - \mathrm{E}[X]^2$
# 
# The *standard deviation* is just the square root of the variance, denoted $\mathrm{std}[X]$.
# 

#%% [markdown]
# ## [Transformation Theorem](https://www.statlect.com/glossary/transformation-theorem)
#
# This theorem allows the expected value of a function of a random variable to be calculated 
# without knowing the probability distribution of the function itself.
#
# For discrete variables: 
# 
# Consider the function $g: \mathbb{R} \rightarrow \mathbb{R}$. 
# Define $Y = g(X)$. 
# Then, $\mathrm{E}[Y] = \mathrm{E}[g(X)] = \sum\limits_{x \in R_X} g(x) \: p_X(x)$, 
# where $p_X(x)$ is the PMF of $X$.
#
# The standard formula would have stated $\mathrm{E}[Y] = \sum\limits_{y \in R_Y} y \: p_Y(y)$
#
# Similarly for continuous random variable, 
# $\mathrm{E}[Y] = \mathrm{E}[g(X)] = \int_{-\infty}^\infty g(x) f_X(x) \mathrm{d}x$.


#%% [markdown]
# ## [Covariance](https://www.statlect.com/fundamentals-of-probability/covariance)
#
# The *covariance* between two random values $X, Y$ is denoted $\mathrm{cov}[X,Y]$ and 
# is defined as: 
# $\mathrm{cov}[X,Y] = \mathrm{E}[(X - \mathrm{E}[X])(Y - \mathrm{E}[Y])] = \mathrm{E}[XY] - \mathrm{E}[X]\mathrm{E}[Y]$.
# 
# Example: 
# - Let $\boldsymbol{X}$ be a $2 \times 1$ discrete random vector with $X_1, X_2$.
# - Its support is $R_X = \{ [1 \: 1], [2 \: 0], [0 \: 0] \}$.
# - The joint PMF is $p_X(\boldsymbol{x}) = \begin{cases} \frac{1}{3} & \boldsymbol{x} = [1 \: 1] \\ \frac{1}{3} & \boldsymbol{x} = [2 \: 0] \\ \frac{1}{3} & \boldsymbol{x} = [0 \: 0] \\ 0 & \textrm{otherwise} \end{cases}$
# - What is $\mathrm{cov}[X_1, X_2]$?
# 
# - $R_{X_1} = \{ 0, 1, 2\}$
# - Marginal PMF of $X_1$
# - $\mathrm{E}[X_1] = \sum\limits_{x \in R_{X_1}} x \: p_{X_1}(x) = 1$
# - etc.
#
# 
# Properties: 
# - $\mathrm{cov}[X,X] = \mathrm{var}[X]$
# - $\mathrm{cov}[X,Y] = \mathrm{cov}[Y,X]$
# - $\mathrm{var}[X + Y] = \mathrm{var}[X] + \mathrm{var}[Y] + 2 \mathrm{cov}[Y,X]$
# - $\mathrm{cov}[a_1 X_1 + a_2 X_2, Y] = a_1 \mathrm{cov}[X_1, Y] + a_2 \mathrm{cov}[X_2, Y]$
# 

#%% [markdown]
# ## [Linear Correlation](https://www.statlect.com/fundamentals-of-probability/linear-correlation)
# 
# Definition: 
# The *linear/Pearson correlation coefficient* of two random variables $X, Y$ is denoted by $\mathrm{corr}[X,Y]$ and is defined as 
# $\mathrm{corr}[X] = \frac{\mathrm{cov}[X,Y]}{\mathrm{std}[X] \mathrm{std}[Y]}$
# 
# Property: $\mathrm{corr}[X,Y] \in [-1, 1]$
# 

#%% [markdown]
# ## [Covariance Matrix](https://www.statlect.com/fundamentals-of-probability/covariance-matrix)
# 
# Definition: 
# - Let $\boldsymbol{X}$ be a $K \times 1$ random vector.
# - Its *covariance matrix* is just the multivariate generalization of $variance$: 
# $\mathrm{var}[\boldsymbol{X}] = \mathrm{E}[(\boldsymbol{X} - \mathrm{E}[\boldsymbol{X}]) (\boldsymbol{X} - \mathrm{E}[\boldsymbol{X}])^\mathsf{T}]$
# - Alternatively, 
# $\mathrm{var}[\boldsymbol{X}] = \mathrm{E}[\boldsymbol{X} \boldsymbol{X}^\mathsf{T}] - \mathrm{E}[\boldsymbol{X}] \: \mathrm{E}[\boldsymbol{X}]^\mathsf{T}$
#
#
# The $(i,j)$-th matrix entry is equal to $\mathrm{cov}[X_i, X_j]$.
# 
# The *cross-covariance* matrix is similarly defined: 
# $\mathrm{cov}[\boldsymbol{X},\boldsymbol{Y}] = \mathrm{E}[(\boldsymbol{X} - \mathrm{E}[\boldsymbol{X}]) (\boldsymbol{Y} - \mathrm{E}[\boldsymbol{Y}])^\mathsf{T}]$
# 
# Property: 
# - $\mathrm{cov}[aX, bX] = a\mathrm{var}[X]b^\mathsf{T}$
# - $\mathrm{cov}[X, Y] = \mathrm{cov}[Y, X]^\mathsf{T}$
#

#%% [markdown]
# ## [Indicator Functions](https://www.statlect.com/fundamentals-of-probability/indicator-functions)
# 
# Definition: 
# - Let $\Omega$ be a a sample space and $E \subseteq \Omega$ be an event.
# - The *indicator* function of $E$ is denoted by $1_E$ or $\chi_E$  
# - it is a random variable defined as  
# $1_E = \begin{cases} 1 & \omega \in E \\ 0 & \textrm{otherwise} \end{cases}$

#%% [markdown]
# ## [Quantile](https://www.statlect.com/fundamentals-of-probability/quantile)
# 
# Recall the definition of the CDF (cumulative distribution function) of some random variable $X$: 
# 
# $F_X(x) = P(X \leq x)$
# 
# Definition of *quantile*: 
# - Let $p \in (0, 1)$.
# - The $p$-quantile of $X$ is denoted $Q_X(p)$.   
# - It is defined as 
# $Q_X(p) = \mathrm{inf}\{ x \in \mathbb{R} \; : \; F_X(x) \geq p\}$
# - $\mathrm{inf} = the *infimum* of a subset $S$ of a set $T$ is 
# the greatest element of $T$ that is less than or equal to all elements of $S$.
# - if the CDF is continuous, strictly increasing on $\mathbb{R}$, and invertible, 
# $Q_X(p) = F_X^{-1}(p)$.
# 
# 
# Special quantiles: 
# - $Q_X(p = \frac{1}{2})$ = *median*
# - $Q_X(p = \frac{i}{4})$ = $i$-th *quartile*
# - $Q_X(p = \frac{i}{10})$ = $i$-th *decile*
# - $Q_X(p = \frac{i}{100})$ = $i$-th *percentile*
# 

#%% [markdown]
# ## [Conditional Expectation](https://www.statlect.com/fundamentals-of-probability/conditional-expectation)
# 
# Definition: 
# - Let $X, Y$ be two random variables.
# - The *conditional expectation* of $X$ given $Y = y$ is denoted 
# $\mathrm{E}[X|Y = y]$.
# - It is as the expectation of $X$ where the weights are the condtional probabilities. 
# - For a discrete random variable (assuming absolute summability): 
# $\mathrm{E}[X|Y = y] = \sum\limits_{x \subseteq R_X} x \: p_{X|Y=y}(x)$
# - For a continuous random variable (assuming absolute integrability): 
# $\mathrm{E}[X|Y = y] = \int\limits_{-\infty}^\infty x \: f_{X|Y=y}(x) \mathrm{d}x$
# 
#
# Example: 
# - Consider the random vector [X \: Y] with support $R_{XY} = \{ [1 \: 3], [2 \: 0], [0 \: 0]\}$.
# - The joint PMF is $p_{XY}(x, y) = \begin{cases} \frac{1}{3} & x = 1, y= 3 \\ \frac{1}{3} & x = 2, y = 0 \\ \frac{1}{3} & x = 0, y = 0 \\ 0 & \textrm{otherwise} \end{cases}$
# 
# - Let's compute the conditional PMF of $X$ given $Y = 0$.
# - The marginal PMF of $Y$ at $y = 0$ is $\begin{align} p_Y(0) &= \sum\limits_{\{ (x,y) \in R_{XY} \; : \; y = 0 \}} p_{XY}(x, y) \\ &= p_{XY}(2, 0) + p_{XY}(0, 0) \\ &= \frac{2}{3} \end{align}$
# - The support of $X$ is $R_X = \{0, 1, 2\}$.
# - Thus, conditional PMF of $X$ given $Y = 0$ is 
# $p_{X|Y=0}(x) = \begin{cases} \frac{p_{XY}(0,0)}{p_Y(0)} = \frac{1}{2} & x = 0 \\ 0 & x = 1 \\ \frac{1}{2} & x = 2 \\ 0 & x \neq R_X \end{cases}$
# - $\mathrm{E}[X|Y=0] = \sum\limits_{x \in R_X} x \: p_{X|Y=0}(x) = 1$
# 
# 
# Law of iterated expectations: $\mathrm{E}[\mathrm{E}[X|Y]] = \mathrm{E}[X]$
# 
# Conditional variance can be similarly defined: 
# $\mathrm{var}[X|Y=y] = \mathrm{E}[X^2|Y=y] - \mathrm{E}[X|Y=y]^2$
# 

#%% [markdown]
# ## Inequalities
# 
# Markov's inequality:
# - Upper bound to the probability that 
# the realization of a random variable $X$ (on sample space $\Omega$) exceeds a given threshold. 
# - Let $X(\omega$) \geq 0 \; \forall \; \omega \in \Omega$.
# - Let $c \in \mathbb{R}_{++}$ (strictly positive).
# - Then, $P(X \geq c) \leq \frac{\mathrm{E}[X]}{c}$ 
# 
# 
# Chebyshev's inequality: 
# - Upper bound to the probability that 
# the absolute deviation of a random variable $X$ from its mean exceeds a given threshold.
# - Let $X$ be a random variable with finite mean $\mu$ and finite variance $\sigma^2$.
# - Let $c \in R_{++}$.
# - Then, $P(|X - \mu| \geq c) \leq \frac{\sigma^2}{c^2}$
# 
# 
# Example:
# - If a population had an average income of \$40k. 
# What is the probability for someone to have an income $\ged$ \$200k?
# 
# $P(X \geq 200,000) \leq \frac{40,000}{200,000} \approx 33\%$
# 
# - If the income distribution has a standard deviation of \$20k. 
# What is the probability for someone to have an income $\leq$ \$10k or $\geq$ \$70k?
# 
# $P(|X - 40,000| \geq 30,000) \leq \frac{20,000^2}{30,000^2} \approx 20\%$
# 
# 
# Jensen's inequality:
# - Let $X$ be an integrable random variable 
# and $g: \mathbb{R}\rightarrow\mathbb{R}$ be a convex function such that $Y = g(X)$ is also integrable.
# - Thus, $\mathrm{E}[g(X)] \geq g(\mathrm{E}[X])$
# - If $g$ is concave, $\mathrm{E}[g(X)] \leq g(\mathrm{E}[X])$
# 
# 
# Example: 
# - Given a strictly positive random variable $X$ with $\mathrm{E}[X] = 1$ 
# but not constant with probability one. What do we know about $\ln(X)$?
# - Since $\ln(x)$ is strictly concave, $\mathrm{E}[\ln(X)] < \ln(\mathrm{E}[X]) = 0$
# 

#%% [markdown]
# ## [Factorization of Joint Probability Mass Functions](https://www.statlect.com/fundamentals-of-probability/factorization-of-joint-probability-mass-functions)
# 
# Given two discrete random variables $X, Y$ (or random vectors), 
# their joint PMF $p_{XY}(x, y)$ can be factorized into: 
# 1. a conditional PMF of $X$ given $Y = y$, $p_{X|Y=y}(x)$
# 2. the marginal PMF of $Y$, $p_Y(y)$
# 
# 
# Method: 
# 1. Marginalize $p_{XY}(x,y)$ by summing it over all possible values of $x$ to obtain $p_Y(y)$.
# 2. If $p_Y(y) > 0$, $p_{X|Y=y} = frac{p_{XY}(x, y)}{p_Y(y)}$.
#  
# 
# If Step 1 is too hard, guess two functions $g(x, y)$ and $h(y)$ such that $g(x, y)$ is PMF; 
# then, $p_{X|Y=y}(x) = g(x,y)$ and $p_Y(y) = h(y)$.
# 

#%% [markdown]
# ## [Factorization of Joint Probability Density Functions](https://www.statlect.com/fundamentals-of-probability/factorization-of-joint-probability-density-functions)
# 
# Given two continuous random variables $X, Y$ (or random vectors), 
# their joint PDF $f_{XY}(x, y)$ can be factorized into: 
# 1. a conditional PDF of $X$ given $Y = y$, $f_{X|Y=y}(x)$
# 2. the marginal PDF of $Y$, $f_Y(y)$
# 
# 
# Method: 
# 1. Marginalize $p_{XY}(x,y)$ by integrating it with respect to $x$ to obtain $f_Y(y)$.
# 2. If $f_Y(y) > 0$, $f_{X|Y=y} = frac{f_{XY}(x, y)}{f_Y(y)}$.
#  
# 
# If Step 1 is too hard, guess two functions $g(x, y)$ and $h(y)$ such that $g(x, y)$ is PMF; 
# then, $f_{X|Y=y}(x) = g(x,y)$ and $f_Y(y) = h(y)$.
# 

#%% [markdown]
# ## [Sums of Independent Random Variables](https://www.statlect.com/fundamentals-of-probability/sums-of-independent-random-variables)
# 
# Let $Z = X + Y$ where $X Y$ are two independent random variable 
# with CDFs $F_X(x), F_Y(y)$. 
# 
# Then, it holds that 
# - $F_Z(z) = \mathrm{E}[F_X(z - Y)]$
# - $F_Z(z) = \mathrm{E}[F_X(z - X)]$
# 
#  
# Example:
# - Let $X$ have the PDF $f_X(x) = \begin{cases} 1 & x \in R_X \\ 0 & \textrm{otherwise} \end{cases}$ 
# and $Y$ similarly. 
# - $F_X(x) = \int_{-\infty}^x f_x(t) \mathrm{d}t = \begin{cases} 0 & x \leq 0 \\ x & 0 < x \leq 1 \\ 1 & x > 1 \end{cases}$
# - $\begin{align} F_Z(z) &= \mathrm{E}[F_X(z - Y)] \\ &= \int_{-\infty}^\infty F_x(z - y) f_Y(y) \mathrm{d}y \\ &= \int_0^1 F_X(z - y) \mathrm{d}y \\ &= \int_{z-1}^z F_X(t) \mathrm{d}t \end{align}$ 
# - if $z \leq 0$, then $F_Z(z) = 0$
# - if $0 < z \leq 1$, then $F_Z(z) = \frac{1}{2}z^2$
# - if $1 < z \leq 2$, then $F_Z(z) = -\frac{1}{2}z^2 + 2z - 1$
# - if $z > 2$, then $F_Z(z) = 1$
 
#%% 

import numpy as np
import matplotlib.pyplot as plt

t = np.arange(-1, 3, 0.01)
F_X = np.zeros(t.shape)
for i in range(len(t)):
    if t[i] <= 0:
        F_X[i] = 0
    elif (t[i] > 0) & (t[i] <= 1):
        F_X[i] = t[i]
    else:
        F_X[i] = 1

F_Z = np.zeros(t.shape)
for i in range(len(t)):
    if t[i] <= 0:
        F_Z[i] = 0
    elif (t[i] > 0) & (t[i] <= 1):
        F_Z[i] = 0.5*t[i]**2
    elif (t[i] > 1) & (t[i] <= 2):
        F_Z[i] = -0.5*t[i]**2 + 2*t[i] - 1
    else:
        F_Z[i] = 1

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4, 3))
ax.plot(t, F_X, label = 'For X and Y')
ax.plot(t, F_Z, label = 'For Z = Z + Y')
ax.set_xlabel('x, y, z')
ax.set_ylabel('Cumulative Distribution Functions')
ax.legend()

#%% [markdown]
# 
# PMF of a discrete sum: 
# $p_Z(z) = p_X(z) \ast p_Y(z) = \sum\limits_{y \in R_Y} p_X(z - y)\: p_Y(y)$
# 
# PDF of a continuous sum: 
# $f_Z(z) = f_X(z) \ast f_Y(z) = \int_{-\infty}^\infty f_X(z - y)\: f_Y(y) \mathrm{d}y$
# 
# Example: 
# - Let $X$ with $R_X = [0, \infty)$ have the PDF $f_X(x) = \begin{cases} \mathrm{e}^{-x} & x \in R_X \\ 0 & \textrm{otherwise} \end{cases}$
# - Similarly for $Y$.
# - For $Z = X + Y$, $R_Z = [0, \infty)$.
# - When $z \in R_Z$, $f_Z(z) = \int_{-\infty}^\infty f_X(z - y)\: f_Y(y) \mathrm{d}y = z \: \mathrm{e}^{-z}$

# %%
t = np.arange(-1, 5, 0.01)
f_X = np.exp(-t)
f_X[t <= 0] = 0
f_Z = np.multiply(t, np.exp(-t))
f_Z[t <= 0] = 0

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4, 3))
ax.plot(t, f_X, label = 'For X and Y')
ax.plot(t, f_Z, label = 'For Z = Z + Y')
ax.set_xlabel('x, y, z')
ax.set_ylabel('Probability Density Functions')
ax.legend()

#%% [markdown]
#
# For $n$ independent random variables, 
# calculate iteratively:
# - $Z = \sum_{i = 1}^n X_i = Y_n = Y_{n-1} + X_n$

#%% [markdown]
# ## [Moments](https://www.statlect.com/fundamentals-of-probability/moments)
# 
# The $n$-th *moment* of a random variable, denoted by $\mu_X(n)$, 
# is the expected value of its $n$-th power: 
# $\mu_X(n) = \mathrm{E}[X^n]$
# 
# The $n$-th *central moments*, denoted by $\bar{\mu}_X{n}$, 
# is simiarly defined:  
# $\bar{\mu}_X{n} = \mathrm{E}[(X -  \mathrm{E}[X])^n]$

#%% [markdown]
# ## [Cross-Moments](https://www.statlect.com/fundamentals-of-probability/cross-moments)
# 
# *Cross-moments* are generalization of moments of a random variable 
# to a random vector.
# 
# Definition: 
# - Let $\boldsymbol{X}$ be a $K \times 1$ random vector.
# - $X_i$ is the $i$-th entry of $X$.
# - Let $n_1, \ldots, n_K \in \mathbb{Z}_+$ and $n = \sum_{k = 1}^K n_k$. 
# - The cross-moment of $X$ of order $n$ is defined as: 
# $\mu_X(n_1, \ldots, n_K) = \mathrm{E}[X_1^{n_1} \: \ldots \: X_K^{n_K}]$
# 
# A *central cross-moment* of $\boldsymbol{X}$ can be defined similarly:
# $\bar{\mu}_X(n_1, \ldots, n_K) = \mathrm{E}[(X_1 - \mathrm{E}[X_1])^{n_1} \: \ldots \: (X_K - \mathrm{E}[X_K])^{n_K}]$
 

#%% [markdown]
# ## [Moment Generating Functions](https://www.statlect.com/fundamentals-of-probability/moment-generating-function)
# 
# The probability distribution of (some) random variable 
# is uniquely determined by its *moment generating functions* (MGF), 
# real functions whose derivatives at zero are equal to 
# the moments of the random variable.
# 
# Definition:
# - Let $X$ be a random variable.
# - If $\mathrm{E}[\mathrm{e}^{tX}]$ exists and is finite for all real $t$ 
# then it is called the MGF of $X$, denoted by $M_X(t)$.
# 
# 
# Example: 
# - Consider $X$ with support $R_X = [0, \infty)$ 
# and PDF $f_X(x) = \begin{cases} \lambda \mathrm{e}^{-\lambda x} & x \in R_x \\ 0 & \textrm{otherwise} \end{cases}$
# - $\begin{align} \mathrm{E}[\mathrm{e}^{tX}] &= \int_{-\infty}^\infty \mathrm{e}^{tX} f_X(x) \mathrm{d}x \\ &= \ldots \\ &= \frac{\lambda}{\lambda - t} \end{align}$
# - Since this expected value exists and is finite for any $t \in [-h, h], \: 0 < h < \lambda$, 
#  $M_X(t) = \frac{\lambda}{\lambda - t}$
# 
# 
# Properties: 
# - $\mu_X(n) = \mathrm{E}[X^n] = \frac{\mathrm{d}}{\mathrm{d}t^n}M_X(t) \Big|_{0}$ 
# 
# - *Equality of distributions*: 
# Two random variables have the same CDF/PMF/PDF if and only if they have the same MGFs for any $t$.
# 
# - $Y = \sum\limits_{i = 1}^n X_i \quad \Rightarrow \quad M_Y(t) = \prod\limits_{i = 1}^n M_{X_i}(t)$

#%% [markdown]
# ## [Characteristic Function](https://www.statlect.com/fundamentals-of-probability/characteristic-function)
# 
# Not all random variables possess a MFG. 
# However, all have a *characteristic function* which has almost identical properties.
# 
# 
# 
# 
#  
