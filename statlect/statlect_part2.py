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
# ## [Baye's Rule](https://www.statlect.com/fundamentals-of-probability/Bayes-rule)
#
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
#   


















#%%
