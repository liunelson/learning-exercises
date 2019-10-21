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
# More rigorously, a *space of events* is required to be a *sigma-algebra*, 
# i.e. have the following properties: 
# 1. $\Omega \in \mathcal{F}$ (whole set); 
# 2. $E \in \mathcal{F} \; \Rightarrow \; E^\complement \in \mathcal{F}$ (closure under complementation);
# 3. if $\{ E_1, E_2, \ldots E_i, \ldots \}$ is a sequence of subsets of \Omega 
# belonging to $\mathcal{F}$, then $\bigcup\limits_{i = 1}^\infty E_i \in \mathcal{F}$ 
# (closure under countable union).
# 
# Note: A subset of $\Omega$ that belongs to the sigma-algebra $\mathcal{F}$ is called *measurable*.


#%% [markdown]
# ## [Conditional Probability](https://www.statlect.com/fundamentals-of-probability/conditional-probability)
#