#%% [markdown]
# # Notes and exercises from [Statlect](https://www.statlect.com)
# --- 

#%% [markdown]
# # Part 1 - [Mathematical Tools](https://www.statlect.com/mathematical-tools/)
#
# ## Part 1.1 - [Set Theory](https://www.statlect.com/mathematical-tools/set-theory)
#
# *Set*: An unordered collection of objects, 
# denoted by a list within curly brackets, 
# e.g. $S = \{a, b, c, d\}$.
#
# The set $A = \{1, 2, 3, 4, 5\}$ is equivalent to 
# the set $B = \{n \in \mathbb{N} \; | \; n \leq 5\}$.
#
# Set membership: Object $a$ belongs to set $A$ if it is one of its *elements*, 
# i.e. $A = \{a, b, c\} \; \Rightarrow \; a \in A$.
#
# Set *inclusion*: If every element of set $A$ also belongs to set $B$, 
# then $A$ is included in $B$ and $A$ is a *subset* of $B$, i.e. $A \subseteq B$.
# 
# If $A \subseteq B$ but $B$ includes elements not in $A$, 
# $A$ is *strictly included* in $B$ or $A$ is a *proper subset* of $B$, 
# i.e. $A \subset B$.
#
# Corollary: $A \subseteq B \; \Rightarrow B \supseteq A$ ($B$ is a *superset* of $A$); 
# similarly, $A \subset B \; \Rightarrow B \supset A$.
#
# Set *union*: Operation that gives the set of all elements belonging to 
# *at least one* of the collected sets, 
# i.e. $A_1 \cup A_2 = \bigcup\limits_{i = 1}^2 A_i = \{x: x \in A_1 \; \mathrm{or} \; x \in A_2\}$.
#
# Set *intersection*: Operation that gives the set of all elements belonging to 
# *all* of the collected sets, 
# i.e. $A_1 \cap A_2 = \bigcap\limits_{i = 1}^2 A_i = \{x: x \in A_1 \; \mathrm{and} \; x \in A_2\}$.
# 
# $A$ intersects $B$ if $A \cap B \neq \varnothing$.
#
# $A$ and $B$ are *disjoint*" if $A \cap B = \varnothing$. 
#
# $\varnothing = \{ \}$ is the *empty set*.
#
# *Universal set*: the set that includes all objects (and itself), 
# often denoted as $\Omega$. 
#
# Set *difference*: Operation that gives the set of all elements of the former 
# but not the latter, 
# i.e. $B \setminus A = \{ x \in B \; | \; x \notin A \}$.
#
# Set *complement*: Operation that gives the set of all non-belonging elements, 
# i.e. $A^\complement = \Omega \setminus A = \{ x \in \Omega \; | \; x \notin A\}$. 
#
# De Morgan's Laws: 
# $\begin{align} (A \cup B)^\complement = A^\complement \cap B^\complement \\ (A \cap B)^\complement = A^\complement \cup B^\complement \end{align}$

#%% [markdown]
# ## Part 1.2 - Combinatorics
# 
# ### Permutations
#
# ### Combinations
#
# ### $k$-Permutations
#
# ### Partitions


#%%
