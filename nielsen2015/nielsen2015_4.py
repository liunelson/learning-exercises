#%% [markdown]
# # Notes and exercises from [Nielsen2015](http://neuralnetworksanddeeplearning.com/)
# --- 

#%% [markdown]
# ## Ch. 4 Computing Functions
#
# The *Universal Approximation Theorem* states that 
# 
# > a feed-forward network with a single hidden layer 
# containing a finite number of neurons can approximate 
# continuous functions on compact subsets of $\mathbb{R}^n$, 
# under mild assumptions on the activation function.
#
# More precisely:
#
# > Let $\sigma: \mathbb{R} \rightarrow \mathbb{R}$ be 
# a non-constant, bounded, and continuous 'activation function.' 
# Let $I_m$ denote the $m$-dimensional unit hypercube $[0, 1]^m$ 
# and $C(I_m)$ denote the space of real-valued continuous functions on $I_m$.
#
# > Then, for any $\epsilon > 0$ and function $f \in C(I_m)$, 
# there exist an integer $N$, real constants $c_i, b_i \in \mathbb{R}$, 
# and real vectors $\mathbf{w}_i \in \mathbb{R}^m$ 
# for $i = 1 \ldots N$ such that: 
# 
# > $F(\mathbf{x}) = \sum\limits_i c_i \: \sigma( \mathbf{w}_i^\mathsf{T} \mathbf{x} + b_i)$ 
#
# > $| F(\mathbf{x}) - f(\mathbf{x}) | < \epsilon \quad \forall \quad \mathbf{x} \in I_m$ 
#   
# Graphical proof: 
# A pair of neurons with some $c_i, b_y, \mathbf{w}_i$ can generate 
# a function $F(x) \approx rect(x)$ which can build each point $f(x)$.


#%%
