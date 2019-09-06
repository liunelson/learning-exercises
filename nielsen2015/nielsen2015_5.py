#%% [markdown]
# # Notes and exercises from [Nielsen2015](http://neuralnetworksanddeeplearning.com/)
# --- 

#%% [markdown]
# ## Ch. 5 Deep Neural Networks are Hard to Train?
#
# Deeper networks means more layers for abstraction. 
# 
# However, it has been observed that, while some layers are learning well, 
# others often become stuck and are not learning at all.
# 
# There's an intrinsic instability with learning by gradient descent 
# in many-layer neural networks.
#
# The Vanishing/Exploding-Gradient Problem
#
# By inspection, the learning rate, represented by $|\boldsymbol{\delta}^k|$, 
# for small $k$, # i.e. early layers, is unstable, 
# either *exploding* or *vanishing* relative to that of later layers.
#
# Consider a neural network with $N$ hidden layers but only 1 neuron: 
# $a_k = \sigma(z_k)$ and $a_k = w_k a_{k-1} + b_k$ for $k = 1 \ldots N$. 
#
# Then, 
# $\delta_1 = \frac{\partial C}{\partial z_1} = \frac{\partial C}{\partial a_4} \sigma^\prime(z_1) \prod\limits_{k = 2}^N \sigma^\prime(z_k) \: w_k$
#
# Thus, depending on whether $|\sigma^\prime(z_k) \: w_k|$ is greater or less than $1$, 
# $\delta_1$ may grow or shrink exponentially.
#
# This is avoided only when $w_k$ is large enough but not too large 
# such that $\sigma^\prime(w_k a_{k-1} + b_k) \approx 0$.
#

#%%
