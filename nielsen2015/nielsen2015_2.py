#%% [markdown]
# # Notes and exercises from [Nielsen2015](http://neuralnetworksanddeeplearning.com/)
# --- 

#%% [markdown]
# ## Ch. 2 Backpropagation
#
# ## Sec. 2.1 Matrix Notation
#
# $w_{ij}^{k}$: *weight* for the connection to the $i^\mathrm{th}$ neuron 
# in the $k^\mathrm{th}$ layer from the $j^\mathrm{th}$ neuron 
# in the $(k-1)^\mathrm{th}$ layer.
#
# $b_i^k, z_i^k, a_i^k$: *bias*, *weighted output*, *activation* 
# for the $i^\mathrm{th}$ neuron in the $k^\mathrm{th}$ layer.
# 
# Therefore: 
#
# $ \begin{equation}
#   z_i^k = \sum\limits_j w_{ij}^k a_j^{k-1} + b_i^k \\
#   a_i^k = \sigma (z_i^k)
# \end{equation} $
# 
# or in matrix notation:
#
# $ \begin{equation}
#   \mathbf{z}_k = \mathbf{W}_k \mathbf{a}_{k-1} + \mathbf{b}_k \\
#   \mathbf{a}_k = \sigma (\mathbf{z}_k)
# \end{equation} $
# 
# where $(\mathbf{a}_k)_i = a_i^k$ etc. 
# and $\mathbf{W}_k$ is the $N_k \times N_{k-1}$ weight matrix.


#%% [markdown]
# ## Sec. 2.2 Cost Function
#
# Assume a *cost function* of the form: 
# 
# $C = \frac{1}{n} \sum\limits_x C_x $ 
# and $C_x = \frac{1}{2}| \mathbf{y}(\mathbf{x}) - \mathbf{a}_N (x)|^2 $ 
# 
# where $(\mathbf{x}, \mathbf{y})$ is one of $n$ training example pairs 
# and $\mathbf{a}_N(\mathbf{x})$ is the activation of the last or $N^\mathrm{th}$ layer of the network.

#%% [markdown]
# ## Sec. 2.3 The Hadamard Product
#
# Define $\odot$ as the Hadamard/Schur/element-wise matrix product: 
# 
# ($\mathbf{u} \odot \mathbf{v})_i = u_i v_i$

#%% [markdown]
# ## Sec. 2.4 Equations of Backpropagation
# 
# Define $\delta_i^k = \frac{\partial C}{\partial {z_i^k}}$ as *error* in the $i^\mathrm{th}$ neuron 
# of the $k^\mathrm{th}$ layer. 
# ---
# **Equation 1**: 
#
# $ \begin{equation}
#   \delta_i^N = \frac{\partial C}{\partial a_i^N} \sigma^\prime(z_i^N) \\
#   \boldsymbol{\delta}_N = \nabla_{a_N} C \odot \sigma^\prime (\mathbf{z}_N)
# \end{equation} $
# 
# where $\nabla_{\mathbf{a}_N} C = \mathbf{a}_N - \mathbf{y} $ and $\sigma^\prime(z) = \frac{\mathrm{d} \sigma}{\mathrm{d} z}$.
# ---
# **Equation 2**:
#
# $ \begin{equation}
#   \boldsymbol{\delta}_k = \left( W_{k+1}^\mathsf{T} \boldsymbol{\delta}_{k+1} \right) 
#       \odot \sigma^\prime(z_i^N) \\
# \end{equation} $
#  
# where the error propagates backwards from the output layer.
# ---
# **Equation 3**: 
#
# $ \begin{equation}
#   \frac{\partial C}{\partial b_i^k} = \delta_i^k
# \end{equation} $
# ---
# **Equation 4**:
#
# $ \begin{equation}
#   \frac{\partial C}{\partial w_{ij}^k} = \delta_i^k a_j^{k-1}
# \end{equation} $
# ---
# Errors can be calculated by propagating Eq. 1 backwards with Eq. 2. 
# 
# Since $\sigma^\prime(z) \sim \mathrm{e}^{-z^2}$, 
# Eq. 1 means that errors vary little # when $|z| \gg 0$ 
# or the activation of the output neurons is near $0$ or $1$.
# Thus, biases (Eq. 3) learn slowly when the output neuron is saturated 
# and weights (Eq. 4) learn slowly when the output neuron is saturated 
# or the input neuron is low-activated. 

#%% [markdown]
# ## Sec. 2.5 The Backpropagation Algorithm
# 
# 1. Input $\mathbf{x}$ and set $\mathbf{a}^1 = \sigma(\mathbf{x})$ 
# 2. Feedforward: $\mathbf{z}_k = \mathbf{W}_k \mathbf{a}_{k-1} + \mathbf{b}_k$ and $\mathbf{a}_k = \sigma(\mathbf{z}_k)$
# 3. Output error $\mathbf{\delta}_N = \nabla_{\mathbf{a}_N} \odot \sigma^\prime(\mathbf{z}_N)$
# 4. Backpropagate: $\mathbf{\delta}_k = \left( W_{k+1}^\mathsf{T} \mathbf{\delta}_{k+1} \right) \odot \sigma^\prime(\mathbf{z}_k)$
# 5. Output: $\frac{\partial C}{\partial w_{ij}^k} = \delta_i^k a_j^{k-1}$ and $\frac{\partial C}{\partial b_i^k} = \delta_i^k$


#%%
