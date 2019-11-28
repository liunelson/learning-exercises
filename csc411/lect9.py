# %%[markdown]
# # Notes from [CSC 411](http://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/)
# --- 

# %%[markdown] 
# ## Lecture 9 - Support Vector Machines and Boosting
# 
# Consider the binary classification problem again, 
# $\begin{align} z &= w^\mathsf{T} x + b \\ y &= \mathrm{sgn}(z) \end{align}$.
# 
# Previously, this was solved using 
# logistic regression with a cross-entropy loss function.
# 
# Alternative: 
# - geometric solution;
# - desired decision boundary is a line (i.e. a $K-1$ hyperplane) 
# separating the two classes; 
# - defined by $f(x) = w^\mathsf{T} x + b$ for some $(w, b)$;
# - the *optimal seperating hyperplane* 
# is the one that maximizes the distance to the closest point  
# from either class (maximizing the *margin* of the classifier);
# - recall the signed point-plane distance formula, 
# $d = \frac{w^\mathsf{T}x^\prime + b}{||w||_2}$;
# - then we get the optimization problem
# - $\min ||w||_2^2$ such that $t_n (w^\mathsf{T}x_n + b) \leq 1$ 
# for all training examples $i = 1,\ldots,N$;
# - the important training examples are those 
# with margin $||w||_2$ (called *support vectors*);
# - this algorithm is called *support vector machine* (SVM)   
# - similar algorithms are called *max-margin* or *large-margin*;
# 
# 
# If the data is not separable:
# - some points could be allowed to be misclassified  
# using *slack variables* $\xi_n \geq 0$;
# - *soft-margin* constraint: 
# $t_n(w^\mathsf{T} x_n + b) \geq 1 - \xi_n$
# -  *soft-margin* objective: 
# $\min\limits_{w,b,\xi} \frac{1}{2} ||w||_2^2 + \gamma \sum\limits_{n=1}^N \xi_n$
# - new hyperparameter $\gamma$ for trading off margin with slack;
# 
# 
# Note that the soft-margin SVM can be written 
# to be a linear classifier with *hinge loss* 
# and a $L_2$ regularizer:
# - write $y_n(w, b) = w^\mathsf{T}x + b$;
# - rewrite constraint as $\xi_n \geq 1 - t_n y_n(w, b)$;
# - then the slack penalty can be written as 
# $\sum\limits_{n=1}^N \xi_n = \sum\limits_{n=1}^N \max\{0, 1 - t_n y_n(w,b)\}$; 
# - replace $\max\{0,y\}$ by $(y)_+$; 
# - thus the optimization problem becomes 
# $ \min\limits_{w,b} \sum\limits_{n=1}^N \left(1 - t_n y_n(w,b)\right)_+ + \frac{1}{2 \gamma} ||w||_2^2$
# - this is just a linear classifier 
# - with a loss function $\mathcal{L}(y,t) = (1-ty)_+$ (i.e. *hinge loss*) 
# and a $L_2$-norm regularizer;
# - to find $w$, use gradient descent;
# 
# 
# Recall AdaBoost to reinterpret it in terms of loss functions:
# - consider a class $\mathcal{H}$ of hypothesis; 
# - each hypothesis is $h_i: x \rightarrow \{-1,+1\}$ 
# (a weak learner or a *base*);
# - an additive model with $m$ terms is given by $H_m(x) = \sum\limits_{i=1}^m \alpha_i h_i(x)$;
# - a way to fit additive models is *stagewise training*:
#    1. initialize $H_0(x) = 0$;
#    2. for $m = 1$ to $T$, compute the $m$-th hypothesis and its coefficients, $(h_m, \alpha_m) \leftarrow \arg \min\limits_{h \in \mathcal{H},\alpha} \sum\limits_{n=1}^N \mathcal{L}(H_{m-1}(x_n) + \alpha h(x_n), t_n)$
#    3. add it to the additive model, $H_m = H_{m-1} + \alpha_m h_m$;
# - consider a exponential loss function $\mathcal{L}(y,t) = \mathrm{e}^{-ty}$;
# - propagating...
# - we recover the AdaBoost algorithm:
# - $\begin{align} h_m &\leftarrow \arg \min\limits_{h \in \mathcal{H}} \sum\limits_{n=1}^N w_n^{(m)} 1_{h(x_n) \neq t_i} \\ \alpha &= \frac{1}{2} \ln \left(\frac{1 - \epsilon_m}{\epsilon_m} \right)\\ \epsilon_m &= \frac{\sum\limits_{n=1}^N w_n^{(m)} 1_{h(x_n) \neq t_i} }{\sum\limits_{n=1}^N w_n^{(m)}} \\ w_n^{(m+1)} &= w_n^{(m)} \exp\left(-\alpha_m h_m(x_n) t_n \right) \end{align}$
# - thus, AdaBoost is just an additive model with stagewise training 
# and an exponential loss function.

# %%[markdown] 
# ## Lecture 10 - Neural Networks (Part I)
# 
# Consider a *unit* that computes 
# the equation $y = \phi(w^\mathsf{T} x + b)$:  
# - $x$ is the input vector;
# - $y$ is the scalar output of the unit;
# - $w$ are the weights associated with the input units;
# - $b$ are the associated biases.
# - similar to the expression for logistic regression 
# $y = \sigma(w^\mathsf{T}x + b)$.
# 
# 
# Connect several units together  
# into a *directed acyclic graph* to  
# give a *feed-forward neural network*:
# - *recurrent neural networks* can have cycles; 
# - structure: an input layer, many hidden layers, an output layer;
# - each layer connects $N$ input units to $M$ output layers;
# - *fully connected* = all input units are connected to all output units; 
# - a fully connected multilayer network is called a *multilayer perceptron*;
# - $y = \phi(W x + b)$
# - examples of $phi(z)$: 
#    1. linear ($y = z$), 
#    2. rectified linear unit (ReLU, $y = \max(0,z)$), 
#    3. soft-ReLU ($y = \ln(1 + \mathrm{e}^z)$),
#    4. hard-threshold/heaviside/unit-step ($y = H(z)$),
#    5. logistic ($y = \frac{1}{1 + \mathrm{e}^z}$),
#    6. tanh ($y = \tanh(z) $);
# 
# 
# Expressive power:
# - any sequence of linear layers is equivalent to a single linear layer;
# - thus, deep *linear* networks are no more expressive than linear regression;
# - however, a multilayer feed-forward network with 
# *nonlinear* activation functions are *universal function approximators*;
# - universality $\Rightarrow$ risk of overfitting;
# 
# 
# The *backpropagation* algorithm:
# 
# 1. forward pass by computing $z, y = \phi(z), \mathcal{L}, \mathcal{R}, \mathcal{L}_\textrm{reg}$
# in order;
# 2. backward pass by computing partial derivatives of $\mathcal{L}_\textrm{reg}$ 
# with chain rule.
# 
# 
# Example: 
# - $\begin{align} z &= W^{(1)}x + b^{(1)} \\ h &= \sigma(z) \\ y &= W^{(2)}x + b^{(2)} \\ \mathcal{L} &= \frac{1}{2}|| t-y||^2 \end{align}$
# - $\begin{align} \frac{\partial L}{\partial y} &= y - t \\ \frac{\partial L}{\partial W^{(2)}} &= \frac{\partial L}{\partial y} h^\mathsf{T} \\ \ldots \end{align}$
# 
# 
# Computational cost:
# - forward pass: one add-multiply operation per weight;   
# - backward pass: two add-multiply operation per weight;
# - cost is linear in number of layers, quadratic in the number of units per layer.

# %%[markdown] 
# ## Lecture 11 - Neural Networks (Part II)
# 
# Problem: 
# fully connected multilayer networks, necessary for real-life object recognition, 
# are prohibitively costly. 
# 
# Solution:
# locally connected layers.
# 
# *Convolutional network*:
# - convolution layer: convolve a *kernel* or *filter* across the inputs 
# - basically force a local set of inputs to share
# the same weights and biases over the entire input vector;
# - conv. hyperparameters: 
# $w \times h$ size of filters, 
# number of filters (depth of output volume), 
# stride length (step size of convolution).
# - pooling layer: 
# combine the filter responses and downsample 
# (e.g. *max* or *average* pooling); 
# - pooling hyperparameters: 
# spatial extent (size of pooling area), stride length; 
      
#%% [markdown]
# ## Lecture 12 - Principal Component Analysis
# 
# PCA is an unsupervised learning algorithm 
# and a linear model with a closed-form solution.
#  
# Projection onto a subspace $\mathcal{S} \in \mathbb{R}^K$:
# - $\mathbf{z} = \mathbf{U}^\mathsf{T} (\mathbf{x} - \mathbf{\mu})$; 
# - $\mathbf{x}^\prime = \mathbf{U}\mathbf{z} + \mathbf{\mu}$;
# - the columns of $\mathbf{U}$ forms an orthonormal basis of $\mathcal{S}$;
# - $\mathbf{\mu}$ is the origin of $\mathcal{S}$;
# - $\mathbf{x}^\prime \in \mathbb{R}^N$ is the *reconstruction* of $\mathbf{x} \in \mathbb{R}^L$ 
# (the point in $\mathcal{S}$ closest to $\mathbf{x}$);
# - $\mathbf{z} \in \mathbb{R}^K$ is the *(latent) representation* (or *code*) of $\mathbf{x}$.
# 
# 
# When $K \ll L$, this mapping is called *dimensionality reduction*; 
# learning such a mapping is called *representation learning*.
# 
# Choosing a good subspace $\mathcal{S}$:
# - set $\boldsymbol{\mu} = \frac{1}{N}\sum\limits_{n=1}^N \mathbf{x}_n$;
# - two equivalent criteria:
#    1. minimize *reconstruction error*, 
# $\min \frac{1}{N}\sum\limits_{n=1}^N ||\mathbf{x}_n - \mathbf{x}^\prime_n ||^2$;
#    2. maximize variance of the code vectors, 
# $\max \sum\limits_{i=1}^K \mathrm{var}[z_i] = \ldots = \frac{1}{N} \sum\limits_{n=1}^N ||\mathbf{z}_n||^2$;
# - both are equivalent since 
# $\frac{1}{N}\sum\limits_{n=1}^N ||\mathbf{x}_n - \mathbf{x}^\prime_n ||^2 + \frac{1}{N}\sum\limits_{n=1}^N ||\mathbf{x}^\prime_n - \boldsymbol{\mu}||^2 = \frac{1}{N}\sum\limits_{n=1}^N ||\mathbf{x}_n - \boldsymbol{\mu}||^2 = \textrm{const}$.
# 
# 
# Recall *spectral decomposition*: 
# - $\mathbf{A}$ is a symmetry matrix with a full set of eigenvectors and eigenvalues $\lambda_i$;
# - $\mathbf{A} = \mathbf{Q} \mathbf{D} \mathbf{Q}^\mathsf{T}$;
# - $\mathbf{Q}$ is orthogonal with columns $p$ as the eigenvectors of $A$;
# - $\mathbf{D} = \mathrm{diag}({\lambda_i})$;
# - $\mathbf{A}$ is positive semidefinite iff $\lambda_i \geq 0 \; \forall \; i$;
# 
# 
# Define the *empirical/sample covariance matrix*: 
# - $\mathbf{\Sigma} = \frac{1}{N} \sum\limits_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^\mathsf{T}$; 
# - $\mathbf{\Sigma}$ is symmetric and positive semidefinite;
# - the optimal PCA subspace is spanned by the top $K$ eigenvectors of $\mathbf{\Sigma}$; 
# - they are the *principal components*;
# - see the *Courant-Fischer min-max theorem*.
# - note that $\mathrm{cov}[\mathbf{z}] = \ldots = [\mathbf{I} \: \mathbf{0}] D [\mathbf{I} \: \mathbf{0}]^\mathsf{T}$
# - i.e. the basis vectors of $\mathbf{z}$ are de-correlated.
# 
# 
# *Autoencoders*: 
# - they are feed-forward neural networks that predict $\mathbf{x}$ given $\mathbf{x}$; 
# - non-trivial example: 
# add a *bottleneck layer* whose dimension (number of units in layer) 
# is much smaller than that of the inputs;  
# - reasons: 
#    1. map high-dim. data to lower dimensions for visualization;
#    2. learn abstract features in an unsupervised way;
# - linear autoencoder:  
#    1. a $L$-d input layer, $K$-d hidden layer; $L$-d output layer ($\mathbf{x} \xrightarrow{\mathbf{W}_1} \mathbf{z} \xrightarrow{\mathbf{W}_2} \mathbf{y}$);
#    2. linear activations with squared-error loss $\mathcal{L} = ||\mathbf{x} - \mathbf{y}||^2$;
#    3. $\mathbf{y} = \mathbf{W}_2 \mathbf{W}_1 \mathbf{x}$;
#    4. $\mathbf{W}_1$ encodes while $\mathbf{W}_2$ decodes;
#    5. optimal mapping is just PCA ($\mathbf{W}_1 = \mathbf{U}^\mathsf{T}$, $\mathbf{W}_2 = \mathbf{U}$);
# - nonlinear autoencoder: 
#    1. projection onto a nonlinear manifold; 
#    2. i.e. nonlinear dimensional reduction;
#    3. better than linear autoencoders; 
