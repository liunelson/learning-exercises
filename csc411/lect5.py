# %%[markdown]
# # Notes from [CSC 411](http://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/)
# --- 

# %%[markdown] 
# ## Lecture 5 - Ensembles (Part II) 
# 
# A *weak* learner is a learning algorithm that outputs
# a hypothesis (e.g. a classifier) that performs only slightly better 
# than by chance.
# 
# Examples: decision stumps (trees with only one split).
# 
# *Boosting*:
# train (weak) classifiers sequentially,  
# each time focusing on training data points
# that were previously misclassified.
# 
# *Adaptive boosting* (AdaBoost): 
# - at each iteratiion, assigning larger weights $w_i$ 
# to data points $x_i$ that were mis-classified;
# - ensemble classifier $H$ is the weighted sum 
# of the weak classifiers $h_j$; 
# - bias is reduced by making subsequent classifiers 
# focus on their predecessors' mistakes;
# - Steps: 
#   1. given training dataset of size $N$ with $x_i \in \mathbb{R}^d$, 
# $y_i \in \{-1, 1\}$ (i.e. two classes);
#    2. initialize weights as $w_i = \frac{1}{N}$;
#    3. for iteration $m$,
#       $\begin{align} \epsilon_m &= \frac{\sum\limits_{i=1}^N w_i 1_{h(x_i) \neq y_i}}{\sum\limits_{i=1}^N w_i} \\ \alpha_m &= \frac{1}{2} \ln \left( \frac{1 - \epsilon_m}{\epsilon_m} \right) \\ w_{i,m+1} &= w_{i,m} \exp \left(-\alpha_m y_i h_m(x_i) \right) \end{align}$;
#    4. $H(x) = \mathrm{sgn} \left(\sum_{m=1}^M a_m h_m(x) \right)$.
# 
#  
# AdaBoost can be interpreted as a stage-wise estimation procedure 
# for an additive logistic regression model  
# wherein the minimized loss function is $L(y, h(x)) = \mathrm{E}[\mathrm{e}^{-y h(x)}]$.
# 
# Assuming each weak learning has error $\epsilon_m \leq \frac{1}{2} - \gamma \; \forall \; m$, 
# the training error of $H(x)$ is $L_N(H) = \frac{1}{N} \sum\limits_{i=1}^N 1_{H(x_i) \neq y_i} \leq \mathrm{e}^{-2 \gamma^2 M}$ 
  
# %%[markdown] 
# ## Lecture 6 - Linear Regression
# 
# Problem:
# Given observations $\{(x_N, t_N)\}$, 
# find parameters $w \in \mathbb{R}^{K \times 1}, b \in \mathbb{R}^{N \times 1}$ 
# such that $y = X w + b$ and $t_n \approx y_n$,  
# where $X \in \mathbb{R}^{N \times K}$ and $y \in \mathbb{R}^\times{N \times 1}$
# 
# Define square-error loss function: $\mathcal{L}(y_n, t_n) = \frac{1}{2}(y_n - t_n)^2$.
#   
# Define cost function: $\begin{align} \mathcal{C} &= \frac{1}{N} \sum\limits_{n=1}^N \mathcal{L}(y_n, t_n) \\ &= \frac{1}{2 N} \sum\limits_{n=1}^N (x_n w + b_n - t_n)^2 \end{align}$.
# 
# Minimize $\mathcal{C}$ by setting partial derivatives to zero 
# gives $w = (X^\mathsf{T} X)^{-1} X^\mathsf{T} t$.
# 
# Alternatively, use *gradient descent* (iterative algorithm for optimization).
# - initialize $w_k$ to some value;
# - update them by $w_k \leftarrow w_k - \alpha \frac{\partial \mathcal{C}}{\partial w_k} = w_k - \frac{\alpha}{N} \sum\limits_{n=1}^N (y_n - t_n) x_{n,k}$
# - in vector form, $w \leftarrow w - \alpha \: \nabla_w \mathcal{C} = w - \frac{\alpha}{N} X^\mathsf{T} (y - t)$;
# - $\alpha$ is the *learning rate*;
#   
# *Feature mapping*:
# - consider data that is modeled by $y = w_0 + w_1 x + w_2 x^2 + w_3 x^3$;
# - define the *feature map* $\psi(x) = \left[ 1 \: x \: x^2 \: x^3 \right]$;
# - we get back the same regression problem, $y = \psi(x)^\mathsf{T} w$
# - the degree of the polynomial is a hyperparameter like $k$ in kNN 
# and it can be tuned using cross-validation (training, validation, test); 
# - alternative to tuning is *regularization*.
# 
# 
# Regularization: 
# - a *regularizer* is a function that quantifies the cost of one hypothesis vs. another;
# - it appears as an extra term $\mathcal{R}$ in the cost function,  
# $\mathcal{L} \rightarrow \mathcal{L} + \lambda \mathcal{R}$;
# - an example is the $L^2$ regularizer, $\mathcal{R} = \frac{1}{2} |w|^2;
# - $\lambda$ is a hyperparameter to be tuned by cross-validation again;
# - thus, $w \leftarrow \begin{align} & w - \alpha \: \nabla_w \mathcal{C} \\ &= (1-\alpha \lambda)w - \alpha \end{align}$
# - or $w(n) \sim \mathrm{e}^{-\lambda n}$
# - $L^1$ regularization is $\mathcal{R} = |w|^1$. 
#    
# %%[markdown]
# 
# ## Lecture 7 - Linear Classification (Part I) 
# 
# Classification: 
# predicting discrete-valued targets ($t \in \{0, 1\}$).  
# 
# Binary classification is equivalent to linear regression:  
# $\begin{align} z &= x^\mathsf{T} w \\ y &= \begin{cases} 1 & z \leq 0 \\ 0 & z > 0 \end{cases} \end{align}$ 
# 
# However, the function $y(z)$ needs to be differentiable  
# for gradient descent to work:  
# use $y(z) = \sigma(z)$ where $\sigma(z) = \frac{1}{1 + \mathrm{e}^{-z}}$.   
# 
# We can use $\mathcal{L}_\textrm{SE}(y, t) = \frac{1}{2}(y - t)^2$ 
# or the *cross-entropy loss* function: 
# - $\begin{align} \mathcal{L}_\textrm{CE}(y, t) &= \begin{cases} -\ln y & t = 1 \\ -ln(1-y) & t = 0 \end{cases} \\ &= -t \ln y - (1-t)\ln(1-y) \end{align}$ 
# 
#  
# For numerical stability, combine both the logistic activation function 
# with the cross-entropy loss function as one: 
# - the *logistic cross-entropy loss*; 
# - $\mathcal{L}_\textrm{LCE} = t\ln(1+\mathrm{e}^{-z}) + (1-t)\ln(1+\mathrm{e}^{-z})$;
# - `L_LCE = t * np.logaddexp(0, -z) + (1 - t) * np.logaddexp(0, z)`.
# 
#  
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import logistic

y = np.arange(0, 1, 0.01)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4))
_ = 0
ax[_].plot(y, -np.log(y), label = '$t = 1$')
ax[_].plot(y, -np.log(1 - y), label = '$t = 0$')
ax[_].set_xlabel('y')
ax[_].set_ylabel('Cross-Entropy Loss')
_ = ax[_].legend()

_ = 1
z = np.arange(-3, 3, 0.01)
ax[_].plot(z, np.heaviside(-z, 0.5), label = '0-1')
ax[_].plot(z, 0.5*(z - 1)**2, label = 'least-square')
ax[_].plot(z,  0.5*(logistic.cdf(z) - 1)**2, label = 'log. least-square')
ax[_].plot(z, np.logaddexp(0, -z) , label = 'log. cross-entropy')
ax[_].set_ylim(ax[_-1].get_ylim())
ax[_].set_xlabel('z')
ax[_].set_ylabel('Loss')
_ = ax[_].legend()

# %%[markdown]
# 
# ## Lecture 8 - Linear Classification (Part II) 


# %%
