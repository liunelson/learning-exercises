# %%[markdown]
# # Notes from [CSC 411](http://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/)
# --- 

# %%[markdown] 
# ## Lecture 13 - Probabilistic Models (Part I)
# 
# ### Maximum Likelihood (ML) Estimation
# 
# Likelihood function $L(\theta)$  
# is the probability of the observed data, 
# as a function of some model parameter $\theta$.
# 
# Log-likelihood function: $\ell(\theta) = \log L(\theta)$
# 
# Example: 
# - consider a observed sequence of $N$ temperature values $\{x_n\}$ 
# that behave as a random variable $X$;
# - $X$ follows a Gaussian distribution $\mathcal{N}(x; \mu,\sigma^2)$
# - $\mu, \sigma^2$ are unknown;
# - the log-likelihood is 
# $\ell(\mu, \sigma) = \log \left( \prod\limits_{n=1}^N \mathcal{N}(x_n; \mu, \sigma) \right)$;
# - then $\left. \frac{\partial \ell}{\mathrm{\partial}\mu} \right|_\hat{\mu} = 0 \: \Rightarrow \hat{\mu} = \bar{x}$
# - and $\left. \frac{\partial \ell}{\mathrm{\partial}\sigma} \right|_\hat{\sigma} = 0 \: \Rightarrow \hat{\sigma} = \frac{1}{N} \sum\limits_{n=1}^N (x_n - \mu)^2$
#  
# 
# If there is no closed-form solution associated with some complicated PDF, 
# then use gradient descent.
# 
# Note that we've been doing ML estimation all along: 
# - e.g. linear regression with squared-error loss,  
# $\begin{align} p(t|y) &= \mathcal{N}(t;y,\sigma^2) \\ -\log\left(p(t|y)\right) &= \frac{1}{2 \sigma^2}(y-t)^2 + \textrm{const} \end{align}$
# - e.g. logistic regression with cross-entropy loss, 
# $\begin{align} p(t|y) &= y \\ -\log\left(p(t|y)\right) &= - \left(t \log(y) + (1-t)\log(1-y)\right) \end{align}$  
# 
# 
# Two approaches to classification: 
# 1. *discriminative* classifiers estimate parameters of decision boundary
# directly from labeled examples;
#    - learn $p(y|\mathbf{x})$ directly (logistic regression models);  
#    - learn mappings fro inputs to classes (decision trees);
# 2. *generative* (*Bayes*) classifiers model the distribution characteristic of the class; 
#    - build a model of $p(\mathbf{x}|y)$;
#    - apply Bayes' Rule.
#
#
# ### Naive Bayes
#  
# Bayes classifer: 
# - aim: classify emails into either `spam` ($c = 1$) or `not-spam` ($c = 0$);
# - using *bag-of-words* features, get binary vector $\mathbf{x} = [x_1 \; \ldots \; x_K]^\mathsf{T}$ for each email;
# - compute class probabilities using Bayes' Rule, 
# $p(c|\mathbf{x}) = \frac{p(\mathbf{x}|c) p(c)}{p(\mathbf{x})}$;
# - i.e. $\textrm{posterior} = \frac{\textrm{class likelihood} \times \textrm{prior}}{\textrm{evidence} }$;
# - compute $p(\mathbf{x}) = p(\mathbf{x}|c = 0) p(c = 0) + p(\mathbf{x}|c = 1) p(c = 1)$;
# - to known $p(\mathbf{x}|c)$ and $p(c)$, 
# requires defining a joint distribution $p(c,x_1, \ldots,x_K)$
# - but this requires $2^{K+1}$ entries!
# - impose a structure as constraints.
# 
# 
# *Naive Bayes* assumes that the word features $x_k$  
# are *conditionally independent* given the class $c$; 
# thus, $p(c,x_1,\ldots,x_K) = p(c) p(x_1|c) \ldots p(x_K|c)$.
# 
# Compact representation of the joint distribution: 
# - prior probability of class, $p(c = 1) = \theta_C$;
# - conditional probability of word feature given class, $p(x_k = 1|c) = \theta_{k,c}$
# - $2K + 1$ parameters total (instead of $2^{K+1}$).
# 
# 
# This model can be represented as an *directed graph* (*Bayesian network*): 
# $c$ is the root node and $x_k$ are the leaf nodes.
#
# The parameters can be learned efficiently since the log-likelihood 
# decomposes into independent terms:
#  
# $\begin{align} \ell(\boldsymbol{\theta}) &= \log \left(\prod\limits_{n=1}^N p(c^{(n)}, \mathbf{x}^{(n)}) \right) \\ &= \ldots \\ &= \sum\limits_{n=1}^N \log \left( p(c^{(n)}) \right) + \sum\limits_{k=1}^K \sum\limits_{n=1}^N \log \left( p(x_k^{(n)}|c^{(n)})) \right) \end{align}$
# 
# Maximize $\sum_n \log \left(p(x_k^{(n)} | c^{(n)}) \right)$: 
# - let $\theta_{a,b} = p(x_k = a|c = b)$; $\theta_{1,b} = 1 - \theta_{0,b}$;
# - then 
# $\log \left( p(x_k^{(n)}|c^{(n)})) \right) = c^{(n)} x_k^{(n)} \log(\theta_{1,1}) + c^{(n)} (1 - x_k^{(n)}) \log(1 - \theta_{1,1}) + (1 - c^{(n)}) x_k^{(n)} \log(\theta_{1,0}) + (1 - c^{(n)}) (1 - x_k^{(n)}) \log(1 - \theta_{1,0})$
# - obtain the ML estimates by setting the derivatives to zero; 
# - thus $\theta_{1,1} = \frac{N_{1,1}}{N_{1,1} + N_{0,1}}$ and $\theta_{1,0} = \frac{N_{1,0}}{N_{1,0} + N_{0,0}}$;
# 
# 
# Predict the class by performing an *inference* using Bayes' Rule: 
# $p(c|\mathbf{x}) = \frac{p(c) p(\mathbf{x}|c)}{\sum_{c^\prime} p(c^\prime) p(\mathbf{x}|c^\prime)} \propto p(c) \prod_k p(x_k|c)$
# 
# Naive Bayes is a very inexpensive algorithm: 
# - training time: 
#    1. compute co-occurence counts of each feature with labels; 
#    2. one pass only to do ML estimation of parameters;
# - test time: apply Bayes' Rule;
# - not very accurate though due to *naive* assumption.   
  

# %%[markdown] 
# ## Lecture 14 - Probabilistic Models (Part II)
# 
# Problem with ML estimation: what if too little data?
# 
# Example: count coin flips but too few to even get a single `tail` event.
# 
# Such *data sparsity* causes $\ell(\theta)$ to be $-\infty$. 
# 
# In ML estimation, the observations are treated random variables but 
# the parameters are not.
# 
# In *Bayesian parameter estimation*, the latter are random variables too: 
# - prior distribution $p(\boldsymbol{theta})$ encodes beliefs about parameters before observation;
# - likelihood is as before.
# - we update our beliefs about the parameters by computing the posterior distribution.
# 
# 
# Recall the Bernoulli-type coin flip example; 
# - the likelihood is $L(\theta) = p(\mathcal{D}) = \theta^{N_H} (1 - \theta)^{N_T}$; 
# - specify the prior $p(\theta)$;
# - choose an *uninformative prior*:
#    1. could be an uniform distribution;
#    2. assuming 50% is more probable, we choose a *beta distribution* 
#       $p(\theta;a,b) = \mathrm{Beta}(\theta;a,b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \theta^{a-1} (1 - \theta)^{b-1}$
# - compute the posterior distribution, 
#   $\begin{align} p(\theta|\mathcal{D}) & \propto p(\theta) p(\mathcal{D}|\theta) \\ & \propto \theta^{a - 1 + N_H} (1 - \theta)^{b - 1 + N_T} \\ &= \mathrm{Beta}(\theta; N_H + a; N_T + b)\end{align}$ 
# - convenient that the posterior is just another Beta distribution; 
# - thus, $\mathrm{E}[\theta|\mathcal{D}] = \frac{N_H + a}{N_H + N_T + a + b}$;
# - $a, b$ are known as *pseudo-counts* as a result.
# 
# 
# As $N_H, N_T$ become large, the data/likelihood overwhelms the prior.
# 
# The posterior itself is used to compute the *posterior predictive distribution*: 
# 
# $p(\mathcal{D}^\prime|\mathcal{D}) = \int p(\theta|\mathcal{D}) p(\mathcal{D}^\prime|\theta) \mathrm{d}\theta$.
# 
# In the above example: 
# $\begin{align} p(x^\prime = \mathrm{H}|\mathcal{D}) = \int p(\theta|\mathcal{D}) p(x^\prime = \mathrm{H}|\theta) \mathrm{d}\theta \\ &= \int \mathrm{Beta}(\theta; N_H + a, N_T + b) \theta \mathrm{d}\theta \\ &= \ldots \\ &= \frac{N_H + a}{N_H + N_T + a + b} \end{align}$.
# 
# Comparing ML estimation with Bayesian estimation: 
# - Bayesian estimation can handle data sparsity; 
# - former is an optimization problem 
# while later is an integration problem;
# - former is easier due to gradient descent;
# 
# 
# ### Maximum A-Posteriori (MAP) Estimation
# 
# MAP estimation: find the most likely parameter settings under the posterior. 
# 
# This converts the Bayesian parameter estimation problem into an optimization one:  
# 
# $\begin{align} \hat{\theta}_\textrm{MAP} &= \arg \max\limits_\theta p(\theta|\mathcal{D}) \\ &= \arg \max\limits_\theta p(\theta,\mathcal{D}) \\ &= \arg \max\limits_\theta \log p(\theta) + \log p(\mathcal{D}|\theta) \end{align}$
# 
# In the coin-flip example: 
# - the joint probability is 
# $\log p(\theta, \mathcal{D}) = \log p(\theta) + \log p(\mathcal{D}|\theta) = \textrm{const} + \log \mathrm{Beta}(\theta; N_H+a, N_T+b)$
# - maximize by finding critical point, 
# $\frac{\mathrm{d}}{\mathrm{d} \theta} \log(\theta, \mathcal{D}) = 0$;
# - solving for $\theta$, $\hat{\theta}_\textrm{MAP} = \frac{N_H + a - 1}{N_H + N_t + a + b - 2}$.
#  
# 
# | | Formula | $(N_H = 2, N_T = 0)$ |  $(N_H = 55, N_T = 45)$ | 
# |:---:|:---:|:---:|:---:|
# |$\hat{\theta}_\textrm{ML}$| $\frac{N_H}{N_H + N_T}$ | 1 | 0.55 |
# |$\hat{\theta}_\textrm{BE}$| $\frac{N_H + a}{N_H + N_T + a + b}$ | 0.67 | 0.548 |
# |$\hat{\theta}_\textrm{MAP}$| $\frac{N_H + a - 1}{N_H + N_T + a + b - 2}$ | 0.75 | 0.549 | 

# %% [markdown]
# ### Gaussian Discriminant Analysis (GDA)
# 
# In a generative model, we don't try to separate the classes; try to model 
# we try to model the class distribution $p(\mathbf{x}|t = k)$ 
# (which could be very complex).
# 
# Recall Bayes classifier: 
# $h(\mathbf{x}) = \arg \max p(t = k|\mathbf{x}) = \arg \max \frac{p(\mathbf{x}|t = k) p(t = k)}{p(\mathbf{x})}$.
# 
# Consider a continuous $\mathbf{x}$ 
# and model $p(\mathbf{x}|t = k)$ as a multivariate Gaussian distribution.
# 
# For $\mathbf{x} \in \mathbb{R}^d$, 
# *Gaussian discriminant analysis* (or *Gaussian Bayes classifier* (GBC)) assumes:
# 
# $p(\mathbf{x}|t = k) = \frac{1}{(2 \pi)^{d/2} \sqrt{|\Sigma_k|}} \exp \left( -\frac{1}{2} (\mathbf{x} - \mathbf{\mu}_k)^\mathsf{T} \Sigma_k^{-1} (\mathbf{x} - \mathbf{\mu}_k) \right)$
# 
# Each class $k$ has an associated mean vector $\mathbf{\mu}_k$ 
# and covariance matrix $\Sigma_k$ (with $\mathcal{O}(d^2)$ parameters).
#
# Example: 
# $\mathbf{X}$ is a $N \times d$ matrix, 
# $N$ observations/instances/examples of 
# some value with $d$ inputs/features/attributes.
# 
# $\Sigma = \mathrm{cov}[\mathbf{x}] = \mathrm{E}[(\mathbf{x} - \mathbf{\mu})^\mathsf{T} (\mathbf{x} - \mathbf{\mu})]$
# 
# Note that $(\mathbf{x} - \mathbf{\mu}_k)^\mathsf{T} \Sigma_k^{-1} (\mathbf{x} - \mathbf{\mu}_k)$ 
# is the *Mahalanobis distance* 
# (a measure of the distance from $\mathbf{x} to $\mathbf{\mu}_k$ in terms of $\Sigma_k$) 
# 
# The log of the class posterior is then: 
# $\begin{align} \log p(t_k|\mathbf{x}) &= \log p(\mathbf{x}|t_k) + \log p(t_k) - \log p(\mathbf{x}) \\ &= \end{align}$
#
# Not very good if class-conditional data is not multivariate Gaussian.
# 
  
# %%[markdown] 
# ## Lecture 15 - k-Means
# 
# Use unsupervised learning when 
# one wants to infer the *latent* (unobserved) causal structure 
# underlying the data.
# 
# Consider the *k-means clustering* algorithm and then 
# reformulate it as a *latent variable model* 
# and apply the *expectation-maximization* (EM) algorithm.
# 
# Clusters: 
# groups of data points/examples which are similar within 
# but dissimilar without, i.e. *multimodal* distribution. 
# 
# Clustering: group unlabelled data points into clusters 
# (NP-hard problem).
# 
# Assume: 
# - $N$ data points $\mathbf{x}_n \in \mathcal{R}^d$; 
# - $\mathbf{x}_n$ belongs to $K$ classes;  
# - similarity measure $\rightarrow$ Euclidean distance;
# 
#  
# Alternate between *cluster assignment* and *computing cluster means*.
#  
# k-means clustering: 
# - initialization, randomly choose points as cluster centres; 
# - iteratively alternate
#    1. *cluster assignment* (assign data points to their nearest cluster);
#    2. *refitting* (move cluster centres to the centre of gravity of assigned points);


# %%
