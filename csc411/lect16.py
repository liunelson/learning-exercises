# %%[markdown]
# # Notes from [CSC 411](http://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/)
# --- 

# %%[markdown] 
# ## Lecture 16 - Expectation-Maximization (Part I-II)
#
# Reformulate clustering in terms of a *generative* model 
# by focusing on deriving the probability distribution 
# that could generate the observed data 
# (instead of focusing on decision boundaries). 
# 
# 
# Define the joint distribution as 
# $p(\mathbf{x}, z) = p(\mathbf{x}| z) p(z)$ 
# where $z$ are the class labels.
# 
# Since $z$ is a priori unknown in unsupervised learning, we write 
# $p(\mathbf{x}) = \sum\limits_z p(\mathbf{x}, z) = \sum\limits_z p(\mathbf{x}| z) p(z)$.
# 
# This is a *mixture model*.
# 
# Example: *Gaussian mixture model* (GMM) 
# - a GMM represent a distribution as $p(\mathbf{x}) = \sum\limits_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}| \mu_k, \Sigma_k)$; 
# - $\pi_k$ are *mixing coefficients* such that $\sum_k \pi_k = 1$ and $\pi_k \geq 0 \: \forall \: k$; 
# - GMM is a density estimator; 
# - GMMS are *universal approximators of densities* (with enough Gaussians);
# 
# 
# Fitting GMMs by maximum likelihood algorithm: 
# - define the log-likelihood $\ell(\boldsymbol{\pi}, \mu, \Sigma) = \ln p(\mathbf(X)|\mu,\mu,\Sigma) = \sum\limits_{n=1}^N \ln \left( \sum\limits_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}^{(n)}|\mu_k,\Sigma_k ) \right)$
# - maximize it with respect to $\Theta = {\pi_k, \mu_k, \Sigma_k}$;
# 
# 
# From above, $z$ is a hidden/latent variable:   
# - hidden/latent variables are model variables which are always unobserved;
# - let $z \sim \textrm{Categorial}(\boldsymbol{\pi})$; 
# - then $\begin{align} p(\mathbf{x}) &= \sum\limits_{k=1}^K p(\mathbf{x}, z = k) \\ &= \sum\limits_{k=1}^K p(z = k) p(\mathbf{x}|z = k) \end{align}$
# - where $p(z = k) = \pi_k$ and $p(\mathbf{x}|z=k) = \mathcal(N)(\mathbf{x}|\mu_k,\Sigma_k)$.
# 
# 
# If we knew the $z^{(n)}$ associated with each $\mathbf{x}^{(n)}$, 
# this ML problem could be solved easily: 
# - $\ell(\boldsymbol{\pi}, \mu, \Sigma) = \sum\limits_{n=1}^N \left( \ln p(\mathbf{x}^{(n)}|z^{(n)}; \mu, \Sigma) + \ln p(z^{(n)}|\boldsymbol{\pi}) \right)$
# - recall Gaussian Bayes classifiers;
# - solution: $\begin{align} \mu_k &= \frac{\sum\limits_{n=1}^N 1_{z^{(n)} = k} \; \mathbf{x}^{(n)}}{\sum\limits_{n=1}^N 1_{z^{(n)} = k}} \\ \mu_k &= \frac{\sum\limits_{n=1}^N 1_{z^{(n)} = k} \; (\mathbf{x}^{(n)} - \mu_k) (\mathbf{x}^{(n)} - \mu_k)^\mathsf{T} }{\sum\limits_{n=1}^N 1_{z^{(n)} = k}} \\ \pi_k &= \frac{1}{N} \sum\limits_{n=1}^N 1_{z^{(n)} = k} \end{align}$
# 
# 
# Using the *expectation-maximization* algorithm: 
# - *E-step*: compute the posterior probability over $z$ given current model (contribution of each Gaussian to each data point); 
# - *M-step*: change the parameters of each Gaussian to maximize the probability that it generates its assigned data points.
# 
# 
# Recall the k-means clustering algorithm. 
# - the assignment step $\leftrightarrow$ the E-step;
# - the refitting step $\leftrightarrow$ the M-step.
# 
# 
# The EM algorithm is a powerful method for finding ML solutions for such latent-variable models.
# 
# Derivation: 
# - write the log-likelihood function $\ell(\mathbf{X},\Theta) = \sum\limits_n \ln (P(\mathbf{x}^{(n)});\Theta) = \sum\limits_n \ln \left( \sum\limits_k P(\mathbf{x}^{(n)}), z^{(n)} = k;\Theta) \right)$; 
# - introduce a new distribution $q$, $\ell(\mathbf{X},\Theta) = \sum\limits_n \ln \left( \sum\limits_k q_k \frac{P(\mathbf{x}^{(n)}), z^{(n)} = k;\Theta)}{q_k} \right)$;
# - from Jensen's inequality for concave functions like $\ln$, 
# $f(\mathrm{E}[x]) = f\left(\sum\limits_i p_i x_i \right) \geq \sum\limits_i p_i f(x_i) = \mathrm{E}[f(x)]$;
# - then $\sum\limits_n \ln \left( \sum\limits_k q_k \frac{P(\mathbf{x}^{(n)}), z^{(n)} = k;\Theta)}{q_k} \right) = \sum\limits_n \sum\limits_k q_k \ln \left( \frac{P(\mathbf{x}^{(n)}), z^{(n)} = k;\Theta)}{q_k} \right) $;
# - this moved the summation outside of the $\ln$;
# - maximizing this lower bound to force $\ell$ to increase;
# - how to pick $q_k$?
# - suppose $q_k = p(z^{(n)} = k|x^{(n)},\Theta^\textrm{old})$; 
# - then optimize $Q(\Theta) = \sum\limits_n \sum\limits_k p(z^{(n)} = k|x^{(n)},\Theta^\textrm{old}) \ln P(\mathbf{x}^{(n)}), z^{(n)} = k;\Theta)$; 
# - this is just the expectation over the distribution $P$, $Q(\Theta) = \mathrm{E}[\ln P(\mathbf{x}^{(n)}), z^{(n)} = k;\Theta)]$;
# - maximizing $Q(\Theta)$ now gives a better lower bound for the log-likelihood. 
# 
#%% [markdown] 
# General EM algorithm:
#  
# 1. initialize $\Theta^\textrm{old}$;
# 2. E-step, evaluate $p(\mathbf{Z}|\mathbf{X},\Theta^\textrm{old})$;
# and compute $Q(\Theta,\Theta^\textrm{old}) = \sum\limits_z p(\mathbf{Z}|\mathbf{X},\Theta^\textrm{old}) \ln p(\mathbf{Z}|\mathbf{X},\Theta)$;
# 3. M-step, maximize: $\Theta^\textrm{new} = \arg \max\limits_\Theta Q(\Theta,\Theta^\textrm{old})$; 
# 4. evaluate log-likelihood and check for convergence; if not converged, set $\Theta^\textrm{old} = \Theta^\textrm{new}$ and repeat step 2.
# 
# 
# As applied to GMM: 
# - initialize the parameters $\mu_k, \Sigma_k, \pi_k$ (with k-means); 
# - conditional probability of $\mathbf{z}$ given $\mathbf{x}$ by Bayes' Rule: 
# $\begin{align} \gamma_k &= p(z = k|\mathbf{x}) \\ &= \frac{p(z = k) p(\mathbf{x}|z = k)}{p(\mathbf{x})} \\ &= \frac{p(z = k) p(\mathbf{x}|z = k)}{\sum\limits_{j=1}^K p(z = j) p(\mathbf{x}| z = j)} \\ &= \frac{\pi_k \; \mathcal{N}(\mathbf{x}|\mu_k,\Sigma_k)}{\sum\limits_{j=1}^K \pi_j \; \mathcal{N}(\mathbf{x}|\mu_j, \Sigma_j)} \end{align}$
# - $\gamma_k$ is like the responsibility of cluster $k$ towards $\mathbf{x}$; 
# - compute the expected log-likelihood: 
# $\mathrm{E}_{P(z^{(n)}|\mathbf{x}^{(n)})} \left[ \sum\limits_{n=1}^N \ln P(\mathbf{x}^{(n)}, z^{(n)} | \Theta) \right] = \ldots = \sum\limits_{k=1}^K \sum\limits_{n=1}^N \gamma_k^{(n)} \left( \ln \pi_k + \ln \mathcal{N}(\mathbf{x}^{(n)}; \mu_k, \Sigma_k) \right) $; 
# - optimize these $k$ Gaussians with weights $\gamma_k^{(n)}$;
# - re-estimated parameters: 
# $\begin{align} \mu_k &= \frac{1}{N_k} \sum\limits_{n=1}^N \gamma_k^{(n)} \mathbf{x}^{(n)} \\ \Sigma_k &= \frac{1}{N_k} \sum\limits_{n=1}^N \gamma_k^{(n)} (\mathbf{x}^{(n)} - \mu_k) (\mathbf{x}^{(n)} - \mu_k)^\mathsf{T} \\ \pi_k &= \frac{N_k}{N} \\ N_k &= \sum\limits_{n=1}^N \gamma_k^{(n)}\end{align}$
# - evaluate the log-likelihood: 
# $\ell(\mathbf{X}, \Theta) = \ln p(\mathbf{X}|\pi,\mu,\Sigma) = \sum\limits_{n=1}^N \ln \left( \sum\limits_{k=1}^K \pi_k \; \mathcal{N}(\mathbf{x^{(n)}| \mu_k, \Sigma_k }) \right)$.  
# 
# 
# EM + Gaussian mixture model is just like soft k-means clustering
# with fixed priors and covariance.          

# %%[markdown] 
# ## Lecture 18 - Matrix Factorization 
# 
# Recall PCA: 
# - it is a matrix factorization problem;
# - can be extended to matrix completion (data matrix not fully observed);
# - each $N$ $D \times 1$ input vector $\mathbf{x}^{(i)}$ is approximated as 
# $\mathbf{U} \mathbf{z}$;
# - $\mathbf{U}$ is $D \times K$ orthogonal basis matrix and $z$ is the $K \times N$ code vector;  
# - square error: $|| \mathbf{X} - \mathbf{U}\mathbf{Z}||_\textrm{F}^2$; 
# - Frobenius norm, $||\mathbf{A}||_\textrm{F}^2 = \sum_{i,j} a_{i,j}^2$; 
# - result: $\mathbf{X} \approx \mathbf{U} \mathbf{z}$;
# - i.e. optimal low-rank matrix factorization.
#  
# 
# *Singular value decomposition* (SVD): 
# - $\mathbf{X} = \mathbf{U} \mathbf{S} \mathbf{V}^\mathsf{T}$; 
# - $\mathbf{U}, \mathbf{V}$ are unitary matrices (orthonormal columns);
# - $\mathbf{S} = \mathrm{diag}(s_i)$; 
# - the first $k$ SVD vectors correspond to the first $k$ PCA components ($\mathbf{Z} = \mathbf{S} \mathbf{V}^\mathsf{T}$);
# 
# 
# Suppose $\mathbf{X}$ is only partially observed; 
# how to fill in blanks? 
# Application: recommender systems like Youtube and Netflix. 
#  
# Algorithms: 
# - alternating least square (ALS) method;
# - gradient descent;
# - non-negative matrix factorization;
# - low-rank matrix completion;
# - etc.
# 
# 
# Alternating least square (ALS) method:    
# - assume $\mathbf{X}$ is low rank;
# - using squared-error loss;
# - do $\min\limits_{\mathbf{U},\mathbf{Z}} \frac{1}{2} \sum\limits_{x_{i,j} \textrm{observed}} (x_{i,j} - \mathbf{u}_i^\mathsf{T} \mathbf{z}_j)^2$;
# - this objective function is non-convex and this problem is NP-hard;
# - it is convex though as a function of $\mathbf{U}$ or $\mathbf{Z}$ individually;
# - solution: fix $\mathbf{U}$ and optimize $\mathbf{Z}$, vice versa and repeat until convergence$; 
# - algorithm: 
#    1. initialize $\mathbf{U}, \mathbf{V}$ randomly;
#    2. for every $i = 1, \ldots, D$, 
#       do $\mathbf{u}_i = \left( \sum\limits_{j \; : \; x_{ij} \neq 0} z_{j} \mathbf{z}_j^\mathsf{T} \right)^{-1} \sum\limits_{j \; : \; x_{ij} \neq 0} x_{ij} \mathbf{z}_j $
#    3. for every $j = 1, \ldots, N$, 
#       do $\mathbf{z}_i = \left( \sum\limits_{i \; : \; x_{ij} \neq 0} u_{i} \mathbf{u}_i^\mathsf{T} \right)^{-1} \sum\limits_{j \; : \; x_{ij} \neq 0} x_{ij} \mathbf{u}_i $
#    4. repeat until convergence.
# 
# 
# k-means as matrix factorization:
# - define matrix $\mathbf{R}$ where its rows are the indicator vectors $\mathbf{r}$;
# - define matrix $\mathbf{M}$ where it rows are the cluster centres $\mu_k$;
# - reconstruction of the data is thus $\mathbf{X} \approx \mathbf{R} \mathbf{M}$;  
# - k-means distortion function in matrix form: 
#   $\sum\limits_{n=1}^N \sum\limits_{k=1}^K r_k^{(n)} || \mathbf{m}_k - \mathbf{x}^{(n)} ||^2 = || \mathbf{X} - \mathbf{R} \mathbf{M}||_\textrm{F}^2$; 
# 
# 
# *Co-clustering*:
# - co-clustering clusters both the rows and columns of the data matrix $\mathbf{X}$;
# - indicator matrix for rows $\times$ matrix of means for each block $\times$ indicator matrix for columns;
# 
# 
# *Sparse coding*:
# - represent natural data (images and sounds) $\mathbf{x}$ 
# using a dictionary of basis functions $\{\mathbf{a}_k\}_{k=1}^K$;  
# - $\mathbf{x} \approx = \sum\limits_{k=1}^K s_k \mathbf{a}_k = \mathbf{A} \mathbf{s}$;
# - Since only a few basis functions are used, $\mathbf{s}$ is a sparse vector;
# - choose $\mathbf{s}$ with cost function $\min\limits_\mathbf{s} \left( || \mathbf{x} - \mathbf{A}\mathbf{s}||^2 + \beta ||\mathbf{s}||_1 \right)$;
# - learn dictionary by optimizing both $\mathbf{A}$ and $\{\mathbf{s}_k\}_{n=1}^N$
# - i.e. $\min\limits_{\mathbf{A},\{\mathbf{s}_n\}} \left(|| \mathbf{X} - \mathbf{A}\mathbf{S}||_\textrm{F}^2 + \beta ||\mathbf{s}_n||_1 \right)$;
# - subject to $||\mathbf{a}_k||^2 \leq 1 \: \forall \: k$;    
# - fit using ALS;
# - learned dictionary efficiently tiles all degrees of freedom. 
   
# %%[markdown] 
# ## Lecture 19 - Bayesian Linear Regression
# 
# Both parametric and non-parametric models for regression and classification 
# have been covered: 
# - parametric, e.g. linear and logistic regression, neural networks, SVM, naive Bayes, GDA; 
# - non-parametric, e.g. k-nearest neighbour (KNN)
# 
# 
# Next: Bayes linear regression (parametric model).
# 
# Recall linear regression:
# - given training set of inputs and targets 
# $\{(\mathbf{x}^{(n)}, t^{(n)})\}_{n=1}^N$; 
# - linear model, $y = \mathbf{w}^\mathsf{T} \boldsymbol{\psi}(mathbf{x})$; 
# - squared-error loss function, 
# $\mathcal{L}(y, t) = \frac{1}{2} (t - y)^2$;
# - $L^2$ regularization, $\mathcal{R}(\mathbf{w}) = \frac{1}{2} \lambda || \mathbf{w}||^2$;
# - analytical solution (set gradient to $0$): $\mathbf{w} = ( \mathbf{\Psi}^\mathsf{T} \mathbf{\Psi} + \lambda \mathbf{I})^{-1} \mathbf{\Psi}^\mathsf{T} \mathbf{t}$;
# - approximate solution (gradient descent): $\mathbf{w} \leftarrow (1 - \alpha \lambda) \mathbf{w} - \alpha \mathbf{\Psi}^\mathsf{T} (\mathbf{y} - \mathbf{t})$.
# 
# 
# Extensions: 
# - assuming Gaussian noise, 
# we get maximum likelihood method under this model;
# - i.e. $\begin{align} \mathbf{t} | \mathbf{x} & \sim \mathcal{N}(\mathbf{y}(\mathbf{x}), \sigma^2) \\ \log \prod_{n=1}^N p(t^{(n)}|\mathbf{x}^{(n)}; \mathbf{w},b)^{1/N} & = \ldots = \mathrm{const} - \frac{1}{2 N \sigma^2} \sum\limits_{n=1}^N (t^{(n)} - \mathbf{y}(\mathbf{x}))^2 \end{align}$;
# - $L^2$ regularizer can be viewed as MAP inference with Gaussian prior
# - MAP inference: 
# $\begin{align} \arg\max\limits_\mathbf{w} \log p(\mathbf{w}|\mathcal{D}) &= \arg\max\limits_\mathbf{w} \left( \log p(\mathbf{w}) + \log p(\mathcal{D}|\mathbf{w}) \right) \\ \log p(\mathcal{D}|\mathbf{w}) &= \textrm{as above} \end{align}$;
# - Gaussian prior:  
# $\begin{align} \mathbf{w} &\sim \mathcal{N}(\mathbf{m},\mathbf{S}) \\ \log p(\mathbf{w}) &= \ldots \\ &= -\frac{1}{2 \eta} ||\mathbf{w}||^2 + \textrm{const}\end{align}$ 
# 
# 
# Full Bayesian inference:
# - make prediction by averaging over all likely explanations
# under the posterior distribution;
# - compute posterior using Bayes' Rule: 
# $p(\mathbf{w}|\mathcal{D}) \propto p(\mathbf{w}) p(\mathcal{D}|\mathbf{w})$;
# - make prediction using the posterior predictive distribution:    
# $p(t|\mathbf{x},\mathcal{D}) = \int p(\mathbf{w}|\mathcal{D}) p(t|\mathbf{x},\mathbf{w}) \mathrm{d}\mathbf{w}$.
# 
# 
# Bayesian linear regression:
# - make predictions using all possible weights, weighted by their posterior probability;
# - prior distribution: $\mathbf{w} \sim \mathcal{N}(0, \mathbf{S})$;
# - likelihood: $t | \mathbf{x},\mathbf{w} \sim \mathcal{N}(\mathbf{y}(\mathbf{x}), \sigma^2)$;  
# - posterior distribution: 
# $\begin{align} \log p(\mathbf{w}|\mathcal{D}) &= \log p(\mathbf{w} + \log p(\mathcal{D}|\mathbf{w}) + \mathrm{const} \\ &= \ldots \\ &= \log \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma}) \end{align}$;
# - i.e. a multivariate Gaussian distribution, 
# $\begin{align} \boldsymbol{\mu} &= \frac{1}{\sigma^2} \mathbf{\Sigma} \mathbf{\Psi}^\mathsf{T} \mathbf{t} \\ \mathbf{\Sigma}^{-1} &= \frac{1}{\sigma^2} \mathbf{\Psi}^\mathsf{T} \mathbf{\Psi} + \mathbf{S}^{-1} \end{align}$;
# - posterior predictive distribution: 
# $p(t|\mathbf{x},\mathcal{D}) = \int p(\mathbf{w}|\mathcal{D}) p(t|\mathbf{x},\mathbf{w}) \mathrm{d}\mathbf{w} = \int \mathcal{N}(t; \mathbf{y}(\mathbf{x}), \sigma) \mathcal{N}(\mathbf{w}; \mathbf{\mu},\mathbf{\Sigma}) \mathrm{d}\mathbf{w}$;
# - i.e. a Gaussian with parameters 
# $\mu_\textrm{pred} = \mu^\mathsf{T} \boldsymbol{\psi}(\mathbf{x})$ 
# and $\sigma_\textrm{pred}^2 = \boldsymbol{\psi}(\mathbf{x})^\mathsf{T} \mathbf{\Sigma} \boldsymbol{\psi}(\mathbf{x}) + \sigma^2$.    
#
# 
# As more data points $(\mathbf{x}, t)$ are observed,
# the posterior predictive distribution narrows.
# 
# Example use: 
# - *decision theory*: 
# choose a single prediction $y$ to minimize the expected squared-error loss;
# - $\arg \min\limits_y \mathrm{E}_{p(t|\mathbf{x},\mathcal{D})}[(y - t)^2] = \mathrm{E}_{p(t|\mathbf{x},\mathcal{D})}[t]$; 
# 
#     
# 
# *Black-box optimization*: 
# - minimize a function with just queries of function values;
# - i.e. no gradient; 
# - each query could be expensive so few as possible;
# - e.g. minimize validation error of an ML algorithm with respect to its hyperparameters.
# - *Bayesian optimization*: 
#    - approximate the function with simpler functions (*surrogate function*);
#    - condition on the few data points $\in \mathcal{D}$ to infer posterior using Bayesian linear regression;
#    - define an *acquisition function* to choose the next point to query;
#    - desired properties: high for good and uncertain points, low for known points; 
#    - candidates: 
#        1. *probability of improvement* (PI), $\textrm{PI} = \mathrm{Pr}(f(\theta) < \gamma - \epsilon)$;
#        2. *expected improvement* (EI), $\textrm{EI} = \mathrm{E}[\max(\gamma - f(\theta), 0)]$.
#    - maximize the acquisition function using gradient descent;
#    - use random restarts to avoid local maxima. 

# %%[markdown] 
# ## Lecture 20 - Gaussian Processes
# 
# Gaussian processes: 
# - generalization of Bayesian linear regression; 
# - distributions over functions.
# 
# 
# A Bayesian linear regression model defines a distribution over functions: 
# - $f(\mathbf{x}) = \mathbf{w}^\mathsf{T} \mathbf{\psi}(\mathbf{x})$;
# - $\mathbf{w}$ sampled from the prior $\mathcal{N}(\mathbf{\mu}_\mathbf{w}, \mathbf{\Sigma}_\mathbf{w})$; 
# - let $\mathbf{f} = (f_1, \ldots, f_N) = (f(\mathbf{x}_1), \ldots, f(\mathbf{x}_N))$; 
# - by linear transformation of Gaussian random variables, $\mathbf{f}$ is a Gaussian too, with: 
#    - $\mathrm{E}[f_n] = \mathbf{\mu}_\mathbf{w} \mathbf{\psi}(\mathbf{x})$; 
#    - $\mathrm{cov}[f_m, f_n] = \mathbf{\psi}(\mathbf{x}_m) \mathbf{\Sigma}_\mathbf{w} \mathbf{\psi}(\mathbf{x}_n)$; 
# - $\mathbf{f} \sim \mathcal{N}(\mathbf{\mu}_\mathbf{f}, \mathbf{\Sigma}_\mathbf{f})$;
#    - $\mathbf{\mu}_\mathbf{f} = \mathrm{E}[\mathbf{f}] = \mathbf{\Psi} \mathbf{\mu}_\mathbf{w}$;
#    - $\mathbf{\Sigma}_\mathbf{f} = \mathrm{cov}[\mathbf{f}] = \mathbf{\Psi} \mathbf{\Sigma}_\mathbf{w} \mathbf{\Psi}^\mathsf{T}$;
# - assume noisy Gaussian observations, $y_n \sim \mathcal{N}(f_n, \sigma^2)$;
# - i.e. $\mathbf{y} \sim \mathcal{N}(\mathbf{\mu}_\mathbf{y}, \mathbf{\Sigma}_\mathbf{y})$; 
#    - $\mathbf{\mu}_\mathbf{y} = \mathbf{\mu}_\mathbf{f}$;
#    - $\mathbf{\Sigma}_\mathbf{y}) = \mathbf{\Sigma}_\mathbf{f}) + \sigma^2 \mathbf{I}$;
# - let $\mathbf{y}, \mathbf{y}^\prime$ be the training and test data;
#    - both are jointly Gaussian;
#    - $\mathbf{y}^\prime | \mathbf{y} \sim \mathcal{N}(\mathbf{\mu}_{\mathbf{y}^\prime | \mathbf{y}}, \mathbf{\Sigma}_{\mathbf{y}^\prime | \mathbf{y}})$;
#    - $\mathbf{\mu}_{\mathbf{y}^\prime | \mathbf{y}} = \mathbf{\mu}_\mathbf{y} + \mathbf{\Sigma}_{\mathbf{y}^\prime \mathbf{y}} \mathbf{\Sigma}_{\mathbf{y} \mathbf{y}}^{-1} (\mathbf{y} - \mathbf{\mu}_\mathbf{y})$;
#    - $\mathbf{\Sigma}_{\mathbf{y}^\prime | \mathbf{y}} = \mathbf{\Sigma}_{\mathbf{y}^\prime \mathbf{y}^\prime} + \mathbf{\Sigma}_{\mathbf{y}^\prime \mathbf{y}} \mathbf{\Sigma}_{\mathbf{y} \mathbf{y}}^{-1} \mathbf{\Sigma}_{\mathbf{y} \mathbf{y}^\prime}$;
# - then the marginal likelihood is the PDF of a Gaussian, $p(\mathbf{y}|\mathbf{X}) = \mathcal{N}(\mathbf{y}; \mathbf{\mu}_\mathbf{y}, \mathbf{\Sigma}_\mathbf{y})$; 
# - after defining $\mathbf{\mu}_\mathbf{f}, \mathbf{\Sigma}_\mathbf{f}$, we can forget about $\mathbf{w}$.
# 
# 
# Gaussian processes: 
# - specify 
#    - a mean function $\mathrm{E}[f(\mathbf{x}_n)] = \mu(\mathbf{x}_n)$; 
#    - a convariance function (called *kernel function*) $\mathrm{cov}[f(\mathbf{x}_m), f(\mathbf{x}_n)] = k(\mathbf{x}_m, \mathbf{x}_n)$;
# - let $\mathbf{K}_\mathbf{X}$ denote the kernel matrix for points $\mathbf{X}$ (also called the *Gram matrix*);
# - i.e. $(\mathbf{K}_\mathbf{X})_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$; 
# - require that $\mathbf{K}_\mathbf{X}$ be positive semidefinite for any $\mathbf{X}$; 
# - $\mu, k$ can be arbitrary...
# - this defines a distribution over *function values*; 
# - can be extended to a distribution over *functions* using the *Kolmogorov Extension Theorem*; 
# - such distribution over functions is called a *Gaussian process*;
# 
# 
# *Kernel Trick*:
# - many algorithms can be written in terms of dot products between feature vectors  
#   $\langle \mathbf{x}, \mathbf{x}^\prime \rangle = \mathbf{\psi}(\mathbf{x})^\mathsf{T} \mathbf{\psi}(\mathbf{x}^\prime)$;
# - a *kernel* implements an inner product between feature vectors;
# - e.g. feature vector $\phi(\mathbf{x}) = [1 \: \sqrt{2}x_1 \: \ldots \: \sqrt{2}x_d \: \sqrt{2}x_1 x_2 \: \ldots \: \sqrt{2} x_{d-1}x_d \: x_1^2 \ldots x_d^2]$
# - the *quadratic kernel* can compute the inner product in linear time; 
#    - $\begin{align} k(\mathbf{x}, \mathbf{x}^\prime) &= \langle \mathbf{x}, \mathbf{x}^\prime \rangle \\ &= 1 + \sum\limits_{i=1}^d 2 x_i x_i^\prime + \sum\limits_{i,j=1}^d x_i x_j x_i^\prime x_j^\prime \\ &= \left( 1 + \langle \mathbf{x},\mathbf{x}^\prime \rangle \right)^2 \end{align}$;     
# - many algorithms can be written in terms of kernels (*kernelized*);
# - useful composition rules:
#    - $k(\mathbf{x},\mathbf{x}^\prime) = \alpha$ is a kernel;
#    - if $k_1, k_2$ are kernels and $a, b \geq 0$, then $a k_1 + b k_2$ is a kernel too;
#    - $k(\mathbf{x},\mathbf{x}^\prime) = k_1(\mathbf{x},\mathbf{x}^\prime) k_2(\mathbf{x},\mathbf{x}^\prime)$ is also kernel;
# - before artificial neural networks, kernel SVM is the best at classification;   
# 
# 
# Computational cost of the kernel trick:
# - allows the use of very high-dim. feature spaces but at a cost;
# - Bayesian linear regression: need to invert a $d \times d$ matrix;
# - GP regression: need to invert a $N \times N$ matrix;
# - $\mathcal{O}(N^3)$ cost is typical of kernel methods; 
# 
# 
# GP kernels: 
# - define kernel function by giving a set of basis functions 
# and put a Gaussian prior on $\mathbf{w}$;
# - e.g. *squared-exponential* or *Gaussian* or *radial basis function* (RBF) kernel  
#    - $k(\mathbf{x}_i,\mathbf{x}_j) = \sigma^2 \exp \left(- \frac{|| \mathbf{x}_i - \mathbf{x}_j ||^2}{2 \ell^2} \right)$ 
# - this is a *kernel family* with hyperparameters $\sigma, \ell$;
# - $\sigma^2$ is the output variance;
# - $\ell$ is the lengthscale;
# - choice of these hyperparameters heavily affects the predictions; 
# - tune them by e.g. maximizing marginal likelihood;    
# - this kernel is *stationary* since it depends only on $\mathbf{x}_i - \mathbf{x}_j$; 
# - most kernels are stationary;
