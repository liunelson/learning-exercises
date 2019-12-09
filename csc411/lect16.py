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
# *Efficient coding hypothesis*
#  
#  
#        

# %%
