#%% [markdown]
# # Notes and exercises from [Statlect](https://www.statlect.com)
# --- 

#%% [markdown]
# # Part 5 - [Fundamentals of Statistics](https://www.statlect.com/fundamentals-of-statistics/)
#
# ## Statistical Inference
# 
# The act of using observed data (the *sample*) to infer unknown features 
# of the underlying probability distribution. 
# 
# A sample is the vector of realizations $x_1, \ldots, x_n$ 
# of $n$ independent random variables $X_1, \ldots, X_n$ 
# having a common distribution function $F_X(x)$.
# 
# In other words, the sample $\xi = [x_1 \: \ldots \: x_n]$ is 
# the realization of a random vector $\Xi = [X_1 \: \ldots \: X_n]$ 
# with joint CDF $F_\Xi(\xi) = F_X(x_1) \ldots F_X(x_n)$.
#  
# An individual realization $x_i$ is known as an *observation* from the sample.
# 
# A *statistical model* (or *model specification* or just *model*) 
# is a set of joint CDFs to which $F_\Xi(\xi)$ is assumed to belong, 
# i.e. features that $F_\Xi(\xi)$ is assumed to have.
# 
# Example: assuming that all $n$ random variables are mutually independent and have a common CDF, 
# a model would be the subset of joint CDFs wherein all the marginal CDFs are equal 
# and their product is equal to the underlying CDF.
# 
# ### Parametric Models
# - Let $\Psi$ be a model for $\Xi$. is called *parametric* if the joint CDFs belonging to it   
# - Let $\Theta \subseteq \mathbb{p}$ be a set of $p$-dimensiona real vectors.
# - Let $\gamma(\theta)$ be a correspondence that associates a subset of $\Psi$ to each $\theta \in \Theta$.
# - The triple $(\Phi,\Theta,\gamma)$ is a *parametric model* if $\Psi = \bigcup\limits_{\theta \in \Theta} \gamma(\theta)$.
# - $\Theta$ is the *parameter space* and $\theta$ is a *parameter.
# - If $\gamma$ maps each parameter to an unique joint CDF, 
# then $(\Phi,\Theta,\gamma)$ is a *parametric family*.
# - If $\gamma$ is one-to-one (i.e. each CDF is associated with just one parameter), 
# then the parametric family is said to be *identifiable*.
# - Let $\theta_0$ be the parameter that is associated with $F_\Xi(\xi)$,  
# if it is unique, then it is called the *true parameter*.
#
#
# ### Statistical Inferences
# These are statements about the unknown distribution $F_\Xi(\xi)$  
# based on the observed sample $\xi$ and the statistical model $\Psi$.
#   
# They take the form of *model restrictions*.
#
# Given a subset of the original model $\Psi_R \subset \Psi$, 
# such restrictions can be either an inclusion restriction ($F_\Xi \in \Psi_R$)  
# or exclusion restriction ($F_\Xi \notin \Psi_R$).
# 
# Common statistical inferences:
# 1. In *hypothesis testing*, a restriction $F_\Xi \in \Psi_R$ is proposed 
# for either rejection or otherwise.
# 2. In *estimation*, some restriction must be chosen among a set of many.
# 3. In *Bayesian inference*, the observed sample $\xi$ is used to update 
# the subjective probability that the restriction is true.
# 

# %%[markdown]
# ## Point Estimation
# 
# *Point estimation* is the act of choosing a parameter $\hat{\theta} \in \Theta$ 
# to be the best guess of the unknown true parameter $\theta_0$; 
# $\hat{\theta}$ is called an *estimate* of $\theta_0$.
# 
# An *estimator* $\hat{\theta}(\xi)$ is a function that produces 
# a parameter estimate $\hat{\theta}$ from each sample $\xi in \Xi$.
# 
# *Estimation error*:  $\epsilon = \hat{\theta} - \theta_0$ 
# 
# A *loss function* $L(\hat{\theta}, \theta_0)$ maps $\Theta \times \Theta$ into $\mathbb{R}$, 
# quantifying the loss incurred by estimating $\theta_0$ with $\hat{\theta}$.
# 
# Examples:
#
# 1. *Absolute error*, $L(\hat{\theta}, \theta_0) = | \hat{\theta} - \theta_0 |$.
# 2. *Square error*, $L(\hat{\theta}, \theta_0) = | \hat{\theta} - \theta_0 |^2$.
# 
# 
# When an estimator is used, 
# $L(\hat{\theta}(\Xi), \theta_0)$ can be considered to be a random variable.
# Its expected value is called the *statistical risk* and is denoted by 
# $R(\hat{\theta}) = \mathrm{E}[L(\hat{\theta}(\Xi), \theta_0)]$.
# 
# *Mean absolute error*: $R(\hat{\theta}) = \mathrm{E}[| \hat{\theta} - \theta_0 |]$
# 
# *Mean square error* (MSE): $R(\hat{\theta}) = \mathrm{E}[| \hat{\theta} - \theta_0 |^2]$
# 
# *Root mean square error* (RMSE): $R(\hat{\theta}) = \sqrt{\mathrm{E}[| \hat{\theta} - \theta_0 |^2]}$
# 
# Bias-variance decomposition:
# $\mathrm{MSE}(\hat{\theta}) = \mathrm{tr}(\mathrm{var}[\hat{\theta}]) + |\mathrm{Bias}[\hat{\theta}]|^2$
#  
# Criteria to evaluate estimators:
#
# 1. Unbiasedness, $\hat{\theta}$ is *unbiased* if $\mathrm{E}[\hat{\theta}(\Xi)] = \theta_0$ 
# (estimator produces estimated values that are correct on average).
#
# 2. Consistency, $\hat{\theta}$ is *weakly/strongly consistent* if 
# the sequence of estimators produced by a sequence of samples $\{xi_n\}$ converges 
# in probability/almost surely to the true parameter $\theta_0$.
# 

# %%[markdown]
# ## Mean Estimation
# 
# Consider the sample $\xi_n = [x_1 \: \ldots \: x_n]$ made up of $n$ independent draws 
# from a probability distribution with unknown mean $\mu$ and variance $\sigma^2$.
# Therein is $n$ realizations $x_i$ of $n$ independent random variables $X_i$ 
# all having the same distribution.
# 
# An estimator of $\mu$ is the sample mean: 
# $\hat{\mu} = \bar{X}_n = \frac{1}{n} \sum\limits_{i=1}^n X_i$.
# 
# $\hat{\mu}$ is unbiased since 
# $\mathrm{E}[\hat{\mu}] = \ldots = \mu$.
# 
# The variance of $\hat{\mu}$: 
# $\mathrm{var}[\hat{\mu}] = \ldots = $\frac{\sigma^2}{n}$.
# 
# The risk/MSE of $\hat{\mu}$: 
# $\mathrm{MSE}(\hat{\mu}) = \ldots = \mathrm{var}[\hat{\mu}]$.
# 
# Since $\{X_n\}$ is an IID sequence with finite $\mu$ and $\sigma^2$, 
# the sample mean $\bar{X}_n$ is asymptotically normal.
# 
# ## Variance Estimation
# 
# An estimator of the variance:
# $\widehat{\sigma^2} = \frac{1}{n}\sum\limits_{i=1}^n (X_i - \mu)^2$.
# 
# This estimator is unbiased:
# $\mathrm{E}[\widehat{\sigma^2}] = \ldots = \sigma^2$.
# 
# The variance of $\widehat{\sigma^2}$ goes to zero as $n$ approaches infinity: 
# $\mathrm{var}[\widehat{\sigma^2}] = \ldots = $\frac{2 \sigma^4}{n}$.
# 
# $\widehat{\sigma^2}$ has a gamma distribution with parameters $n$ and $\sigma^2$.
# 
# The risk/MSE of $\widehat{\sigma^2}$:
# $\mathrm{MSE}(\hat{\mu}) = \ldots = \mathrm{var}[\widehat{\sigma^2}]$.
# 
# If the mean is unknown, two other estimators can be used: 
# $\widehat{\sigma^2} = \begin{cases} S_n^2 = \frac{1}{n} \sum\limits_{i=1}^n (X_i - \bar{X}_n)^2 & \textrm{unadjusted sample variance} \\ s_n^2 = \frac{1}{n-1} \sum\limits_{i=1}^n (X_i - \bar{X}_n)^2 & \textrm{adjusted sample variance} \end{cases}$       
# 
# The *unadjusted sample variance* $S_n^2$ is a biased estimator of the true variance $\sigma^2$: 
# $\mathrm{E}[S_n^2] = \ldots = \frac{n-1}{n} \sigma^2$. 
# 
# The *adjusted sample variance* $s_n^2$ is unbiased though:
# $\mathrm{E}[s_n^2] = \ldots = \sigma^2$. 
# 
# The sum of squared deviations from the true mean is always larger than that from the sample mean; 
# the $n-1$ factor exactly corrects for this bias.
# 
# $n-1$ is called the *number of degrees of freedom*; 
# it is the number os sample points ($n$) minus the number of parameters to be estimated 
# ($1$ for the true mean $\mu$).
# 
# The variance of these estimators is: 
# $\begin{align} \mathrm{E}[S_n^2] &= \frac{n-1}{n} \frac{2\sigma^4}{n} \\ \mathrm{E}[s_n^2] &= \frac{2\sigma^4}{n-1} \end{align}$ 
# 

# %%[markdown]
# ## Set Estimation
#
# A *set estimation* is the act of choosing a subset $T$ of the parameter space $\Theta$ 
# such that $T$ has some high probability (called *coverage probability* $C$) 
# of containing the true and unknown parameter $\theta_0$.
# 
# Such subset $T$ is called a *set estimate* of (or *confidence set* for) $\theta_0$.  
# 
# If $\Theta \subseteq \mathbb{R}$, 
# $T = [a, b]$ is called an *interval estimate* or *confidence interval*.
# 
# If $T = T(\xi)$, it is called a *set estimator*.
# 
# The coverage probability is defined as $C(T,\theta_0) = P_{\theta_0}(\theta_0 \in T(\Xi))$  
# where the notation indicates that it is calculated using the CDF $F_\Xi(\xi; \theta_0)$ 
# associated with $\theta_0$.
#  
# Since $C$ is rarely known, the *confidence coefficient* (or *level of confidence*) 
# is calculated: 
# $c(T) = \mathrm{inf}\limits_{\theta \in \Theta} C(T,\theta)$. 
# 
# The size of a confidence set is called its *measure* 
# (as in Lebesgue measure, the generalization of volume to higher dimensions).
# 
# ## Set Estimation of the Mean
# 
# For a normal IID sample with *unknown mean* and *known variance*: 
# - The estimator of the true mean is the sample mean, $\hat{\mu} = \bar{X}_n$. 
# - The interval estmator is then $T_n = \left[\bar{X}_n - \sqrt{\frac{\sigma^2}{n}z}, \bar{X}_n + \sqrt{\frac{\sigma^2}{n}z} \right]$ 
# where $z \in \mathbb{R}_{++}$ is some constant.
# - The coverage probability of $T_n$ is 
# $C(T_n;\mu) = P(-z \leq Z \leq z)$ where $Z$ is a standard normal random variable.
# - Since $C$ does not depend on the unknown $\mu$, $c(T_n) = C(T_n;\mu)$.
# - Size of $T_n$: $\lambda(T_n) = 2 \sqrt{\frac{\sigma^2}{n}}z$.
# 
# 
# If *unknown variance*: 
# - Using the adjusted sample variance as the variance estimator,  
# $T_n^{(a)} = \left[\bar{X}_n - \sqrt{\frac{s_n^2}{n}z}, \bar{X}_n + \sqrt{\frac{s_n^2}{n}z} \right]$     
# - The coverage probability:
# $C(T_n^{(a)};\mu,\sigma^2) = P(-z \leq Z_{n-1} \leq z)$ 
# where $Z$ is a standard Student's t random variable.
# - The confidence coefficient: 
# $c(T_n^{(a)}) = C(T_n^{(a)};\mu, \sigma^2)$
# - The size: $\lambda(T_n^{(a)}) = 2 \sqrt{\frac{s_n^2}{n}}z$
# - The expected size: $\mathrm{E}[\lambda(T_n^{(a)})] = \sqrt{\frac{2}{n-1}} \frac{\Gamma(n/2)}{\Gamma((n-1)/2)} 2 \sqrt{\frac{\sigma^2}{n}}z$
# 
# 
# ## Set Estimation of the Variance
# 
# For a normal IID sample with *known mean* and *known variance*:  
# - The interval estimator: $T_n = [\frac{n}{z_2}\widehat{\sigma_n^2}, \frac{n}{z_1}\widehat{\sigma_n^2}]$ 
# where $\widehat{\sigma_n^2} = \frac{1}{n}\sum\limits(X_i - \mu)^2$ 
# and $z_1 < z_2$ are strictly positive constants.
# - Coverage probability: 
# $C(T_n;\sigma^2) = P(\sigma \in T_n) = P(z_1 \leq Z \leq z_2)$ 
# where $Z$ is a chi-square random variable with $n$ degrees of freedom.
# - Confidence coefficient:
# $c(T_n) = C(T_n;\sigma^2)$
# - Size of the interval estimator: 
# $\lambda(T_n) = \ldots = n \left(\frac{1}{z_1} - \frac{1}{z_2} \right) \widehat{\sigma_n^2}$.
# - Expected size: $\mathrm{E}[\lambda(T_n)] = \ldots = n \left(\frac{1}{z_1} - \frac{1}{z_2} \right) \sigma^2$.
#
# 
# If *unknown mean*:
# - Using the adjusted sample variance as the variance estimator,  
# $T_n = [\frac{n-1}{z_2}\widehat{s_n^2}, \frac{n-1}{z_1}\widehat{s_n^2}]$
# - Coverage probability: 
# $C(T_n;\mu,\sigma^2) = P(\sigma \in T_n) = P(z_1 \leq Z_{n-1} \leq z_2)$ 
# where $Z_{n-1}$ is a chi-square random variable with $n-1$ degrees of freedom.
# - $\ldots$
# 

# %%[markdown]
# ## Hypothesis Testing 
# 
# Consider a random vector $\Xi$ with support $R_\Xi$ 
# and unknown joint CDF $F_\Xi(\xi)$, 
# where $\xi$ is a realization of $\Xi$ (i.e. a sample) 
# and $F_\Xi$ is assumed to belong to a set $\Phi$ (i.e. the statistical model).
#   
# Null hypothesis:
# - Make a statement (statistical inference) about a model restriction $\Phi_R \subset \Phi$.
# - Two options: (1) reject or (2) do not reject the restriction $F_\Xi \in \Phi_R$. 
# - If parametric model: (1) reject or (2) do not reject the restriction $\theta_0 \in \Theta_R$.
# - The *null hypothesis* (denoted $H_0$) is that the restriction is true ($H_0: \theta_0 \in \Theta_R$).
# 
# 
# *Alternative hypothesis*: $H_1: \theta_0 \in \Theta_R^\complement$  
# 
# Types of errors:
#  
# 1. *Type I error*, rejecting $\theta_0 \in \Theta_R$ when it is true. 
# 
# 2. *Type II error*, not rejecting $\theta_0 \in \Theta_R$ when it is false.
# 
# 
# Critical region:
# - A test of hypothesis divides $R_\Xi$ into two disjoint subsets: $C_\Xi \cup C_\Xi^\complement = R_\Xi$.  
# - The set of all $\xi$ for which the null hypothesis is rejected, 
# denoted $C_\Xi$, is called the *critical region* (or *rejection region*), 
# - $C_\Xi = \{\xi \in R_\Xi \: : \: H_0 \textrm{ is rejected whenever the sample is observed} \}$.
# 
# 
# Test statistics: 
# - A critical region can be implicitly defined in terms of a *test statistics*.
# - A test statistics is a random variable $S$ whose realization is a function of the sample $\xi$, 
# $S = s(\Xi)$.
# - A critical region for $S$ is a subset $C_S \subset \mathbb{R}$ such that  
# $s(\xi) \in C_S \; \Rightarrow \; \xi \in C_\Xi \; \Rightarrow \; H_0 \textrm{ is rejected}$.
# - Conversely, $s(\xi) \notin C_S \; \Rightarrow \; \xi \notin C_\Xi \; \Rightarrow \; H_0 \textrm{ is not rejected}$. 
# - If $C_S^\complement = [a, b]$, then $a, b$ are called *critical values* of the test.
# - A hypothesis test has a function known as its *power function* $\pi(\theta)$ 
# that associates the probability of rejecting $H_0$ to each parameter $\theta \in \Theta$, 
# i.e. $\pi(\theta) = P_\theta(\Xi \in C_\Xi)$. 
# - When $\theta \in \Theta_R$, $\pi(\theta)$ describes the probability of a Type I error. 
# - The maximum such probability is the *size* or *level of significance* 
# of the test (denoted $\alpha$), 
# $\alpha = \sup\limits_{\theta \in \Theta_R} \pi(\theta)$.
# 
# 
# The ideal test should have size $0$ 
# (no probability of rejecting the null hypothesis when it is true) 
# and power $1$ when $\theta_0 \notin \Theta_R$ 
# (guaranteed to reject the null hypothesis when it is false).
# 
# 
# %%[markdown]
# ## Hypothesis Tests about the Mean
# 
# Assuming normal IID samples 
# (*unknown mean* $\mu$ and *known variance* $\sigma^2$).
# 
# Let's test the null hypothesis $H_0$ that $\mu = \mu_0$ 
# for some specific value $\mu_0 \in \mathbb{R}$. 
# 
# The alternative hypothesis: $H_1 : \mu \neq \mu_0$.
# 
# Consider the sample mean $\bar{X}_n = \frac{1}{n} \sum\limits_{i=1}^n X_i$.
# 
# A test statistics called *normal z-statistics* is defined as 
# $Z_n = \frac{\bar{X}_n - \mu_0}{\sqrt{\sigma^2/n}}$.
# The resulting test is called a *normal z-test*.
# 
# Let $z \in \mathbb{R}_{++}$. 
# Let's reject $H_0$ if $|Z_n| > z$,  
# i.e. the critical region is $C_{Z_n} = (-\infty,-z) \cup (z,\infty)$ 
# with critical values $\pm z$.
# 
# The power function of the test is 
# $\pi(\mu) = P_\mu(Z_n \notin [-z, z]) = 1 - P\left( -z + \frac{\mu_0 - \mu}{\sqrt{\sigma^2/n}} \leq Z \leq z + \frac{\mu_0 - \mu}{\sqrt{\sigma^2/n}} \right)$
# where $Z$ is a standard normal random variable.
# 
# The size of the test is just 
# $\alpha = \pi(\mu_0) = P_{\mu_0}(Z_n \notin [-z, z]) = 1 - P(-z \leq Z \leq z)$.     
# 
# 

# %%[markdown] 
# 
# Let's assume now *unknown mean* $\mu$ 
# and *unknown variance* $\sigma^2$.
# 
# Define a new test statistics $Z_n^{(a)}$ using the sample mean $\bar{X}_n$
# and the adjusted sample variance: 
# $Z_n^{(a)} = \frac{\bar{X}_n - \mu_0}{\sqrt{s_n^2/n}}.
# 
# This is called *Student's t-statistics* 
# and the resulting test is called a *Student's t-test*.
# 
# As before, 
# the critical region is $C_{Z_n^{(a)}} = (-\infty,-z) \cup (z,\infty)$ 
# with critical values $\pm z$.
# 
# The power function of the test is 
# $\pi^{(a)}(\mu) = P_\mu(Z_n^{(a)} \notin [-z, z]) = 1 - P\left( -\sqrt{\frac{n-1}{n}} z \leq W_{n-1} \leq \sqrt{\frac{n-1}{n}} z \right)$
# where $W_{n-1}$ is a non-central standard Student's t random variable 
# with $n-1$ degrees of freedom and $c = \frac{\mu - \mu_0}{\sqrt{\sigma^2/n}}$.
# 
# The size of the test is just 
# $\alpha = \pi^{(a)}(\mu_0) = 1 - P(-z \leq W_{n-1} \leq z)$.     
# 

# %%[markdown]
# ## Hypothesis Tests about the Mean
# 
# Assuming normal IID samples 
# (*known mean* $\mu$ and *unknown variance* $\sigma^2$). 
# 
# Null hypothesis $H_0: \sigma^2 = \sigma_0^2$.
# 
# Set the test statistic to be $\chi_n^2 = \frac{n}{\sigma_0^2} \widehat{\sigma_n^2}$ 
# where $\widehat{\sigma_n^2} = \frac{1}{n}\sum\limits_{i=1}^n (X_i-\mu)^2$.
# 
# This test statistic is called *chi-square statistic* 
# and the test is called a *chi-square test*.
# 
# Define the critical region as $C_{\chi_n^2} = [0,z_1) \cup (z_2,\infty]$ 
# where $z_1 < z_2$ and $z_i \in \mathbb{R}_{++}$ are the critical values.
# 
# The power function of the test: 
# $\pi(\sigma^2) = P_{\sigma^2}(\chi_n^2 \notin [z_1,z_2]) = 1 - P \left( \frac{\sigma_0^2}{\sigma^2}z_1 \leq \kappa_n \leq \frac{\sigma_0^2}{\sigma^2}z_2 \right)$.
# 
# $\kappa_n$ is a chi-square random variable with $n$ degrees of freedom.
# 
# The size is just $\alpha = \pi(\sigma_0^2) = 1 - P(z_1 \leq \kappa_n \leq z_2)$.
#  
#  
# If *unknown mean* $\mu$ and *unknown variance* $\sigma^2$.
# 
# The test statistic becomes $\chi_n^2 = \frac{n-1}{\sigma_0^2} \widehat{s_n^2}$ 
# where $s_n$ is the adjusted sample variance.
# 
# Everything else is as before.
#  

# %%[markdown]
# ## Estimation Methods
# 
# Previously, the concept of estimators was defined.
#  
# Let's discuss methods to derive them.
# 
# A estimator $\hat{\theta}$ is an *extremum estimator* 
# if it can be represented as a solution of the optimization problem 
# $\hat{\theta} = \arg \max\limits_{\theta \in \Theta} Q(\theta, \xi)$ 
# where $Q$ is some function of parameter $\theta$ and sample $\xi$.
# 
# In *maximum likelihood* (ML) estimation, 
# $\hat{\theta}$ is the ML estimator of $\theta$ 
# and we maximize the likelihood of the sample, 
# i.e. $Q(\theta, \xi) = L(\theta; \xi)$:
# 
# 1. If $\Xi$ is discrete, $L(\theta;\xi) = p_\Xi(\xi;\theta)$ 
# is the joint PMF of $\xi$ associated with the distribution corresponding to $\theta$ for fixed $\xi$.
# 
# 2. If $\Xi$ is continuous, $L(\theta;\xi) = f_\Xi(\xi;\theta)$ 
# is the joint PDF of $\Xi$ associated with the distribution corresponding to $\theta$ for fixed $\xi$.
# 
# 
# In *generalized method of moments* (GMM) estimation, 
# $Q(\theta, \xi) = -d(G(\theta; \xi),0)$.
# 
# In *nonlinear least squares* (NLS) estimation,  
# the sample is $n$ realizations $y_i$ of $Y_i$ and $x_i$ of $X_i$. 
#  
# The NLS estimator is an extremum estimator since it can be written with 
# $Q(\theta, \xi) = -\sum\limits_{i=1}^n \left(y_i - g(x_i;\theta) \right)^2$.
# 

# %%[markdown]
# ## Maximum Likelihood
# 
# ML estimation allows the use of a sample to estimate the parameters of 
# the probability distribution that generated the sample.
# 
# A ML estimator of $\theta_0$ is 
# $\hat{\theta} = \arg \max\limits_{\theta \in \Theta} L(\theta;\xi)$.
# 
# The same value of $\hat{\theta}$ can be obtained if 
# $L(\theta;\xi)$ is replaced by $l(\theta;\xi) = \ln(L(\theta;\xi))$,  
# where $l$ is called the *log-likelihood*.
# 
# It can be shown that $\hat{\theta}$ is asymptotically normal around $\theta_0$: 
# $\sqrt{n}(\hat{\theta}_n - \theta_0) \overset{d}{\to} \mathcal{N} \left( 0, \frac{1}{\mathrm{var}[\left. \nabla_\theta \: l(\theta; X) \right|_{\theta = \theta_0} ]} \right)$.
# 
# i.e. $\hat{\theta}_n \overset{d}{\to} \mathcal{N} \left( \theta_0, \frac{1}{n} \frac{1}{\mathrm{var}[\left. \nabla_\theta \: l(\theta; X) \right|_{\theta = \theta_0} ]} \right)$ 
# 
# *Information equality*: 
# $\mathrm{var}[\left. \nabla_\theta \: l(\theta; X) \right|_{\theta = \theta_0} ] = - \mathrm{E}[\left. \nabla_{\theta\theta} \: l(\theta;X) \right|_{\theta = \theta_0}]$
# 
# Note that:
# - Left is the covariance matrix of the *score vector* (*Fisher information matrix*).   
# - Right is the negative of expected value of the Hessian of the log-likelihood.
# - The score vector is the gradient of the log-likelihood.
# - It can be proved that the expected value of the score is equal to $0$. 
# 

# %%[markdown]   
# ### ML Estimation for a Poisson Distribution
# 
# Assume:
# - An IID sequence $\{X_n\}$ made up of $n$ independent draws from a Poisson distribution.
# - The PMF is $p_X(x_i) = \begin{cases} \mathrm{e}^{-\lambda_0} \frac{1}{x_i} \lambda_0^{x_i} & x_i \in R_X \\ 0 & \textrm{otherwise} \end{cases}$ 
# where $R_X = \mathbb{Z}_+$.
# 
#  
# By definition, the likelihood and log-likelihood functions are: 
# $\begin{align} L(\lambda;x_1,\ldots,x_n) &= \prod\limits_{i=1}^n p_X(x_i) \\ l(\lambda;x_1,\ldots,x_n) &= \ln(L(\lambda;x_1,\ldots,x_n)) \\ &= -n\lambda - \sum\limits_{i=1}^n \ln(x_i !) + \ln(\lambda) \sum\limits_{i=1}^n x_i \end{align}$
# 
# The ML estimator of $\lambda$ is obtained as follows:
# - By definition, $\hat{\lambda} = \arg \max\limits_\lambda l(\lambda;x_1,\ldots,x_n)$
# - A 1st-order condition for an extremum is $\begin{align} \frac{\mathrm{d}}{\mathrm{d}\lambda} l(\lambda;x_1,\ldots,x_n) &= 0 \\ -n + \frac{1}{\lambda} \sum\limits_{i=1}^n x_i &= 0 \\ \lambda &= \frac{1}{n}\sum\limits_{i=1}^n x_i \end{align}$ 
# - Thus, $\hat{\lambda} = \frac{1}{n}\sum\limits_{i=1}^n x_i$.
# - This is just the sample mean $\bar{x}_n$.
# 
# 
# Asymptotical behaviour:
# $\hat{\theta}_n \overset{d}{\to} \mathcal{N} \left(\lambda_0, \frac{\lambda_0}{n} \right)$.    
# 

# %%[markdown]   
# ### ML Estimation for a Normal Distribution
# 
# Same as before: derive $\hat{\theta} = \begin{bmatrix} \hat{\mu}_n \\ \widehat{\sigma^2} \end{bmatrix}$ 
# from log-likelihood function and ML definition.
# 
# $\hat{\mu}_n = \ldots = \frac{1}{n} \sum\limits_{i=1}^n x_i = \bar{x}_n$, i.e. the sample mean.
# 
# $\widehat{\sigma^2}_n = \ldots = \frac{1}{n} \sum\limits_{i=1}^n (x_i - \hat{\mu}_n)^2 = S_n$, i.e. the unadjusted sample mean.
# 
# Asymptotical behaviour:
# $\hat{\theta}_n = \begin{bmatrix} \hat{\mu}_n \\ \widehat{\sigma^2} \end{bmatrix} \overset{d}{\to} \mathcal{N} \left( \begin{bmatrix} \lambda_0 \\ \sigma_0^2 \end{bmatrix}, \begin{bmatrix} \sigma_0^2/n & 0 \\ 0 & 2 \sigma_0^4/n \end{bmatrix} \right)$.
# 

# %%[markdown]   
# ### ML Estimation for a Normal Linear Regression Model
#
# Objective: estimate the parameters of the linear regression model 
# $y_i = x_i \beta_0 + \epsilon_i$ 
# where $x_i$ is the $1 \times K$ vector of regressors, 
# $\beta_0$ is the $K \times 1$ vector of regression coefficients 
# and $\epsilon_i$ is an unobservable error term.
# 
# In matrix form, $y = X \beta_0 + \epsilon$ 
# where $y$ is the $N \times 1$ vector of observations, 
# $N \times K$ is the matrix of regressors, 
# $\epsilon$ is $N \times 1$ vector of error terms.
# 
# Assume $\epsilon$ has a multivariate normal distribution conditional on $X$ with mean $0$ 
# and covariance $\sigma_0^2 I_N = \mathrm{var}[\epsilon_i | X] I_N$.   
# 
# Thus, $f_Y(y_i|X) = \frac{1}{\sqrt{2 \pi \sigma_0^2}} \mathrm{e}^{-(y_i - x_i \beta_0)^2 /(2 \sigma_0^2)}$
#  
# The parameters to be estimated are just $\beta, \sigma^2$.
# 
# The log-likelihood function: 
# $l(\beta,\sigma^2; y, X) = \ln(f_Y(y_i|X)) = \ldots$
# 
# Solving the ML equation: 
# - $\hat{\beta}_N = (X^\mathsf{T} X)^{-1} X^\mathsf{T} y$
# - $\widehat{\sigma^2}_N = \frac{1}{N} \sum\limits_{i = 1}^N (y_i - x_i \hat{\beta}_N)^2$
# 
#  
# $\hat{\beta}_N$ is just the usual OLS estimator 
# and $\widehat{\sigma^2}_N$ is the unadjusted sample variance of the residuals $e_i = y_i - x_i \hat{\beta}_N$.
# 
# Asymptotical behaviour:
# $\hat{\theta}_N = \begin{bmatrix} \hat{\beta}_N \\ \widehat{\sigma^2}_N \end{bmatrix} \overset{d}{\to} \mathcal{N} \left( \begin{bmatrix} \beta_0 \\ \sigma_0^2 \end{bmatrix}, \begin{bmatrix} \frac{\sigma_0^2}{n} (\mathrm{E}[x_i^\mathsf{T} x_i])^{-1} & 0 \\ 0 & 2 \frac{\sigma_0^4}{n} \end{bmatrix} \right)$.
# 

# %%[markdown]
### ML Estimation Algorithm
# 
# The ML optimization problem: $\hat{\theta} = \arg \max\limits_{\theta \in \Theta} \ln \left( L(\theta; \xi) \right)$
# 
# where: 
# - $\Theta$ is the parameter space;
# - $\xi$ is the sample and the realization of the random vector $\Xi$;
# - $L(\theta;\xi)$ is the likelihood of the sample, 
# i.e. the joint PMF $p_\Xi(\xi;\theta)$ or PDF $f_\Xi(\xi;\theta)$ where $\xi$ is fixed; 
# - $\hat{\theta}$ is the estimator and the value for which the log-likelihood is maximized.
# 
# 
# In some special cases, this problem could be solved analytically, 
# i.e. $\hat{\theta}$ can be written as a function of $\xi$.
# In others, numerical algorithms are necessary.
# 
# Avoid constrained optimization. 
# One way is to reparameterize: 
# - $\theta \in (0, \infty)$ becomes 
# $\theta^\prime = \mathrm{e}^\theta$.
# - $\theta \in (0, 1)$ becomes $\theta^\prime = \frac{\mathrm{e}^\theta}{1 + \mathrm{e}^\theta}$.
# 
# 
# Another way is to use a *penalty function* $p$:
# - Given the constrained problem $\hat{\theta} = \arg \max\limits_{\theta \in \Theta} \ln \left( L(\theta; \xi) \right)$,
# - solved the associated unconstrained problem $\hat{\theta} = \arg \max\limits_{\theta \in \mathbb{R}^k} \left[ \ln \left( L(\theta; \xi) \right) - p(\theta) \right]$,
# - where $p(\theta) = \begin{cases} 0 & \theta \in \Theta \\ \infty & \textrm{otherwise} \end{cases}$.    
# 
#  
# ### ML Hypothesis Testing
# 
# Assume that some unknown parameter $\theta \in \Theta$ ($\Theta \subseteq \mathbb{R}^p$) 
# with true value $\theta_0$ has been estimated by ML methods.
# 
# We now want to test the null hypothesis $H_0: \theta_0 \in \Theta_R$ 
# where $\Theta_R \subset \Theta$.
#
# Popular tests: 
# 
# 1. *Wald test* (uses the unrestricted estimate $\hat{\theta}$)
# 
# 2. *Score test* (uses the restricted estimate $\hat{\theta}^R$)
# 
# 3. *Likelihood ratio test* (uses both estimates)
# 

# %%[markdown]
# ### Wald Test
# 
# Let $\hat{\theta}_n$ be the estimator/estimate of a $p \times 1$ parameter $\theta_0$ 
# obtained by ML estimation 
# ($\hat{\theta}_n = \arg \max\limits_{\theta \in \Theta} \ln \left( L(\theta; \xi_n) \right)$).
# with sample size $n$. 
# 
# The Wald statistic is 
# $W_n = n\: g(\hat{\theta}_n)^\mathsf{T} \: \left[ J_g(\hat{\theta}_n) \hat{V}_n J_g(\hat{\theta}_n)^\mathsf{T} \right]^{-1} g(\hat{\theta}_n)$.
# 
# $g: \mathbb{R}^p \rightarrow \mathbb{R}^{r}$ is a vector-valued function such that  
# $\Theta_R = \{ \theta \in \Theta \: : \: g(\theta) = 0 \}$.
#
# $J_g(\theta)$ is the Jacobian of $g$ (the $r \times p$ matrix of partial derivatives of $g$ 
# with respect to $\theta$).
#   
# $\hat{V}_n$ is a consistent estimate of the asymptotic covariance matrix of $\hat{\theta}_n$, 
# 
# There are three estimates of $\hat{V}_n$:
#  
# 1. *Outer product of gradients* (OPG) estimate, 
# $\hat{V}_n^\textrm{O} = \left(\frac{1}{n} \sum\limits_{i=1}^n \nabla_\theta \ln( f_X(x_i;\hat{\theta}_n)) \: \nabla_\theta \ln( f_X(x_i;\hat{\theta}_n))^\mathsf{T} \right)^{-1}$
# 2. *Hessian* estimate,
# $\hat{V}_n^\textrm{H} = \left(-\frac{1}{n} \sum\limits_{i=1}^n \nabla_{\theta,'theta}\ln( f_X(x_i;\hat{\theta}_n)) \right)^{-1}$ 
# 3. *Sandwich* estimate, 
# $\hat{V}_n^\textrm{S} = \hat{V}_n^\textrm{H} \left( \hat{V}_n^\textrm{O} \right)^{-1} \hat{V}_n^\textrm{H}$
# 
# 
# Under the null hypothesis $H_0$, i.e. $g(\theta_0) = 0$, $W_n$ converges in distribution 
# to a chi-square distribution with $r$ DOFs.
# 
# The Wald test is performed by fixing some critical value $z$ and 
# rejecting $H_0$ is $W_n > z$.
# 
# The size of test is $\alpha = P(W_n > z) = 1 - P(W_n \leq z) \approx 1 - F(z)$, 
# where $F$ is the CDF of a chi-square distribution with $n$ DOFs.
# 
# Alternatively, we set $z = F^{-1}(1 - \alpha)$.
# 
# Example:
# - Let $\Theta = \mathbb{R}^2$ and sample size $n = 90$.
# - Suppose $\hat{\theta}_n = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$ 
# and $\hat{V}_n = \begin{bmatrix} 10 & 5 \\ 5 & 10 \end{bmatrix}$.
# - Test restriction $\theta_{0, 1} + \theta_{0, 2} = 10$.
# - Then, $g : \mathbb{R}^2 \rightarrow \mathbb{R}^1$ is $g(\theta) = g(\theta_1, \theta_2) = \theta_1 + \theta_2 - 10$.      
# and $r = 1$.
# - The Jacobian of $g$ is $J_g(\theta) = \left[\frac{\partial g}{\partial \theta_1} \: \frac{\partial g}{\partial \theta_2} \right] = [1 \: 1]$.
# - $g(\hat{\theta}_n) = 5 + 6 - 10 = 1$ and $J_g(\hat{\theta}_n) = [1 \: 1]$.
# - The Wald statistic is thus $W_n = \ldots = 3$.
# - Suppose we want size $\alpha = 5%$.
# - Then, the critical value is $z = F^{-1}(1 - \alpha) = 3.84$ 
# where $F(z)$ is the chi-square CDF with $1$ DOF.
# - Since $W_n < z$, the null hypothesis is not rejected.

# %%[markdown]
# ### Score Test
# 
# Define $\hat{\theta}_n^R$ as the estimate obtained by ML estimation 
# over the restricted parameter space $\Theta_R$.
# 
# The score or *Lagrange multiplier* (LM) statistic is 
# $\textrm{LM}_n = \frac{1}{n} \nabla_\theta \left( \ln \left[L(\hat{\theta}_n^R;\xi_n )\right] \right)^\mathsf{T} \: \hat{V}_n \: \nabla_\theta \left( \ln \left[L(\hat{\theta}_n^R;\xi_n )\right] \right)$.
#  
# $\nabla_\theta \left( \ln \left[L(\hat{\theta}_n^R;\xi_n )\right] \right)$ 
# is the score, i.e. the gradient of the log-likelihood function.
# 
# Same convergence as Wald statistic.
# 
# The test is performed as in the Wald test.    

# %%[markdown]
# ### Likelihood Ratio Test
# 
# Define $\hat{\theta}_n$ and $\hat{\theta}_n^R$ as before.
# 
# The likelihood ratio statistic is 
# $\textrm{LR}_n = \ln \left[ \left( \frac{L(\hat{\theta}_n;\xi_n )}{L(\hat{\theta}_n^R;\xi_n )} \right)^2 \right] = 2 \left( \ln \left[L(\hat{\theta}_n;\xi_n )\right] - \ln \left[L(\hat{\theta}_n^R;\xi_n )\right] \right)$
# 
# Same convergence as Wald statistic.
# 
# The test is performed as in the Wald test.
# 
  
# %%[markdown]
# ### Model Selection Criteria
# 
# Consider, for some sample $\xi_N$ enerated by unknown CDF $f$, 
# $M = 2$ models:
#    
# 1. Normal distribution with joint CDF $f_1(\xi_N;\theta_1) = \prod\limits_{i=1}^n \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2 \sigma^2} \right) $
# 
# 2. Exponential distribution with joint CDF $f_2(\xi_N;\theta_2) = \prod\limits_{i=1}^n \lambda \exp(-\lambda x_i) 1_{\{ x_i > 0\}}$.
# 
# 
# Then, the parameter vectors are $\theta_1 = [\mu \; \sigma^2]$ and $\theta_2 = \lambda$.
# 
# Then, the model parameters $\theta_m$ are estimated by ML estimation 
# to give estimators $\hat{\theta}_1, \ldots, \hat{\theta}_M$.
# 
# Let $S$ be the index of the chosen model, i.e. $S \in [1, \ldots, M]$.
# 
# General criterion: 
# - Measure dissimilarity between the $m$ model CDF with the true CDF by the KL divergence.  
# - $D_\textrm{KL}(f(\xi_N), f_m(\xi_N;\hat{\theta}_m)) = \mathrm{E} \left[ -\ln \left( \frac{f_m(\xi_N; \hat{\theta}_m)}{f(\xi_N)} \right) \right]$.
# - The expected value is with respect to the true CDF $f$.   
# - Ideally, $S = \arg \min\limits_{m=1, \ldots, M} \mathrm{E}[D_\textrm{KL}(f(\xi_N), f_m(\xi_N;\hat{\theta}_m))]$ 
# where this $\mathrm{E}$ is over the sampling distribution of $\hat{\theta}_m$.
# - But $f$ and sampling distribution of $\hat{\theta}_m$ are unknown!
# 
# 
# Popular criteria:
# - *Akaike information criterion* (AIC)
# - *Corrected Akaike information criterion* (CAIC)
# - *Bayesian information criterion* (BIC)
# 
# 
# Akaike information criterion:
# - Let $S = \arg \min\limits_{m=1, \ldots, M} \textrm{AIC}_m$, 
# - where $\textrm{AIC}_m = K_m - \ln(f_m(\xi_n;\hat{\theta}_m))$ 
# - where $K_m = \mathrm{dim}(\hat{\theta}_m)$
# 
#  
# Corrected Akaike information criterion:
# - $\textrm{CAIC}_m = \textrm{AIC}_m + \frac{2 K_m^2 + 2 K_m}{N - k_m - 1}$ 
# - This approximation is more precise for smaller samples.
# 
#  
# Bayesian information criterion: 
# $\textrm{BIC}_m = K_m \ln(N) - 2 \ln(f_m(\xi_n;\hat{\theta}_m))$ 
# 
# All these criteria penalize the dimensionality/complexity $K_m$ of the model. 
#  

# %% [markdown]
# ## Conditional Models
# 
# Recall that a statistical model consists of: 
# - a sample $\xi$ that is a realization of a random vector $\Xi$; 
# - its unknown joint CDF $F_\Xi(\xi)$; 
# - the sample is used to infer characteristics of $F_\Xi(\xi)$;
# - a model for $\Xi$ is used to make such inferences
# - such model is just a set of joint CDFs to which $F_\Xi(\xi)$ is assumed to belong.
# 
# 
# In a conditional model, the sample is split into inputs $x$ and outputs $y$: 
# $\xi = [y \: x]$.
# 
# The object of interest: 
# the conditional CDF of the outputs given the inputs
# $F_{Y|X=x}(y)$.
# 
# The distribution of the inputs $x$ are ignored.
# 
# Terminology:
# - *Regression model*, a conditional model wherein the output is continuous.
# - *Classication model*, a conditional model wherein the output is discrete.
# - Inputs: predictors, independent/explanatory variables, features, regressors.
# - Outputs: predictands, dependent/response/target variables, regressands. 
# 
# 
# Examples: 
# - linear regression model;
# - logistic classication model.

# %% [markdown]
# ## Linear Regression
# 
# Assume a sample of realizations $(y_i, x_i)$ for $i = 1, \ldots, N$; 
# the outputs $y_i$ are scalars while the associated inputs are $1 \times K$ vectors.
# 
# Postulate: $y_i = x_i \beta + \epsilon_i$.
# 
# $\beta$ is a $K \times 1$ vector of constants called *regression coefficients*;
# $\epsilon_i$ is an unobservable error term.
# 
# Matrix notation: $y = X \beta + \epsilon$
# 
# $y = \begin{bmatrix} y_1 \\ \vdots \\ y_N \end{bmatrix}, X = \begin{bmatrix} x_1 \\ \vdots \\ x_N \end{bmatrix}, \epsilon = \begin{bmatrix} \epsilon_1 \\ \vdots \\ \epsilon_N \end{bmatrix}$
# 
# $X$ is a $N \times K$ matrix often called the *design matrix*.
# 
# It is assumed that the first entry of $x_i$ (or first column of $X$) is equal to $1$;
# the corresponding $\beta$ is called *intercept*.
# 
# Inference about this model is carried out as point and set estimation and hypothesis testing 
# about $\beta$ and $\mathrm{var}[\epsilon]$.
# 
# OLS estimation:
# - A common estimator of $\beta$ is the *ordinary least squares* (OLS) estimator.
# - Definition: $\hat{\beta} = \arg \min\limits_b \sum\limits_{i=1}^N (y_i - x_i b)^2$.
# 
# 
# A popular linear regression model is 
# the *normal linear regression model* (NLRM) by assuming: 
# - $X$ has full rank (i.e. $X^\mathsf{T}X$ is invertible and $\hat{\beta}$ as defined) 
# - $\epsilon$ has a multivariate normal distribution 
# conditional on $X$ and all $\epsilon_i$ are mutually independent and have constant variance 
# (i.e. $\mathrm{var}[\epsilon] = \sigma^2 I$). 
# 
#  
# Properties:
# - If $X$ has full rank, then $\hat{\beta} = (X^\mathsf{T}X)^{-1} X^\mathsf{T} y$.
# - $\hat{\beta}$ is unbiased, i.e. $\mathrm{E}[\hat{\beta}] = \mathrm{E}[\hat{\beta}|X] = \beta$ 
# - $\mathrm{var}[\hat{\beta}|X] = \sigma^2 (X^\mathsf{T} X)^{-1}$.
# - $\hat{\beta}$ is consistent, i.e. i.e. converges to $\beta_0$ as $N \rightarrow \infty$.
# - $\hat{\beta}$ is asymptotically normal.
# - The OLS estimator of the error terms $\sigma^2$ is 
# the adjusted sample variance of the residuals ($e_i = y_i - x_i \hat{\beta}$),  
# $\widehat{\sigma^2} = \frac{1}{N - K} \sum\limits_{i = 1}^N e_i^2$.
# - $\widehat{\sigma^2}$ is unbiased, 
# i.e. $\mathrm{E}[\widehat{\sigma^2}] = \mathrm{E}[\widehat{\sigma^2}|X] = \sigma^2$.
# - $\widehat{\sigma^2}$ has a gamma distribution with parameters $N-K$ and $\sigma^2$. 
# 
#

# %% [markdown]
# ### R squared of a Linear Regression
# 
# The (unadjusted) *R squared* of a linear regression is denoted $R^2$ 
# and is defined as $R^2 = 1 - \frac{S_e^2}{S_y^2}$.
# 
# $S_y^2$ is the unadjusted sample variance of the residuals, 
# $S_y^2 = \frac{1}{N} \sum\limits_{i=1}^N (y_i - \bar{y})^2$
# where $\bar{y} = \frac{1}{N}\sum\limits_{i=1}^N y_i$ is the sample mean.
#  
# $S_e^2$ is the unadjusted sample variance of the outputs, 
# $S_e^2 = \frac{1}{N} \sum\limits_{i=1}^N (e_i - \bar{e})^2$ 
# where $\bar{e} = 0$.
# 
# $R^2$ is a goodness-of-fit measure; $R^2 = 1$ when perfect, $0<R^2<1$ otherwise. 
# 
# The *adjusted R squared* is $r^2 = 1 - \frac{N-1}{N-K} \frac{S_e^2}{S_y^2} = 1 - \frac{s_e^2}{s_y^2}$.$.
# where $s_y^2 = \frac{1}{N-1} \sum\limits_{i=1}^N (y_i - \bar{y})^2$ 
# $s_e^2 = \frac{1}{N-K} \sum\limits_{i=1}^N (e_i - \bar{e})^2$ 
# are the adjusted sample variances.
# 
# $\frac{N-1}{N-K}$ is called a *degrees of freedom adjustment*.
# 
#  
# ### Hypothesis Testing in Linear Regression
# 
# A *t test*:
# - Test a restriction on a single coefficient.  
# - Null hypothesis, $H_0: \beta_k = q$.
# - Test statistic is $t = \frac{\hat{\beta}_k - q}{\sqrt{\hat{\sigma^2} S_{kk}}}$
# - $S_kk$ is the $k$-th diagonal entry of $(X^\mathsf{T}X)^{-1}$.
# - $t$ has a standard Student's t distribution with $N-K$ DoFs.
# - $H_0$ is rejected if $t$ is outside the acceptance region. 
# 
# 
# A *F test*:
# - Testing a set of linear restrictions.  
# - Null hypothesis, $H_0: A\beta = q$ 
# - where $A$ is a $L \times K$ matrix, $q$ is a $L \times 1$ vector, 
# and $L$ is the number of restrictions.
# - Test statistic is $F = \frac{1}{L} (A \hat{\beta} - q)^\mathsf{T} \left[\widehat{\sigma^2} A \: (X^\mathsf{T} X)^{-1} A^\mathsf{T} \right] (A \hat{\beta} - q)$.
# - $F$ has an F distribution with $L$ and $N-K$ DoFs.
# 
# 
# Since the ML estimator of $\beta$ is equal to the OLS estimator, 
# the usual ML-based tests can be used on $\beta$.
# 
# If the OLS estimator is asymptotically normal:
# - *z test*: like the t test but using $z_N = \frac{\hat{\beta}_k - q}{\sqrt{\frac{\hat{V}_{kk}}{N}}}$ 
# where $\hat{V}$ is the asymptotic covariance matrix of $\beta$.
# - $z_N$ has a standard normal distribution.
# - *chi-square test*: like the F test but using $\chi_N^2 = (A \hat{\beta} - q)^\mathsf{T} \left[\frac{1}{N} A \: \hat{V} A^\mathsf{T} \right]^{-1} (A \hat{\beta} - q) $.    
# - $\chi_N^2$ has a chi-square distribution with $L$ DOFs.
# 

# %% [markdown]
# ### Gauss-Markov Theorem
# 
# Under certain conditions, the OLS estimator of $\beta$ of a linear regression model, 
# $\hat{\beta} = (X^\mathsf{T}X)^{-1} X^\mathsf{T} y$,   
# is the *best linear unbiased estimator* (BLUE), 
# i.e. the one with the smallest variance.
# 
# ### Generalized Least Squares
# 
# The *generalized least squares* (GLS) estimator of $\beta$ 
# is a generalization of the OLS estimator.
# 
# The GLS estimator is used when the OLS one is not BLUE.  
# The Gauss-Markov theorem does not apply   
# when no homoskedasticity ($\epsilon_i$'s have different variances)
# or presence of serial correlation (covariance between $\epsilon_i$'s is not zero).
# 
# GLS estimator: 
# - $\mathrm{var}[\epsilon|X] = V$ 
# where $V$ is a $N \times N$ symmetric positive definite matrix.
# - Write $V = \Sigma \Sigma^\mathsf{T}$
# - Write $\Sigma^{-1} y = \Sigma^{-1} X \beta + \Sigma^{-1}\epsilon \: \rightarrow \: \tilde{y} = \tilde{X} \beta + \tilde{epsilon}$.
# - Then, $\hat{\beta}_\textrm{GLS} = (\tilde{X}^\mathsf{T}\tilde{X})^{-1} \tilde{X}^\mathsf{T} \tilde{y} = (X^\mathsf{T} V^{-1} X)^{-1} X^\mathsf{T} V^{-1} y$.
# - $\hat{\beta}_\textrm{GLS}$ is BLUE.
# 
# 
# The OLS problem was to solve $\hat{\beta}_\textrm{OLS} = \arg \min\limits_b (y - X \; b)^\mathsf{T} (y - X \; b)$. 
# 
# The GLS problem is instead $\hat{\beta}_\textrm{GLS} = \arg \min\limits_b (y - X \; b)^\mathsf{T} V^{-1} (y - X \; b)$. 
# 
# When $V$ is diagonal (i.e. uncorrelated $\epsilon_i$'s), 
# the GLS estimator is called the *weighted least squares* (WLS) estimator, 
# $\hat{\beta}_\textrm{GLS} = \arg \min\limits_b (y - X \; b)^\mathsf{T} V^{-1} (y - X \; b) = \sum\limits_{i=1}^N V_{ii}^{-1} (y_i - X_i b)^2$. 
# 
# $V$ is usually unknown. 
# In *feasible* GLS, 
# it is replaced by its estimate $\hat{V}$: 
# $\hat{\beta}_\textrm{FGLS} = (X^\mathsf{T} \hat{V}^{-1} X)^{-1} X^\mathsf{T} \hat{V}^{-1} y$
# 
# However, there is no general method for estimating $V$.
# One way is to run OLS once, 
# assume a diagonal $V$, and take $\hat{V}_{ii} = \alpha \hat{V}_{i-1,i-1} + (1 - \alpha) \hat{\epsilon}_i$ 
# where $\epsilon_i = y_i - X_i \hat{\beta}_\textrm{OLS}$.
#   
              
# %% [markdown]
# ### Multicollinearity
# 
# *Multicollinearity* is a problem that occurs when one or more regressors are highly correlated.
# Even when $N \gg 0$, $\hat{\beta}_\textrm{OLS}$ still has high variance.
# 
# In such cases, the matrix $X^\mathsf{T}X$ is 
# either not full rank or close to be rank deficient. 
# 
# A measure of multicollinearity is the *variance inflation factor* (VIF):
# $\mathrm{VIF}_k = \frac{1}{1 - R_k^2}$ 
# where $R_k^2$ is the R squared of a regression 
# in which $X_{\cdot k}$ is the dependent variable 
# and $X_{\cdot j}$ ($j \neq k$) are the dependent variables.
# 
# Watch out for when $\mathrm{VIF}_k > 10$ 
# or the *condition number* of $X^\mathsf{T}X$ is $>20$.
# 
# Remedies:
# - Increase sample size $N$.
# - Drop the regressors with high $\mathrm{VIF}$.
# - Replace high-$\mathrm{VIF}$ regressors with a linear combination of them.
# - Use regularization methods like *ridge* (add $\lambda |b|^2$ to the OLS loss function), 
# *lasso* (*least absolute shrinkage and selection operator*, 
# add $\lambda |b|$ to the OLS loss function), and *elastic net* (both).
# - Use *Bayesian regression*.
# 
# 
# 
# 
# ### Ridge Regression
# 
# *Ridge regression* is a linear regression model whose coefficients $\beta$ 
# are not estimated by OLS but by the ridge estimator $\hat{\beta}_\lambda$ 
# where $\lambda \leq 0$ (it is biased but has smaller variance).
# 
# $\hat{\beta}_\lambda = \arg \min\limits_b \sum\limits_{i=1}^N (y_i - x_i b) + \lambda \sum\limits_{k=1}^K b_k^2$
# 
# Solution: $\hat{\beta}_\lambda = (X^\mathsf{T} X + \lambda I)^{-1} X^\mathsf{T} y$.
# 
# Assuming $\mathrm{E}[\epsilon|X] = 0$ and $\mathrm{var}[\epsilon|X] = \sigma^2 I$, 
# - the bias is $\mathrm{E}[\hat{\beta}_\lambda|X] - \beta = [(X^\mathsf{T} X + \lambda I)^{-1} - (X^\mathsf{T} X)^{-1}] X^\mathsf{T} X \: \beta$.
# - the variance is $\mathrm{var}[\hat{\beta}_\lambda|X] = \sigma^2 (X^\mathsf{T} X + \lambda I)^{-1} X^\mathsf{T} X (X^\mathsf{T} X + \lambda I)^{-1}$.
# 
# 
# Ridge regression is so-called because the $\lambda I$ term is like adding a diagonal *ridge* 
# to the matrix $X^\mathsf{T} X$.
#
# From the bias-variance decomposition of the mean-squared error: 
# - $\mathrm{MSE}(\hat{\beta}_\lambda|X) = \mathrm{E}[|\hat{\beta}_\lambda|^2 \; |X] = \mathrm{tr}(\mathrm{var}[\hat{\beta}_\lambda | X]) + |\mathrm{Bias}[\hat{\beta}_\lambda | X]|^2$.
# - For OLS, 
# $\mathrm{MSE}(\hat{\beta}_\textrm{OLS}|X) = \mathrm{tr}(\mathrm{var}[\hat{\beta}_\textrm{OLS} | X])$  
# - It can be shown that $\mathrm{MSE}(\hat{\beta}_\lambda|X) < \mathrm{MSE}(\hat{\beta}_\textrm{OLS}|X)$ 
# for some value of $\lambda$.
# 
# 
# A way to find $\lambda$ (a *hyperparameter*) is to use 
# k-fold cross-validation:
# - split sample into two sets (training set and test set);
# - split training set into $k$ subsets;
# - for $i = 1,\ldots,k$, 
# drop the $i$-th subset to merge the remainder into a training subset for the model 
# (find $\hat{\beta}_{\lambda_i}$) and use the $i$-th subset as the *validation set* 
# (calculate the resulting MSE);
# - find the value of $\lambda$ that minimizes the MSE 
# and apply it to the test set for performance metric.
# 
# 
# The ridge estimator is *not* scale-invariant; 
# always use *standardized variables* 
# (substract each variable by its mean and divide by its standard deviation; 
# drop the intercept $\beta_i = 1$ since it is not standardizable).
#  

# %%[markdown]
# ## Classification Models
# 
# A *classification model* is a conditional model 
# wherein the output variable has a *discrete* probability distribution 
# 
# Two types:
#  
# 1. *binary* classification models 
# wherein the output has a Bernoulli distribution conditional on the inputs;
# 2. *multinomial* classification models 
# wherein the output has a multinoulli distribution conditional on the inputs. 
# 
# 
# Let $X = [X_1 \: \ldots \: X_K]$ be a $K \times 1$ random vector with support  
# $R_X = \left\{ x \in \{0,1\}^K \: : \: \sum\limits_{i=1}^K x_i = 1 \right\}$.
# $X$ has a *multinoulli* distribution with probabilities $p_1,\ldots,p_K$  
# if its joint PMF is $p_X(x_1,\ldots,x_K) = \begin{cases} \prod\limits_{i=1}^K p_i^{x_i} & (x_1,\ldots,x_K) \in R_X \\ 0 & \textrm{otherwise} \end{cases}$.
# 
# Example:  
# - Suppose an output variable can take on one of three values (`red`,`green`,`blue`). 
# - Then, it can be represented as a multinoulli random vector 
# with realizations $\begin{cases} [1\:0\:0] & \textrm{red} \\ [0\:1\:0] & \textrm{green} \\ [0\:0\:1] & \textrm{blue} \end{cases}$.
# 
# 
# Assumptions:
# - A sample of data $(y_i,x_i)$ for $i = 1,\ldots,N$ is observed.
# - Each input $x_i$ is a $1 \times K$ vector.
# - Each output $y_i$ can be any $1 \times J$ vector whose entries are $c_j = \delta_{ij}$.
# - There exists $J$ functions $f_j$ such that $P(y_i = c_j|x_i) = f_j(x_i;\theta)$ for all $i,j$.
# 
#  
# *Binary logistic* (or *logit*) classification model
# - The conditional PMF of $y_i$ is $p_{Y_i|X=x_i}(y_i) = \begin{cases} \mathrm{sigm}(x_i\beta) & y_i = 1 \\ 1 - \mathrm{sigm}(x_i\beta) & y_i = 0 \\ 0 & \textrm{otherwise} \end{cases}$.
# - $\beta$ is a $K \times 1$ vector of coefficients and $\mathrm{sigm}(t) = \frac{1}{1 + \mathrm{e}^{-t}}$.
# - Thus, $J = 2$, $f_1(x_i;\theta) = P(y_i = 1|x_i) = \mathrm{sigm}(x_i\beta)$, and $f_2(x_i;\theta) = P(y_i = 0|x_i) = 1 - \mathrm{sigm}(x_i\beta)$.
# 
# 
# *Multinomial logistic* classification model (also called *softwax model*):
# - $J \leq 2$.
# - The conditional PMF of $y_i$ is $f_j(x_i;\theta) = P(y_i = c_j|x_i) = \frac{\mathrm{e}^{x_i \beta_j}}{\sum\limits_{k=1}^K \mathrm{e}^{x_i\beta_k}}$.
# - Each class $j$ corresponds to a $K \times 1$ coefficient vector $\beta_j$.
# - The vector of parameters is just $\theta = [\beta_1 \: \ldots \: \beta_J]$.  
# 
#  
# The parameters of a multinomial logistic classification model 
# can be estimated by the ML estimation method.  
# 
# The likelihood of a sample point $(y_i,x_i)$ is $L(\theta;y_i,x_i) = \prod\limits_{j=1}^J f_j(x_i;\theta)^{y_{ij}}$.
# where $y_{ij} = \delta_{ij}$ is the $i$-th component of the multinoulli vector $y_i$.
# 
# For the whole sample, $L(\theta;y,x) = \prod\limits_{i=1}^N L(\theta;y_i,x_i) = \prod\limits_{i=1}^N \prod\limits_{j=1}^J f_j(x_i;\theta)^{y_{ij}}$.
# 
# The log-likelihood is just $l(\theta;y,x) = \ln(L(\theta;y,x))$.
#   
# The ML estimator of the parameter $\theta$ is 
# then the solution of the optimization problem
# $\hat{\theta} = \arg \max_\limits{\theta} l(\theta;y,x)$.
#   

# %%[markdown]
# 
# ### Binary Logistic Classification Model
# 
# The logistic function $\mathrm{sigm}(t)$ is used to tackle the classification problem 
# as in a linear regression model (as linear combination of $x_i \beta$) 
# while ensuring the terms $x_i \beta$ are between $0$ and $1$.
# 
# Alternatively, consider $z_i = x_i \beta + \epsilon$ such that 
# $y_i = \begin{cases} 1 & z_i \leq 0 \\ 0 & z_i < 0 \end{cases}$.
# 
# Then, 
# $\begin{align} P(y_i = 1|x_i) &= P(z_i \leq 0|x_i)\\&= P(\epsilon_i \leq x_i\beta|x_i)\\&=F(x_i\beta)\end{align}$.
# 
# Thus, the logit model is just the assumption that 
# $\epsilon_i$ has the specific CDF $F_{\epsilon_i}(x_i \beta) = \frac{1}{1 - \exp(x_i \beta)}$.
# 
# If $\epsilon_i$ has a standard normal CDF, we would have the so-called *probit* model.
#  
  
# %%[markdown]
# 
# ## Markov Chain
# 
# A *Markov chain* is a sequence of random variables/vectors $\{X_n\}$
# that possess the *Markov property*:  
# any given term $X_n$ is conditionally independent 
# of all terms preceding $X_{n-1}$, 
# i.e. $F(x_n|x_{n-1}, x_{n-2}, \ldots, x_1) = F(x_n|x_{n-1})$.
# 
# The *state space* $S$ of a Markov chain $\{X_n\}$ is 
# the set of all possible realizations of terms of the chain, 
# i.e. the support of any term $X_n$ is included in $S$.
# 
# $S$ is finite:
# - $S = \{s_1, \ldots, s_J\}$
# - Specify an initial probabiliy distribution for $X_1$, $P(X_1 = s_j) = \pi_{1j}$.
# - Choose a $J \times J$ *transition probability matrix $P$, $P_{ij} = P(X_n = s_j|X_{n-1} = s_i) \: \forall \; n,i,j$.
# - If $P$ is equal for all $n$, we have *time-homogeneity*.
# - $\pi, P$ completely determine the distribution of all terms of the chain, $P(x_n = s_j) = \pi_{nj}, \: \pi_n = \pi_1 P^n$.
# - *Stationary distribution*, an initial distribution $\pi$ such that $\pi_n = \pi$ (thus, $\pi = \pi P$). 
# - *Detailed balance* is satisfied iff $\pi_{bi}P_{ij} = \pi_{bj}P_{ji}$ for any $i,j \leq J$.
# - A chain is *irreducible* iff 
# every state $x$ leads to itself and every other state $y$
# in finite time $\tau_{x,y} = \min \{n > 1 \: : \: X_n = y\}$. 
# - A state $y \in S$ is *recurrent* iff $P_y(\tau_{yy} < \infty) = 1$, otherwise *transient*.
# - A chain is recurrent iff all states in $S$ are recurrent.
# - Period of a state $y$ is the minimum time it takes to return to itself, 
# $d_y = \mathrm{gcd} \{n > 1 \: : \: P_x(X_n = x) > 0\}$.
# - If a chain is irreducible, all $d_y$ are equal to some $d$. 
# - A chain is *aperiodic* iff its period $d = 1$,
# - If a Markov chain with a finite state space $S$ is irreducible, 
# then it has an unique stationary distribution $\pi$.
# - Also: $\frac{1}{n} \sum\limits_{i=1}^N f(X_i) \rightarrow \sum\limits_{j=1}^J \pi_j f(s_j)$ 
# for any bounded function $f$ (*ergodic theorem*). 
# - If also aperiodic, $\lim\limits_{n \rightarrow \infty} \pi_n = \pi$ regardless of $\pi$.
#  
# 
# $S$ is infinite and countable:
# - A state/chain is *positive recurrent* iff $\mathrm{E}_y[\tau_{yy}] < \infty$. 
# - If a Markov chain with a finite state space $S$ is irreducible and positive recurrent,
# then it has an unique stationary distribution.
# - Thus, equivalent ergodic theorem also applies.
# - If also aperiodic, then convergence to the stationary distribution.
#  
# 
# $S$ is infinite and uncountable: ...
# 
  

# %%[markdown]
# 
# ## Autocorrelation
# 
# The autocorrelation coefficient between two terms $X_i, X_j$ of a sequence $\{X_n\}$ is 
# 
# $\rho(i, j) = \frac{\mathrm{cov}[X_i, X_j]}{\sqrt{\mathrm{var}[X_i] \mathrm{var}[X_j]}}$
# 
# This is just the linear correlation coefficient between two random variables 
# of the same sequence.
# 
# The sequence $\{X_n\}$ is said to be *covariance stationary* (or *weakly stationary*) iff
# 
# - all terms have same mean, $\exists \; \mu \in \mathbb{R} \: : \: \mathrm{E}[X_n] = \mu, \; \forall \: n \in \mathbb{N}$;
# - the covariance between any two terms depends only on their distance, $\forall \; j \geq 0, \; \exists \; \gamma_j \in \mathbb{R} \; : \; \mathrm{cov}[X_n, X_{n-j}] = \gamma_j, \; \forall \; n > j$.
# 
# 
# Corollay: 
# 
# - all terms have the same variance, 
# $\exists \; \gamma_0 \in \mathbb{R} \: : \: \mathrm{var}[X_n] = \gamma_0, \; \forall \: n \in \mathbb{N}$;
# - the autocorrelation coefficient between any two terms depends only on their distance 
# (i.e. autocorrelation at *lag* $k$), 
# $\rho(i, j + k) = \rho_k = \frac{\gamma_k}{\gamma_0}$.
# 
# 
# Sample autocorrelation (given the first $N$ realizations of $\{X_n\}$): 
# $\hat{\rho_k} = \frac{\frac{1}{N-k} \sum\limits_{n=1}^{N-k} (X_n - \hat{\mu})(X_{n+k} - \hat{\mu})}{\frac{1}{N} \sum\limits_{n=1}^N (X_n - \hat{\mu})^2}$
# 
# If the sequence is covariance stationary, 
# then this expression is a consistent estimator of $\rho_k$.
# 
# Treating $k$ as a variable, $\rho_k$ can be called the *autocorrelation function* (ACF).
#   

# %%[markdown]
# 
# ## Markov Chain Monte Carlo (MCMC)
#
# Monte Carlo methods give approximations of 
# some feature like the mean of a probability distribution.  
# 
# In a MCMC method:
# - a sample $\xi_n = [x_1 \: \ldots \: x_n]$ is also generated by computer 
# - but it is such that the sequence $\{X_n\}$ is a Markov chain (i.e. not independent) 
# converging to the stationary distribution $F_X(x)$. 
# - A plug-in estimate $T(F_n)$ is made using the empirical distribution $F_n(x)$ 
# (each $x_i$ is assigned a probability of $\frac{1}{n}$).
# - By an ergodic theorem, $T(F_n)$ (e.g. $= \hat{\mu}) converges to $T(F_x)$ (e.g. $=\mu$).
# 
# 
# Popular examples:
#  
# 1. Metropolis-Hastings algorithm; 
# 2. Gibbs sampling algorithm. 
#    

# %%[markdown]
# 
# ## Bayesian Inference
# 
# Recall that in a statistical inference problem: 
# - observations form a sample in the form of a vector $x$;
# - $x$ is a realization of a random vector $X$; 
# - the probability distribution $F_X$ is unknown;
# - define a statistical model, i.e. a set $\Phi$ of possible $F_X$'s;
# - parameterize the model with $\theta$ (optional);
# - use the sample $x$ and model $\Phi$ to make a statement about $F_X$ or $\theta$. 
# 
# 
# Steps:
# 1. Define the *likelihood* $p(x|\theta)$.
#   - it is the PDF of $x$ when the parameter of $F_X$ is $\theta$.
#   - e.g. suppose $x = [x_1 \; \ldots \; x_n]$ where $x_i$ is drawn from a normal distribution 
#     ($\mu unknown, $\sigma^2$ known). 
#   - $\begin{align} p(x_i|\mu) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2 \sigma^2} \right) \\ p(x|\mu) &= \prod\limits_{i=1}^n p(x_i|\mu) \end{align}$
# 2. Define the *prior* $p(\theta)$. 
#   - it is the subjective PDF assigned to $\theta$.
#   - e.g. assuming $\mu$ is likely to be near some value $\mu_0$, $p(\mu) = \frac{1}{\sqrt{2 \pi \tau^2}} \exp\left(-\frac{(\mu - \mu_0)^2}{2 \tau^2} \right)$.
# 3. Define the *prior predictive distribution* $p(x)$.
#   - With the prior and the likelihood, derive the marginal density of $x$.
#   - Recall that a joint PDF can be written as a product of a conditional PDF and a marginal PDF.
#   - $p(x) = \int_\theta p(x, \theta) \mathrm{d}\theta = \int_\theta p(x|\theta) p(\theta) \mathrm{d}\theta$
#   - e.g. $p(x) = \int_{-\infty}^\infty p(x|\mu) p(\mu) \mathrm{d}\mu = \ldots = (2 \pi \sigma^2)^{-n/2} \left|\mathrm{det}\left(I_n + \frac{\tau^2}{\sigma^2}ii^\mathsf{T}\right)\right|\exp \left(-\frac{1}{2\sigma^2}(x - i\mu_0)^\mathsf{T} \left(I_n + \frac{\tau^2}{\sigma^2}ii^\mathsf{T}\right)^{-1}(x - i\mu_0) \right)$.
#   - This is just a multivariate normal distribution with mean $i\mu_0$ and covarance matrix $\sigma^2 I_n + \tau^2 ii^\mathsf{T}$.
#   - $i$ is the $n \times 1$ vector of ones, $I_n$ is the $n \times n$ identity matrix.
# 4. Calculate the *posterior* $p(\theta|x)$.
#   - This is the PDF of $\theta$ conditional on $x$.
#   - Use Bayes' rule, $p(\theta|x) = \frac{p(x|\theta) p(\theta)}{p(x)} = \frac{p(x|\theta) p(\theta)}{\int_\theta p(x|\theta) p(\theta) \mathrm{d}\theta}$.
#   - e.g. $p(\mu|x) = \ldots = (2 \pi \mu_n)^{-1/2} \exp \left(-\frac{(\mu - \mu_n)^2}{2 \sigma_n^2} \right)$
#   - where $\mu_n = \sigma_n^2 \left[ \frac{n}{\sigma^2} \left(\frac{1}{n} \sum\limits_{i=1}^n x_i \right) + \frac{\mu_0}{\tau^2} \right]$
#   - and $\sigma_n^2 = \left(\frac{n}{\sigma^2} + \frac{1}{\tau^2}\right)^{-1} $
#   - As expected, when $n \rightarrow \infty$, all the weight is given to the information from the sample $x$.
# 
# 
# The *posterior predictive distribution*:
# - Suppose that a second sample $y$ is drawn from the same distribution as $x$.
# - Suppose also that the distribution of $y$ is independent on $x$, $p(y|\theta,x) = p(y|\theta)$. 
# - Given $p(\theta|x)$, we want to known the probability of $y$ given $x$ (the posterior predictive distribution).
# - $\begin{align} p(y|x) &= \int_\theta p(y,\theta|x) \mathrm{d}\theta \\ &= \int_\theta p(y|\theta,x) \; p(\theta|x) \mathrm{d}\theta \\ &= \int_\theta p(y|\theta) \; p(\theta|x) \mathrm{d}\theta \end{align}$ 
# - e.g. let $y$ = \{x_{n+1}\} (a single new drawn), 
# it can be shown that $p(x_{n+1}|x)$ is a normal distribution 
# with mean $\mean_n$ and variance $\sigma^2 + \sigma_n^2$.
# 

# %% [markdown]
# ### Hierarchical Bayesian Models
# 
# *Hierarchical* Bayesian Models are those 
# wherein the prior distribution of some of the parameters 
# depends on other parameters.
# 
# Given observed data $x$: 
# - the likelihood depends on two parameter vectors $\theta, \phi$, $p(x|\theta,\phi)$; 
# - the prior is $p(\phi, \theta) = p(\theta|\phi) p(\phi)$;
# - special case: the likelihood is required to be independent of $\phi$, 
# $p(x|\theta,\phi) \rightarrow p(x|\theta)$;
# - in such case, $\phi$ is a *hyper-parameter* and $p(\phi)$ is a *hyper-prior*; 
# 
# 
# Example (random means):
# - Suppose a sample $x = [x_1 \; \ldots \; x_n]$.
# - $x_i$ is drawn from a normal distribution with unknown $\mu_i$ 
# and known common variance $\sigma^2$.
# - $p(x_i|\mu_i) = (2 \pi \sigma^2)^{-1/2} \exp \left(-\frac{(x_i - \mu_i)}{2 \sigma^2} \right)$.
# - Denote $\mu = [\mu_1 \; \ldots \; \mu_n]$.
# - Assuming the $x_i$'s are independent, 
# $p(x|\mu) = \prod\limits_{i=1}^n p(x_i|\mu_i)$.
# - Assuming the $\mu_i$'s are a sample of IID draws from a normal distribution of unknown means $m$ and known common variance $\tau^2$,   
# - $p(\mu|m) = \prod\limits_{i=1}^n p(\mu_i|m)$ = (2 \pi \sigma^2)^{-n/2} \exp \left(-\frac{(\mu_i - m)}{2 \tau^2} \right)$.
# - Assign a normal prior with known mean $m_0$ and variance $u^2$ to the hyper-parameter $m$, 
# - $p(m) = (2 \pi u^2)^{-1/2} \exp \left(-\frac{(m - m_0)}{2 u^2} \right)$
# - Such a model is a hierarchical Bayesian model.
# 
# 
# Example (normal-inverse gamma model):
# - Suppose now $x_i$ is drawn with unknown variance $\sigma^2$ as well.
# - $p(x_i|\mu, \sigma) = (2 \pi \sigma^2)^{-1/2} \exp \left(-\frac{(x_i - \mu)}{2 \sigma^2} \right)$.
# - $p(x|\mu,\sigma) = \prod\limits_{i=1}^n p(x_i|\mu, \sigma)$.
# - Assume $\mu$ is normal with known mean $m$ and variance $\sigma^2/v$ ($\nu$ is a known parameter).      
# - $p(\mu|\sigma^2) = (2 \pi \frac{\sigma^2}{\nu})^{-1/2} \exp \left(-\frac{\nu (\mu - m)}{2 \sigma^2} \right)$.
# - Assign an inverse-gamma (gamma distribution with precision $1/\sigma^2$) 
# prior to $\sigma^2$.
# - $p(\sigma^2) = \frac{(k/h)^{k/2}}{2^{k/2} \Gamma(k/2)} (1/sigma^2)^{k/2+1} \exp(-\frac{k}{2 h \sigma^2})$.
# - $p(\sigma^2 | x) = \frac{p(x|\mu,\sigma^2)  \left( p(\mu|\sigma^2) p(\sigma^2) \right)}{p(x)}
# 
# 
# To compute the posterior distribution:
#  
# 1. Conditional on $\phi$, 
# - the prior predictive distribution of $x$: $p(x|\phi) = \int p(x,\theta|\phi) \mathrm{d}\phi = \int p(x|\theta,\phi) p(\theta|\phi) \mathrm{d}\theta$. 
# - the posterior distribution of $\theta$: $p(\theta|x,\phi) = \frac{p(x|\theta,\phi) \; p(\theta|\phi)}{p(x|\phi)}$.  
# 2. Using 1.,   
# - the prior predictive distribution of $x$: $p(x) = \int p(x,\phi) \mathrm{d}\phi = \int p(x|\phi) p(\phi) \mathrm{d}\phi$. 
# - the posterior marginal distribution of $\phi$: $p(\phi|x) = \frac{p(x|\phi) \; p(\phi)}{p(x)}$.
# 3. The posterior joint distribution of $\phi, \theta$:  
# $p(\phi, \theta | x) = p(\theta|x,\phi) p(\phi|x)$.
# 4. The posterior marginal distribution of $\theta$: 
# $p(\theta|x) = \int p(\phi\theta|x) \mathrm{d}\phi$.
# 
  
# %% [markdown]
# ### Bayesian Estimation of the Parameters of a Normal Distribution
# 
# Case: *unknown* mean $\mu$ and *known* variance $\sigma^2$.
# - Suppose a sample $x = [x_1 \; \ldots \; x_n]$.
# - $x_i$ are drawn IID from a normal distribution.
# - The likelihood is $p(x_i|\mu) = N(\mu, \sigma^2)$ 
# and $p(x|\mu) = \prod\limits_{i=1}^n p(x_i|\mu)$.
# - The prior is $p(\mu) = N(\mu_0, \tau_0^2)$.
# - To calculate the posterior is $p(\mu|x)$, 
#   1. Write $p(x,\mu) = p(x|\mu) p(\mu) = \ldots = h(x) g(\mu,x)$
#   2. By factorization, $p(x) = h(x)$ and $p(\mu|x) = g(\mu, x)$.
# - Thus, the posterior $p(\mu|x)$ is just $N(\mu_n, \sigma_n^2)$.
# - The prior predictive distribution $p(x)$ is 
# $N(\mu_0 i, \sigma^2 I_n + \tau_0^2 ii^\mathsf{T})$.
# - The posterior predictive distribution $p(\tilde{x}|x)$ 
# can be calculated too ($\tilde{x} = [x_{n+1} \; \ldots \; x_{n+m}]$).
# 
# 
# Case: *unknown* mean $\mu$ and *unknown* variance $\sigma^2$.  
# - The prior is hierarchical.
# - See example in previous subsection.

# %% [markdown]
# ### Bayesian Linear Regression
# 
# Recall the normal linear regression model:
# - $y = X\beta + \epsilon$;
# - $y$ is the $N \times 1$ vector of observations of the dependent variable;
# - $X$ is the $N \times K$ matrix of regressors (full rank);
# - $\beta$ is the $K \times 1$ vector of regression coefficients;
# - $\epsilon$ is the $N \times 1$ vector of errors 
# (multivariate normal distribution conditional on $X$ with mean $0$ 
# and covariance matrix $\sigma^2 I_N$)
# 
# 
# Case: *unknown* $\beta$ and *known* variance $\sigma^2$.
# - Since $\epsilon$ is multivariate normal and $y$ is a linear transformation of it, 
# $y$ is also multivariate normal;
# - the likelihood is then $p(y|\beta,X) = (2 \pi)^{N/2} |\mathrm{det}(\sigma^2 I_N)|^{-1/2} \exp \left(-\frac{1}{2} (y - X\beta)^\mathsf{T} (\sigma^2 I_N)^{-1} (y - X \beta) \right)$
# - assume the prior on $\beta$ to be multivariate normal; 
# - with mean $\beta_0$ and covariance $\sigma^2 V_0$, $V_0$ is some $K\times K$ symmetric positive definite matrix; 
# - $p(\beta) = (2 \pi)^{N/2} |\mathrm{det}(\sigma^2 V_0)|^{-1/2} \exp \left(-\frac{1}{2} (\beta - \beta_0)^\mathsf{T} (\sigma^2 V_0)^{-1} (\beta - \beta_0) \right)$
# - Apply factorization to get $p(\beta|y,X)$...   
# - $p(\beta|y,X) = \ldots =  (2 \pi)^{K/2} |\mathrm{det}(\sigma^2 V_N)|^{-1/2} \exp \left(-\frac{1}{2} (\beta - \beta_N)^\mathsf{T} (\sigma^2 V_N)^{-1} (\beta - \beta_N) \right)$
# - where $V_N = (V_0^{-1} + X^\mathsf{T} X)^{-1}$;
# - where $\beta_N = V_N [V_0^{-1} \beta_0 + X^\mathsf{T} y]$;
# - the posterior of $\beta$ is multivariate normal with mean $\beta_N$ and covariance $\sigma^2 V_N$;
# - recall from OLS, $\beta_\textrm{OLS} = (X^\mathsf{T} X)^{-1} X^\mathsf{T} y$ 
# - then, $\beta_N = \ldots = V_N [V_0^{-1} \beta_0 + X^\mathsf{T} X \beta_\textrm{OLS}]$; 
# - so, the posterior mean of $\beta$ is the weighted average of 
#   1. the OLS estimate from $X, y$;
#   2. the prior mean $\beta_0$;
# - since $\mathrm{var}[\beta_\textrm{OLS}] = \sigma^2 (X^\mathsf{T}X)^{-1}$ 
# - and $\mathrm{var}[\beta] = \sigma^2 V_0$ by definition, 
# - then, $V_N = (\mathrm{var}[\beta]^{-1} + \mathrm{var}[\beta_\textrm{OLS}]^{-1})^{-1}$ 
# - and $\beta_N =  (\mathrm{var}[\beta]^{-1} + \mathrm{var}[\beta_\textrm{OLS}]^{-1})^{-1} [\mathrm{var}[\beta]^{-1} \beta_0 + \mathrm{var}[\beta_\textrm{OLS}]^{-1} \beta_\textrm{OLS}]$; 
# - the weights of the weighted average are just the inverse variances (i.e. precision) 
# of each source of information (the prior mean and the OLS estimator);  
# - since $\lim\limits_{N \rightarrow \infty} X^\mathsf{T}X = \infty$, 
# $\beta_\textrm{OLS}$ sensibly becomes more important;
# - therefore, Bayesian regression and frequentist (OLS) regression give the same result as large sample size. 
# - the prior predictive distribution $p(y|X)$ is also calculated by factorization;
# - $p(y|X) = \ldots$ (multivariate normal with mean $X\beta_0$ and covariance $\sigma^2 (X V_0X^\mathsf{T} + I_N)$);
# - the posterior predictive distribution can be calculated for a new sample $(\tilde{y},\tilde{X})$ of size $M$;
# - i.e. predict $\tilde{y}$ from $\tilde{X}$ and previous sample; 
# - the posterior $p(\beta|y,X)$ is the new prior;
# - same likelihood $p(\tilde{y}|\tilde{X},y,X) = p(\tilde{y}|\tilde{X},\beta$;
# - factorization, $p(\beta|\tilde{X},y,X) \; p(\tilde{y}|\tilde{X},y,X) = p(\tilde{y}|\tilde{X},\beta) \; p(\beta|y,X)$;
# - result: $\tilde{y}$ is multivariate normal with mean $\tilde{X}\beta_N$ and covariance $\tilde{X}V_N\tilde{X}^\mathsf{T} + I_M$;     
#
# 
# Case: *unknown* $\beta$ and *unknown* variance $\sigma^2$.
# - Similar as before but hierarchical...
#  


# %%
