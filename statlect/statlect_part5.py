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
# is the joint PMF of $\Xi$ associated with the distribution corresponding to $\theta$.
# 
# 2. If $\Xi$ is continuous, $L(\theta;\xi) = f_\Xi(\xi;\theta)$ 
# is the joint PDF of $\Xi$ associated with the distribution corresponding to $\theta$.
# 
# 
# In *generalized method of moments* (GMM) estimation, 
# $Q(\theta, \xi) = -d(G(\theta; \xi),0)$
# 
#  
#  


# %%
