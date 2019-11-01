#%% [markdown]
# # Notes and exercises from [Statlect](https://www.statlect.com)
# --- 

#%% [markdown]
# # Part 4 - [Asymptotic Theory](https://www.statlect.com/asymptotic-theory/)

#%% [markdown]
# ## Sequences of Random Variables
# 
# Let $\{X_n\}$ be a sequence of random variables.
#
# Terminology: 
# - The sequence $\{x_n\}$ is a *realization* of it $\{X_n\}$ 
# if $x_n$ is a realization of $X_n$.
# 
# - It is a sequence *on* sample space $\Omega$ 
# if all $X_n$ are functions $\Omega \rightarrow \mathbb{R}$. 
# 
# - It is an *independent* sequence (or a sequence of independent random variables) 
# if every finite subset of it is a set of mutually independent random variables.
# 
# - It is an *identically distributed* sequence if 
# any two elements $X_i, X_j$ have the same CDF.   
#  
# - It is an *IID* sequence if it has both of the above properties. 
# 
# - It is *stationary* (or strictly stationary) if 
# two random vectors $[X_{n+1}, \ldots, X_{n+q}]$ and $[X_{n+k+1}, \ldots, X_{n+kq}]$ 
# have the same joint CDF. Corollary: IID $\Rightarrow$ stationary.  
# 
# - It is *covariance stationary* (or weakly stationary) if 
# all $X_n$ have the same mean $\mathrm{E}[X_n] = \mu$ and variance $\mathrm{var}[X_n] = \sigma^2$;  
# the latter condition means $\forall \: j, \: \exists \: \gamma_i \in \mathbb{R} \: : \: \mathrm{cov}[X_n, X_{n-j}] = \gamma_j \: \forall \: n < j$. 
#  
# - It is *mixing* (or strongly mixing) if 
# two groups of terms that are far apart are asymptotically independent, 
# i.e. $\lim\limits_{k\rightarrow \infty} \left(\mathrm{E}[f(\boldsymbol{Y}) g(\boldsymbol{Z})] - \mathrm{E}[f(\boldsymbol{Y})]\mathrm{E}[g(\boldsymbol{Z})] \right) = 0$
# for $Y = [X_{n+1}, \ldots, X_{n+q}]$, $Z = [X_{n+k+1}, \ldots, X_{n+k+q}]$ 
# and any function $f, g$ and any $n, q \in \mathbb{N}$. 
# Corollary: independent $\Rightarrow$ mixing. 
# 
# - It is *ergodic* if either $P(\{X_n\} \in A) = 0$ or $P(\{X_n\} \in A) = 1$ 
# where $A$ is a *shift-invariant* set. 'Ergodic' is weaker than 'mixing'.
#  
# - A subset $A$ is shift invariant  if $\{x_n\} in A \; \Rightarrow \; \{x_n\}_{n>1} in A$, 
# where $\{x_n\}_{n>1} = \{x_2, x_3, \ldots\}$.
# 
# ---

#%% [markdown]
# ## Modes of Stochastic Convergence
# 
# ### Pointwise Convergence:
# A sequence of random variables $\{X_n\}$ defined on sample space $\Omega$ 
# is *pointwise convergent* to a random variable $X$ on $\Omega$ 
# if $\{X_n(\omega)\}$ converges to $X(\omega) \; \forall \: \omega \in \Omega$.
# $X$ is called the *pointwise limit* of the sequence.
# 
# Example:
# - Let $\Omega = \{\omega_1, \omega_2\}$ be a sample space with sample points $\omega_1, \omega_2$. 
# - Let $\{X_n\}$ be a sequence of random variables 
# such that $X_n(\omega) = \begin{cases} \frac{1}{n} & \omega = \omega_1 \\ 1 + \frac{2}{n} & \omega = \omega_2 \end{cases}$
# - Since $X_n(\omega_1) \rightarrow 0$ and $X_n(\omega_2) \rightarrow 1$, 
# $\{X_n\}$ is pointwise convergent to $X$, where $X = \begin{cases} 0 & \omega = \omega_1 \\  1 & \omega = \omega_2 \end{cases}$
# 
# 
# For a sequence of random vectors, the standard criterion for convergence is just 
# $\lim\limits_{n \rightarrow \infty} d(\boldsymbol{X}_n(\omega), \boldsymbol{X}(\omega)) = 0$
# where $d$ is the Euclidean distance.
# 
# ### Almost-Sure Convergence:
# A weakened version of pointwise convergence.
# $\{X_n\}$ is almost-surely convergent if $\{X_n(\omega)\}$ converges for most $\omega \in \Omega$ 
# except those belonging in some subset $\subseteq E$, where $E$ is a zero-probability event. 
# 
# ### Convergence in Probability: 
# A sequence of random variable $\{X_n\}$ is *convergent in probability* to random variable $X$ 
# if there is a high probability that their difference is very small, 
# i.e. $\lim\limits_{n \rightarrow \infty} P(|X_n - X| > \epsilon) = 0$ for any $\epsilon > 0$.
# 
# Example:
# - Let $X$ be a discrete random variable with support $R_X = \{0,1\}$  
# and PMF $p_X(x) = \begin{cases} \frac{1}{3} & x = 1 \\ \frac{2}{3} & x = 0 \\ 0 & \textrm{otherwise} \end{cases}$.
# - Consider the sequence $\{X_n\}$ with terms $X_n = \left(1 + \frac{1}{n} \right) X$.
# - Since $|X_n - X| = \frac{1}{n} X$, $|X_n - X| = 0$ for $X = 0$ (with probability $\frac{2}{3}$)     
# and $|X_n - X| = \frac{1}{n}$ (with probability $\frac{1}{3}$). 
# - Thus, 
# $P(|X_n - X| \leq \epsilon) = \begin{cases} \frac{2}{3} & \frac{1}{n} < \epsilon \\ 1 & \textrm{otherwise} \end{cases}$
# - Let's pick an arbitrarily small $\epsilon$.
# - since $P(|X_n - X| > \epsilon) = 1 - P(|X_n - X| \leq \epsilon)$, $P(|X_n - X| > \epsilon) = 0 \: \forall \: \epsilon > 0$. 
# 
# 
# ### Mean-Square Convergence: 
# A sequence $\{X_n\}$ is *mean-square convergent* to $X$ 
# if $\lim\limits_{n \rightarrow \infty} d(X_n, X)  = 0$ 
# where $d(X_n, X) = \mathrm{E}[(X_n - X)^2]$.
# 
# ### Convergence in Distribution: 
# A sequence of random variable $\{X_n\}$ is *convergent in distribution* (or in law) 
# to random variable $X$ if their CDFs are 'close', 
# i.e. $\lim\limits_{n \rightarrow \infty} F_n(x) = F_X(x)$ for all points $x in \mathbb{R}$.
# 
# Example: 
# - Let $\{X_n\}$ be IID where each term has an uniform distribution on $[0,1]$. 
# - Thus, $F_{X_n}(x) = \begin{cases} 0 & x < 0 \\ x & 0 \leq x < 1\\ 1 & x \geq 1 \end{cases}$.
# - Consider the sequence $\{Y_n\}$ wjere $Y_n = n \left(1 - \max\limits_{1 \leq i \leq n}X_i \right)$.
# - Is $\{Y_n\}$ convergent in distribution?
# - The CDF of $Y_n$ is: 
# $\begin{align} F_{Y_n}(y) &= P(Y_n \leq y) \\ &= \ldots \\ &= 1 - F_{X_n}\left(1 - \frac{y}{n}\right)^n \end{align}$
# - $F_{Y_n}(y) = \begin{cases} 0 & y < 0 \\ 1 - \left(1 - \frac{y}{n} \right)^n & 0 \leq y < n\\ 1 & y \geq n \end{cases}$
# - $\lim\limits_{n \rightarrow \infty} F_{Y_n}(y) = F_Y(y) = \begin{cases} 0 & y < 0 \\ 1 - \mathrm{e}^{-y} & y \geq 0 \end{cases}$
# - $F_Y(y)$ can be shown to be a proper CDF (right-continuous, $-\infty$ limit is 0, $+\infty$ limit is 1).
# - Therefore, $\{Y_n\}$ converges in distribution to $Y$, an exponential random variable.
# 
#  
# ### Relations between the Modes of Convergence: 
# Almost-sure convergence, mean-square convergence $\Rightarrow$ Convergence in probability $\Rightarrow$ Convergence in distribution.
#  
# ---

# %%[markdown]
# ## Laws of Large Numbers (LLNs)
# 
# Let $\{X_n\}$ be a sequence of random variables 
# and $\bar{X}_n$ the sample mean of the first $n$ terms, 
# $\bar{X}_n = \frac{1}{n} \sum\limits_{i = 1}^n X_i$
#  
# A LLN states sufficient conditions that guarantees convergence of $\bar{X}_n$ to a constant as $n$ increases. 
# 
# A LLN is *weak* if $\bar{X}_n$ converges in probability, 
# *strong* if $\bar{X}_n$ converges almost surely.
# 
# ### Chebyshev's WLLN:
# - Let $\{X_n\}$ be uncorrelated and covariance stationary.
# - Then, $\bar{X}_n$ converges in mean square and thus in probability to $\mu$,
# i.e. $\lim\limits_{n \rightarrow \infty} P(|\bar{X}_n - \mu| > \epsilon) = 0$
# 
# 
# ### Chebyshev's WLLN for correlated sequences:
# - Let $\{X_n\}$ be covariance stationary 
# but $\lim\limits_{n \rightarrow \infty} \frac{1}{n} \sum\limits_{i = 0}^n \mathrm{cov}[X_n,X_{n-i}] = 0$.
# - Then, Chebyshev's WLLN applies.
# 
#
# ### Kolmogorov's SLLN:
# - Let $\{X_n\}$ be IID with finite mean $\mathrm{E}[X_n] = \mu < \infty \: \forall \: n \in \mathbb{N}$.
# - Then, $\bar{X}_n$ converges almost surely to $\mu$.
#
# 
# ### Ergodic Theory:
# - Let $\{X_n\}$ be just statioanry and ergodic with finite mean
# - Then, a SLLN applies to $\bar{X}_n$.
#  

# %%[markdown]
# ## Central Limit Theorems (CLTs)
#
# CLTs state conditions for the distribution of some function of the sample mean 
# to converge to a standard normal distribution as the sample size increases.
# 
# ### Lindeberg-Levy CLT:
# - Let $\{X_n\}$ be IID such that finite expectation value $\mu$ and variance $\sigma^2 > 0$ for all n
# - Then, $\sqrt{n} \left(\frac{\bar{X}_n - \mu}{\sigma} \right)$ 
# converges in distribution to some standard normal random variable $Z$.
# 
# 
# Example 1:
# - Let $\{X_n\}$ be a sequence of Bernoulli random variables with $p = \frac{1}{2}$. 
# - i.e. $R_{X_n} = \{0,1\}$ and $p_{X_n}(x) = \begin{cases} p & x = 1 \\ 1-p & x = 0 \\ 0 & otherwise \end{cases}$.
# - Use a CLT to find the distribution for the mean of the 1st 100 terms.
# 
# 
# - The sequence is IID and the mean of a generic term is $\mathrm{E}[X_n] = \sum\limits_{x in R_{X_n}} x \: p_{X_n}(x) = p < \infty$.  
# - The variance of a generic term is $\mathrm{var}[X_n] = \ldots = p - p^2 = \frac{1}{4} < \infty$.
# - Thus, the Lindeberg-Levy CLT applies.
# - $\bar{X}_100 = \frac{1}{100} \sum\limits_{i = 1}^{100} X_i$ 
# has a distribution that is normal with $\mu = \mathrm{E}[X_n] = \frac{1}{2}$ 
# and $\sigma = \frac{\mathrm{var}[X_n]}{n} = \frac{1}{400}$.
# 
# 
# Example 2: 
# - Let $Y$ be a binomial random variable with $n = 100, p = \frac{1}{2}$. 
# - Note that $Y = \sum\limits_{i = 1}^100 X_i$ 
# where $X_i$ are mutually independent Bernoulli random variables wih $p = \frac{1}{2}$.
# Since $\bar{X}_{100} \sim N(\mu = \frac{1}{2},\sigma^2 = \frac{1}{400})$,  
# $Y \sim N(\mu = \frac{100}{2}, \sigma^2 = \frac{100^2}{400}) = N(\mu = 50, \sigma^2 = 25)$. 

# %%[markdown]
# ## Continuous Mapping Theorem
#
# Stochastic convergence of some sequence of random vectors ${\boldsymbol{X}_n}$ to 
# a random vector $\boldsymbol{X}$ is preserved 
# if function $g:\mathbb{R}^K \rightarrow \mathbb{R}^L$ is a continuous function.  
# 
# That is, $g(\boldsymbol{X}_n)$ also converges to $g(\boldsymbol{X})$ 
# in probability, in distribution or almost surely.
# ---

# %%[markdown]
# ## Empirical Distribution
# 
# Let $\xi_n = [x_1 \: \ldots \: x_n]$ be a sample of size $n$ of some variable $X$; 
# $x_i$ is $i$-th observation from the sample. 
# The *empirical* distribution of $\Xi_n$ is the CDF of $X$ and it is defined as 
# $F_n(x) = \frac{1}{n}\sum\limits_{i=1}^n 1_{\{x_i \leq x \}}$.
# 
# In effect, this assigns a probability $\frac{1}{n}$ to each value $x_i$.
# 
# ### The Plug-In Principle:
# A feature of a given distribution can be approximated by the same feature of the empirical distribution 
# of a sample of observations drawn from the given distribution.
# 
# ### The Monte Carlo Method:
# This is a computational method that uses a generated sample from a given probability distribution 
# to produce a plug-in estimate of some of its features, e.g. a moment or a quantile.  
# 
# - Let $X$ be a random variable with CDF $F_X(x)$.
# - Generate a sample $\xi_n = [x_1 \: \ldots \: x_n]$ of realizations.
# - Denote a feature of $F_X$ (mean, variance, etc.) by $T(F_X)$.
# - Denote the empirical distribution of $\xi_n$ by $F_n(x)$.
# - Then, $T(F_n)$ is a *Monte Carlo* approximation of $T(F_X)$.
# 
# 
# Inverse Transformation Method: 
# If $U$ is a pseudo-random number having an uniform distribution on $[0,1]$ and 
# $F_X$ is an invertible CDF, then $X = F_X^{-1}(U)$ has CDF $F_X$.
# (see pseudo-random number sampling).
# 
#  


# %%
