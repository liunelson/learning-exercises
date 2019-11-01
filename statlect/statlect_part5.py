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
# 
# 
# 
# 
#  
