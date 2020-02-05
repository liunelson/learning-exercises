# %% [markdown]
# # Hidden Markov Models
# 
# Following [M. Stamp, "A Revealing Introduction to Hidden Markov Models," (2018)](http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf).
#  

# %% 

# Import modules
import numpy as np
import matplotlib.pyplot as plt

# import nltk
from urllib import request
import random

# %%

# Read Jane Austen's Pride and Prejudice
url = "https://www.gutenberg.org/files/1342/1342-0.txt"
response = request.urlopen(url)
txt_raw = response.read().decode('utf-8-sig')

url = None
response = None
del url, response

# All lowercase
txt_raw = txt_raw.casefold()

# Make character list
# txt_list = list(txt_raw)

# Make vocabulatory list
txt_voc = list(dict.fromkeys(txt_raw))

# %%

def init_param(num_states = None, num_obs = None, rand_seed = 1):

    # Initialize RNG
    random.seed(rand_seed)

    # Declare HMM parameters
    prob_trans = np.ones((num_states, num_states)) / num_states
    prob_emiss = np.ones((num_obs, num_states)) / num_obs
    prob_init = np.ones((num_states, )) / num_states

    # Generate perturbations 
    sig = 0.01
    prob_trans = prob_trans * (1.0 - np.random.normal(0.0, sig, prob_trans.shape))
    prob_emiss = prob_emiss * (1.0 - np.random.normal(0.0, sig, prob_emiss.shape))
    prob_init = prob_init * (1.0 - np.random.normal(0.0, sig, prob_init.shape))

    # Normalize to ensure column-stochasticity
    prob_trans[0, :] = prob_trans[0, :] + (1.0 - prob_trans.sum(axis = 0))
    prob_emiss[0, :] = prob_emiss[0, :] + (1.0 - prob_emiss.sum(axis = 0))
    prob_init[0] = prob_init[0] + (1.0 - prob_init.sum(axis = 0))

    return prob_trans, prob_emiss, prob_init

# %%

# Forward-/alpha-pass algorithm
def forward_pass(prob_trans, prob_emiss, prob_init, Y):

    num_states = prob_trans.shape[0]
    num_obs = prob_emiss.shape[0]
    num_time = Y.shape[0]
    
    # Calculate alpha_n(t) = Pr(Y(0), Y(1), ..., Y(t), X(t) = x_n | theta)
    alpha = np.zeros((num_states, num_time))
    for t in range(num_time):

        if t == 0:
            m = Y[t]
            alpha[:, t] = prob_emiss[m, :] * prob_init

        else:
            m = Y[t]
            alpha[:, t] = prob_emiss[m, :] * np.dot(prob_trans, alpha[:, t - 1])


    # Calculate scaling factor c(t) = 1 / (\sum_n alpha_n(t))
    c = 1.0 / alpha.sum(axis = 0)

    # Scale alpha_n(t) with c(t)
    alpha = np.dot(alpha, np.diag(c))

    return alpha, c

def backward_pass(prob_trans, prob_emiss, prob_init, Y, c):

    num_states = prob_trans.shape[0]
    num_obs = prob_emiss.shape[0]
    num_time = Y.shape[0]

    # Calculate beta_n(t) = Pr(Y(t+1), Y(t+2), ..., Y(T-1) | X(t) = x_n, theta)
    beta = np.zeros((num_states, num_time))
    for t in range(num_time - 1, 0 - 1, -1):

        if t == (num_time - 1):
            beta[:, t] = 1.0

        else:
            m = Y[t + 1]
            beta[:, t] = np.dot(prob_trans, prob_emiss[m, :] * beta[:, t + 1])

    # Scale beta_n(t) with c(t)
    beta = np.dot(beta, np.diag(c))

    return beta

def calc_gammas(prob_trans, prob_emiss, prob_init, Y, alpha, beta):

    num_states = prob_trans.shape[0]
    num_obs = prob_emiss.shape[0]
    num_time = Y.shape[0]

    # Calculate digamma_(n,n')(t) = Pr(X(t) = x_n, X(t+1) = x_n' | Y, theta)
    digamma = np.zeros((num_states, num_states, num_time))
    for t in range(num_time - 1):

        m = Y[t]

        for i in range(num_states):
            for j in range(num_states):
                digamma[i, j, t] = alpha[i, t] * prob_trans[i, j] * prob_emiss[m, j] * beta[j, t + 1]


    # Calculate gamma_n(t) = Pr(X(t) = x_n | Y, theta)
    gamma = np.zeros((num_states, num_time))
    gamma = digamma.sum(axis = 1)
    gamma[:, num_time - 1] = alpha[:, num_time - 1]

    return gamma, digamma

def estim_param(Y, gamma, digamma):

    num_states, num_time = gamma.shape
    num_obs = Y.shape[0]

    # Estimate prob_init: gamma[:, 0]
    prob_init_est = np.zeros((num_states, ))
    prob_init_est = gamma[:, 0]

    # Estimate prob_trans
    prob_trans_est = np.zeros((num_states, num_states))
    numer = digamma.sum(axis = 2)
    denom = gamma[:, 0:(num_time - 1)].sum(axis = 1)
    for j in range(num_states):
        prob_trans_est[:, j] = numer[:, j] / denom[j]
    
    # Estimate prob_emiss
    prob_emiss_est = np.zeros((num_obs, num_states))
    for m in range(num_obs):

        numer = gamma[:, Y == m].sum(axis = 1)
        denom = gamma.sum(axis = 1)
        prob_emiss_est[m, :] = numer / denom

    return prob_trans_est, prob_emiss_est, prob_init_est


def forward_backward_pass(Y, num_states = None, num_obs = None, rand_seed = 1, maxIter = 1e3, log_tol = 1e-3):

    num_iter = 0
    log_prob = 0.0
    log_prob_ = -np.inf
    while (num_iter < maxIter) & (log_prob > log_prob_):

        # Initialize parameters
        if num_iter == 0:
            prob_trans, prob_emiss, prob_init = init_param(num_states, num_obs, rand_seed)

        # Forward pass
        alpha, c = forward_pass(prob_trans, prob_emiss, prob_init, Y)

        # Calculate ln Pr(Y | theta)
        log_prob = -(np.log(c)).sum(axis = 0)

        # Check convergence
        if log_prob > log_prob_:

            # Raise lower bound
            log_prob_ = log_prob
            log_prob += log_tol

            # Backward pass
            beta = backward_pass(prob_trans, prob_emiss, prob_init, Y, c)

            # Gammas
            gamma, digamma = calc_gammas(prob_trans, prob_emiss, prob_init, Y, alpha, beta)
            
            # Estimate parameters
            prob_trans_est, prob_emiss_est, prob_init_est = estim_param(Y, gamma, digamma)

        else:
            print(f'Optimization complete after {num_iter} steps (log_prob = {log_prob_}).')

        num_iter += 1

    return prob_trans_est, prob_emiss_est, prob_init_est

# %%

prob_trans_est, prob_emiss_est, prob_init_est = forward_backward_pass(Y, num_states = 2, num_obs = 5)


# %%
