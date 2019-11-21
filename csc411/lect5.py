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
# 
# 
#  