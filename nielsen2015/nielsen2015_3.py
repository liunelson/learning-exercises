#%% [markdown]
# # Notes and exercises from [Nielsen2015](http://neuralnetworksanddeeplearning.com/)
# --- 

#%% [markdown]
# ## Ch. 3 Improvements
#
# ## Sec. 3.1 The Cross-Entropy Cost Function
#
# Learning slows down since $\frac{\partial C}{\partial w}$ 
# and $\frac{\partial C}{\partial b}$ scale linearly with $\sigma^\prime(z)$, 
# which is nearly flat $\forall z > 0$.
#
# Solution: Use a different cost function.
#
# 

#%%
