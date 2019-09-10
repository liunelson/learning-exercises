#%% [markdown]
# # Notes and exercises from [Nielsen2015](http://neuralnetworksanddeeplearning.com/)
# --- 

#%% [markdown]
# ## Ch. 6 Deep Learning
#
# Deep neural networks are hard to train due to inherent instability of 
# the gradient-descent algorithm by backpropagation.
#
# Let's consider ways to fix this problem.

#%% [markdown]
# ### Sec. 6.1 Convolutional Neural Networks
#
# Previously, a fully connected neural network was used 
# for the MNIST handwritten digit classification problem.
#
# However, it is not very physical since it treated nearby pixels 
# the same as far-apart pixels.
#
# Solution: a *convolutional neural network* to account spatial structure.
#
# - *Local receptive fields*: 
# local subsets of units to which a single hidden unit is connected; 
# it is defined by the *kernel size* (side length of subregion), 
# *padding size* (length of padding outside), and 
# *stride size* (step size of convolution).
#
# e.g. a MNIST image has $N_\textrm{in} \times N_\textrm{in}$ pixels; 
# assuming a local receptive field with kernel size $K$, padding size $P = 0$, 
# and stride size $S = 1$,  
# then, the first hidden layer has $N_\textrm{out} \times N_\textrm{out}$ units.
#
# $N_\textrm{out} = \left \lfloor \frac{N_\textrm{in} + 2P - K}{S} \right \rfloor + 1$
#
# - *Shared weights and biases*: 
# All the hidden units use the same weights and biases.
#
# The output of the $(j,k)$-th hidden unit is 
# $\sigma\left( \sum\limits_{l,m = 0}^K w_{l,m} \: a_{j+l,k+m} + b\right)$
#
# The map from the input layer to the hidden layer is known as 
# a *feature map* with some *kernel* or *filter* defined by *shared weights* $w$ 
# and *shared bias* $b$.

# There may be multiple feature maps defined by sets of $w$ and $b$, 
# each detecting different kinds of features (that are translationally invariant).
# 
# Overall, the number of parameters ($w$, $b$) are greatly reduced. 
# For each feature map, there are only $N_\textrm{out}^2 + 1$. 
# A fully connected network would have $(N_\textrm{in}^2 + 1) N_1$, 
# where $N_1$ is the number of hidden units.
# 
# *Convolutional* because activation can be written as 
# $a^1 = \sigma(b + w \astar a^0)$.
#
# - *Pooling layers*:
# Layer that comes after the convolutional layer.
# 
# A pooling layer downsamples the previous output, 
# mapping several nearby hidden units to a single pooling unit, 
# as a mean to reduce the number of parameters for later layers.
#  
# A common technique is *max-pooling*, 
# wherein the output is simply the maximum activation value 
# in a given subregion (e.g. $2 \times 2$).
# 
# Another technique is *$L^2$ pooling*, 
# wherein the $L^2$ norm is taken.
# 
# Two hyperparameters: pool size (side length of the pooled region) 
# and stride size (step size). 
# If pool size $>$ stride size, it is called *overlapping* pooling.
#  

#%% [markdown]
# An example of a convolutional neural network for the MNIST problem: 
# 
# 1. Input layer with $28 \times 28$ units, one for each image pixel
# 2. Convolutional layer ($5 \times 5$ kernel, three feature maps) with $24 \times 24 \times 3$ outputs
# 3. Pooling layer ($2 \times 2$ kernel) with $12 \times 12 \times 3$ units
# 4. Output layer (fully connected, $12 \times 12 \times 3 \times 10$ weights and $10$ biases) with 10 units 
#  
# Training is done as previously, using stochastic gradient descent   
# and backpropagation.
#  
#  
# 

