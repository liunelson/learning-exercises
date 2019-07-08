#%% [markdown]
# # Notes and exercises from [Nielsen2015](http://neuralnetworksanddeeplearning.com/)
# --- 

#%% [markdown]
# ## Ch. 1.1 - Perceptrons
# 
# $ y = 
#   \begin{cases} 
#       0 & \text{if } \sum_i w_i x_i \geq \text{threshold} \\ 
#       1 & \text{if } \sum_i w_i x_i < \text{threshold}   
#   \end{cases} $ 
#
# where $y = \text{output}$, $x_i = \text{input}$, and $w_i = \text{weights}$. 
#
# Rewrite conditions as $\boldsymbol{w} \cdot \boldsymbol{x} + b \leq 0$, $> 0$    
# where $b = -\text{threshold}$ is the *bias*.
# 
# Perceptrons can be combined to give $\text{AND}, \text{OR}, \text{NAND}$, etc. 
# and perform any basic computation.

#%% [markdown]
# ## Ch. 1.2 - Logistic/Sigmoid Neurons
# 
# To optimize $w_i$, the output $y$ should be defined by 
# a differentiable *transfer* or *activation* function. 
#
# Let's replace it with the *logistic* or *sigmoid* function 
# $f(z) = \frac{1}{1 + \mathrm{e}^{-z}}$ 
# where $z = \boldsymbol{w} \cdot \boldsymbol{x} + b$.
#
# Thus, 
# $\Delta y \approx \sum_i \frac{\partial y}{\partial w_i} \Delta w_i + \frac{\partial y}{\partial b} \Delta b$ 

import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 100)
f = 1/(1 + np.exp(-z))

plt.plot(z, f)
plt.xlabel('$z$')
plt.ylabel('$f(z)$')
plt.title('Logistic Function')
plt.rc('grid', linestyle = '--', color = 'gray')
plt.grid(True)
plt.show()

#%% [markdown]
# ## Ch. 1.3 - Architecture of Neural Networks
# 
# Layers: *input*, *hidden*, and *output*. 
#
# *Feedforward* neural networks: no loop back. 
# *Recurrent* neural networks: feedback loops allowed.

#%% [markdown]
# ## Ch. 1.4 - Simple Network to Classify Handwritten Digits
#
# Read handwritten digits = *segmentation* + *classification*.
# 
# Exercise: classify greyscale images of handwritten digits ($28 \times 28$ pixels) in 
# [MNIST](http://yann.lecun.com/exdb/mnist/) dataset 
# (60,000 labeled training images, 10,000 labeled test images)
# 
# | Filename | Content | Size |
# | --- | --- | --- |
# | `train-images-idx3-ubyte.gz` | training set images | $9912422$ bytes | 
# | `train-labels-idx1-ubyte.gz` | training set labels | $28881$ bytes |
# | `t10k-images-idx3-ubyte.gz` | test set images | $1648877$ bytes | 
# | `t10k-labels-idx1-ubyte.gz` | test set labels | $4542$ bytes |

#%%
# Approach: 784 nodes in input layer, $n = 15$ nodes in single hidden layer, 
# 10 nodes in output layer ($0 \ldots 9$). 
