# %% [markdown]
# # TensorFlow
# 
# Tutorial exercises from [TensorFlow.org](https://www.tensorflow.org/tutorials).

# %% [markdown] 
# ## Quickstart for Beginners 

# Install TensorFlow
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

# Import MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# %%
