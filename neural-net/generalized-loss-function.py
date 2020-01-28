# %% [markdown]
# # A General and Adaptive Robust Loss Function
# 
# [J. T. Barron, "A General and Adaptive Robust Loss Function," arXiv:1701.03077v10 [cs.CV]](https://arxiv.org/abs/1701.03077).
# 
# $\begin{align} f(x, \alpha, c) & = \frac{| \alpha - 2 |}{\alpha} \left( \left( \frac{ \left( x/c \right)^2 }{| \alpha - 2| } + 1 \right)^{\alpha/2} - 1\right) \\ & = \begin{cases} \frac{1}{2} \left( x/c \right)^2 & \textrm{if } \alpha = 2 \\ \log \left( \frac{1}{2} \left( x/c \right)^2 + 1\right) & \textrm{if } \alpha = 0 \\ 1 - \exp \left( -\frac{1}{2} \left( x/c \right)^2 \right) & \textrm{if } \alpha = -\infty \\ \frac{| \alpha - 2 |}{\alpha} \left( \left( \frac{ \left( x/c \right)^2 }{| \alpha - 2| } + 1 \right)^{\alpha/2} - 1\right) & \textrm{otherwise} \end{cases} \end{align}$

# %%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generalized loss function
def loss_func(x, alpha, c):

    L = np.zeros(x.shape)

    if alpha == 2.0:
        L = 0.5 * (x / c)**2
    elif alpha == 0.0:
        L = np.log(0.5 * (x / c)**2 + 1)
    else:
        L = ( np.abs(alpha - 2) / alpha ) * ( ( (x / c)**2 / np.abs(alpha - 2) + 1 )**(0.5 * alpha) - 1)

    return L

# Alternate function with compact support
def loss_func_atan(x, atan_alpha, c):

    L = np.zeros(x.shape)

    if atan_alpha == np.arctan(2.0):
        L = 0.5 * (x / c)**2
    elif atan_alpha == 0.0:
        L = np.log(0.5 * (x / c)**2 + 1)
    elif atan_alpha == -0.5*np.pi:
        L = 1.0 - np.exp(- 0.5 * (x / c)**2)
    else:
        alpha = np.tan(atan_alpha)
        L = ( np.abs(alpha - 2) / alpha ) * ( ( (x / c)**2 / np.abs(alpha - 2) + 1 )**(0.5 * alpha) - 1)

    return L

atan_alpha = np.arange(-0.5 * np.pi, np.arctan(2.0), 0.1)
x = np.arange(-5.0, 5.0, 0.01)
X, Y = np.meshgrid(x, atan_alpha)
Z = np.asarray([loss_func_atan(i, j, 1.0) for i, j in zip(X.reshape(X.size, 1), Y.reshape(Y.size, 1))])
Z = Z.reshape(X.shape)

fig = plt.figure(figsize = (10, 5))
ax = fig.add_subplot(121, projection = '3d')
surf = ax.plot_surface(X, Y, Z, cmap = 'cividis', linewidth = 0, antialiased = False, vmin = 0.0, vmax = 5.0)
ax.set_ylim(-0.5*np.pi, np.arctan(2.0))
ax.set_zlim(0.0, 5.0)
plt.setp(ax, xlabel = '$x / c$', ylabel = '$\\arctan(\\alpha)$')

ax = fig.add_subplot(122)
for i, k in zip([-np.inf, -2.0, 0.0, 1.0, 2.0], ['$\\alpha = -\\infty$ (Welsch/Leclerc)', '$\\alpha = -2$ (Geman-McClure)', '$\\alpha = 0$ (Cauchy/Lorentz)', '$\\alpha = 1$ (Charbonnier/L$_1$-L$_2$)', '$\\alpha = 2$ (L$_2$)']):
    j = np.arctan(i)
    ax.plot(x, loss_func_atan(x, j, 1.0), label = k)
ax.set_ylim(0.0, 5.0)
ax.set_xlim(-5.0, 5.0)
ax.legend()
_ = plt.setp(ax, xlabel = '$x / c$', ylabel = '$L(x, \\alpha, 1)$')


# %%
