# %% [markdown]
# # Prime Numbers with UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction
# 
# UMAP: Uniform Manifold Approximation and Projection
# 
# Following [McInnes](https://umap-learn.readthedocs.io/en/latest/sparse.html) 
# and [Williamson](https://johnhw.github.io/umap_primes/index.md.html).
#
# Represent integers by a vector of their divisibility ($0$ or $1$) by a list of prime numbers.
#
# The space of prime numbers is very large but most integers are only divisible by a few. 
# This is an example of sparse dataset. 

# %%
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import sympy
import scipy.sparse
import umap as umap

# %% 
# List of prime numbers
t = time()
max_int = 10000
prime_num = list(sympy.primerange(2, max_int))
n_primes = len(prime_num)
print(f'{n_primes} prime numbers within the first {max_int} integers ({time() - t:.3f} s).')

# Map prime numbers to column indices
prime_to_col = {p: i for i, p in enumerate(prime_num)}

# Use *list of lists* to store sparse data
t = time()
lil_mat_rows = []
lil_mat_data = []
k = 0
int1 = 0
int2 = max_int
for i in range(int1, int2):
    
    # list of prime factors
    j = sympy.primefactors(i)

    # list of non-zero column indices and matrix values
    lil_mat_rows.append([prime_to_col[p] for p in j])
    lil_mat_data.append([1] * len(j))

    k += len(j)

print(f'{k} non-zero entries in the {int1}-{int2} prime factor matrix ({time() - t:.3f} s).')

# Build sparse matrix
primefactors_mat = scipy.sparse.lil_matrix((len(lil_mat_rows), len(lil_mat_data)), dtype=np.float32)
primefactors_mat.rows = np.array(lil_mat_rows)
primefactors_mat.data = np.array(lil_mat_data)

# Apply UMAP
t = time()
prime_umap = umap.UMAP(n_neighbors = 15, n_components = 2, min_dist = 0.1, metric='cosine', random_state = 0).fit(primefactors_mat)
print(f'Elapsed time for UMAP fit: {time() - t} s')

# %%
# Plot results
fig = plt.figure(figsize = (6, 6), facecolor = 'k')
ax = fig.add_subplot(111)
ax.scatter(prime_umap.embedding_[:, 0], prime_umap.embedding_[:, 1], s = 1, c = range(int1, int2), linewidth = 0, marker = '.')
plt.setp(ax, facecolor = 'k')
plt.setp(ax.spines.values(), color = 'w')
ax.tick_params(axis='both', colors='w')
# plt.setp([ax.get_xticklines(), ax.get_yticklines()], color = 'w')

# %%

# Apply UMAP (3d)
t = time()
prime_umap_3d = umap.UMAP(n_neighbors = 15, n_components = 3, min_dist = 0.1, metric='cosine', random_state = 0).fit(primefactors_mat)
print(f'Elapsed time for UMAP fit: {time() - t} s')

# %% 

fig = plt.figure(figsize = (6, 6))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(prime_umap_3d.embedding_[:, 0], prime_umap_3d.embedding_[:, 1], prime_umap_3d.embedding_[:, 2], s = 1, c = range(int1, int2), marker = '.', linewidth = 0)




