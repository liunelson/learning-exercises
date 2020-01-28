# %% [markdown]
# # Intro to Pandas 
#
# Introduction to the `pandas` Python module following 
# [Google's tutorial](https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb).

import pandas as pd
print(f'Pandas version: {pd.__version__}') 

# Contruct a `Series` object
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
city_pop = pd.Series([852469, 1015785, 485199])

# `DataFrame` object constructed with a `dict`
cities = pd.DataFrame({'Name': city_names, 'Population': city_pop})


# Alternative: load a file into a dataframe
ca_housing = pd.read_csv('https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv', sep = ',')

# `DataFrame.describe()` shows relevant stats
# (count, mean, std, min, max, Q1-Q3)
ca_housing.describe()

# `DataFrame.hist()` plots a histogram of the data
ca_housing.hist('housing_median_age')
ca_housing['housing_median_age'].hist()

# `DataFrame` data can be accessed using usual dict/list operations
ca_housing['housing_median_age']
ca_housing['housing_median_age'][0]

#`Series` objects can be used with most NumPy functions
(np.log(ca_housing['housing_median_age'])).hist()

# For more complex transformations, use `Series.apply` as in `map`
ca_housing['housing_median_age'].apply(lambda val: np.log(val))

# Add new `Series` to an existing `DataFrame`
cities['Surface Area'] = pd.Series([46.87, 176.53, 97.92])
cities['Population Density'] = cities['Population'] / cities['Surface Area']

# Exercise: check if named after saint and is greater than 50 in surface area
import re
i = cities['Name'].apply(lambda val: bool(re.match('San|Saint', val)))
j = cities['Surface Area'] > 50
i & j

# `index` property
for i in cities.index:
    print(cities['Name'][i])

cities.reindex([2, 0, 1])
cities.reindex(np.random.permutation(cities.index))
