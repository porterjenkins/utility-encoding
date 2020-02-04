import numpy as np
from experiments.utils import permutationTest




x = np.array([10.7748, 10.8403, 10.8331])
mean_x = x.mean()
std_x = x.std()

y =  np.array([10.9319, 10.9311, 10.9257])
mean_y = y.mean()
std_y = y.std()

print(mean_x, std_x)
print(mean_y, std_y)

p = permutationTest(x, y, 1000)
print(p)
from scipy.stats import ttest_ind

t, p_val  = ttest_ind(x, y)
print(p_val)