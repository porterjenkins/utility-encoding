import numpy as np
from experiments.utils import permutationTest




x = np.array([0.7273 , 0.7302,0.7302])
mean_x = x.mean()
std_x = x.std()

y =  np.array([0.7050, 0.7482])
mean_y = y.mean()
std_y = y.std()

mean_diff = mean_y - mean_x

print("Difference in means: {:.4f}".format(mean_diff))
p = permutationTest(x, y, 1000)
print(p)
from scipy.stats import ttest_ind

t, p_val  = ttest_ind(x, y)
print(p_val)