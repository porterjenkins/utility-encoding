import numpy as np


def permutationTest(x,y,nperm, method = 'mean', twoSided=True):

    np.random.seed(seed=None)
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array)
    perm_ts = np.zeros(nperm)
    all_obs = np.concatenate([x_array,y_array])

    # Permutation Test: Difference of Means #
    if method == 'mean':
        # If twoSided is True; Perform two sided hypothesis test #
        if twoSided:
            obs_diff = abs(np.mean(x_array) - np.mean(y_array))
        # Otherwise perform one sided test: Ho: median X = median Y, Ha: median X > median Y
        else:
            obs_diff = np.mean(x_array) - np.mean(y_array)

        for i in range(nperm):
            np.random.shuffle(all_obs)
            perm_x = all_obs[:n]
            perm_y = all_obs[n:]
            if twoSided:
                perm_ts[i] = abs(np.mean(perm_x) - np.mean(perm_y))
            else:
                perm_ts[i] = np.mean(perm_x) - np.mean(perm_y)

    # Permutation Test: Difference of Medians #

    elif method == 'median':
        # If twoSided is True; Perform two sided hypothesis test #
        if twoSided:
            obs_diff = abs(np.median(x_array) - np.median(y_array))
        # Otherwise perform one sided test: Ho: median X = median Y, Ha: median X > median Y
        else:
            obs_diff = np.median(x_array) - np.median(y_array)

        for i in range(nperm):
            np.random.shuffle(all_obs)
            perm_x = all_obs[:n]
            perm_y = all_obs[n:]
            if twoSided:
                perm_ts[i] = abs(np.median(perm_x) - np.median(perm_y))
            else:
                perm_ts[i] = np.median(perm_x) - np.median(perm_y)
    else:
        raise Exception('Method for permutation must be mean or median')

    pval = len(perm_ts[perm_ts > obs_diff])/nperm
    return pval



x = np.random.normal(0, 25, 10)
y = np.random.normal(10, 25, 10)
p = permutationTest(x, y, 1000)
print(p)

from scipy.stats import ttest_ind

t, p_val  = ttest_ind(x, y)

print(p_val)