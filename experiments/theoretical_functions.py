import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.utils import preprocess_user_item_df
import numpy as np
import pandas as pd


RANDOM_SEED = 1990
N_USERS = 5
N = 25
K = 3

weights = np.transpose(np.random.multivariate_normal(np.zeros(N), np.eye(N), 1))
w_normed = weights / np.sum(weights)


def logit(x):
    return 1 / (1 + np.exp(-x))


def cobb_douglas(x, w):
    log_u = np.dot(x, w)
    prob = logit(log_u)
    if prob <= .5:
        return 1.0
    else:
        return 0.0



def gen_bundle(n, k):
    idx = np.random.randint(0, n, k)
    x = np.zeros(n)

    for i in idx:
        x[i] = 1.0

    return x, idx


def gen_dataset(N, size, users):
    n_samples = users*size

    X = np.zeros((n_samples, 2), dtype=np.int32)
    y = np.zeros((n_samples, 1), dtype=np.int32)

    cntr = 0
    for i in range(users):
        for j in range(size):

            x_vec, x_i = gen_bundle(N, 1)
            u = cobb_douglas(x_vec, weights)

            X[cntr, 0] = i


            X[cntr, 1] = x_i[0]
            y[cntr] = u

            cntr += 1

    return X, y


X, y = gen_dataset(N, 1000, N_USERS)

df = pd.DataFrame(np.concatenate([X, y], axis=1), columns=['user_id', 'item_id', 'rating'])


df, user_item_rating_map, item_rating_map, user_id_map, id_user_map, item_id_map, id_item_map, stats = preprocess_user_item_df(df)

