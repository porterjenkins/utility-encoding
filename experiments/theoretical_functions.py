import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.utils import preprocess_user_item_df
import numpy as np
import pandas as pd


RANDOM_SEED = 1990
N_USERS = 5
N = 10
K = 3

#weights = np.transpose(np.random.multivariate_normal(np.zeros(N), np.eye(N), 1)).flatten()
weights = np.random.uniform(0, 10, N)
weights = weights / np.sum(weights)

def get_analytical_mrs(w1, w2):

    return w1 + w2

def mrs_mat(x, w):
    n = x.shape[0]
    mrs_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):


            mrs_mat[i, j] = get_analytical_mrs(w1=w[i], w2=w[j])

    return mrs_mat





def logit(x):
    return 1 / (1 + np.exp(-x))


def cobb_douglas(x, w):
    eps = 1e-4
    log_x = np.log(x + eps)
    log_u = np.dot(log_x, w) + np.random.normal(0, 1, 1)[0]
    #u = np.exp(log_u)

    return log_u



def gen_bundle(n, k):
    idx = np.random.randint(0, n, k)
    x = np.zeros(n)

    for i in idx:
        x[i] = 1.0

    return x, idx


def gen_dataset(N, size, users):
    n_samples = users*size

    X = np.zeros((n_samples, 2), dtype=np.int32)
    y = np.zeros((n_samples, 1), dtype=np.float32)

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

MRS = mrs_mat(x=np.arange(N), w=weights)

df = pd.DataFrame(np.concatenate([X, y], axis=1), columns=['user_id', 'item_id', 'rating'])
df, user_item_rating_map, item_rating_map, user_id_map, id_user_map, item_id_map, id_item_map, stats = preprocess_user_item_df(df)


print(df)

