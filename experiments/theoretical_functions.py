import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.utils import preprocess_user_item_df
import numpy as np
import pandas as pd
from model.wide_and_deep import WideAndDeep
from model.encoder import UtilityEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import math


UTILITY = sys.argv[1]

print("Running simulation with {} utility".format(UTILITY))


RANDOM_SEED = 1990
N_USERS = 1000
N = 64
N_SIM = 8
RHO = 2

assert UTILITY in ['cobb-douglas', 'ces']

np.random.seed(RANDOM_SEED)

#weights = np.transpose(np.random.multivariate_normal(np.zeros(N), np.eye(N), 1)).flatten()
weights = np.random.uniform(0, 10, N)
weights = weights / np.sum(weights)

def mrs_error(M1, M2):

    assert M1.shape == M2.shape

    n = M1.shape[0]

    idx = np.triu_indices(n)

    m1 = M1[idx]
    m2 = M2[idx]

    mse = math.sqrt(mean_squared_error(m1, m2))
    return mse

def get_analytical_cobb_douglas_mrs(w1, w2):

    #return w1 + w2
    return -w1/w2

def get_analytical_ces_mrs(w1, w2, rho):

    return - (w1 / w2)**(rho-1)

def get_mrs_mat(x, w, mrs_func, rho=None):
    n = x.shape[0]
    mrs_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            if rho is None:
                mrs_mat[i, j] = mrs_func(w1=w[i], w2=w[j])
            else:
                mrs_mat[i, j] = mrs_func(w1=w[i], w2=w[j], rho=rho)

    return mrs_mat


def compute_pariwise_mrs(grad):

    n = len(grad)
    mrs_mat = np.zeros((n, n))

    for i, g_i in enumerate(grad):
        for j, g_j in enumerate(grad):

            if g_j == 0.0:
                g_j = 1e-3

            mrs_mat[i, j] = - (g_i / g_j)

    return mrs_mat



def logit(x):
    return 1 / (1 + np.exp(-x))


def cobb_douglas(x, w):
    # TODO: Think more about this. This assumes that we are already at one. What's the utility at 2?
    eps = 1.0
    log_x = np.log(x + eps)
    log_u = np.dot(log_x, w) + np.random.normal(0, 1, 1)[0]
    u = np.exp(log_u)

    return u

def ces(x, w, rho):

    x_power = np.power(x, rho)
    inner_prod = np.dot(x_power, w)
    u = np.power(inner_prod, rho)
    return u


def gen_bundle(n, k):
    idx = np.random.randint(0, n, k)
    x = np.zeros(n)

    for i in idx:
        x[i] = 1.0

    return x, idx


def gen_dataset(n_items, n_users, n_reviews, utility_type):

    n_samples = n_users * n_reviews

    X = np.zeros((n_samples, 2), dtype=np.int32)
    y = np.zeros((n_samples, 1), dtype=np.float32)

    cntr = 0
    for i in range(n_users):
        items_reviews_i = np.random.permutation(n_items)[:n_reviews]
        for j in items_reviews_i:

            x_vec, x_i = gen_bundle(N, 1)

            if utility_type == 'cobb-douglas':
                u = cobb_douglas(x_vec, weights)
            elif utility_type == 'ces':
                u = ces(x_vec, weights, RHO)

            X[cntr, 0] = i


            X[cntr, 1] = j
            y[cntr] = u

            cntr += 1

    return X, y


def clip(arr, k):

    n = len(arr)
    arr.sort()
    return arr[:n-k]



X, y = gen_dataset(N, N_USERS, N // 2, utility_type=UTILITY)

print(X.shape)

if UTILITY == 'cobb-douglas':
    MRS = get_mrs_mat(x=np.arange(N), w=weights, mrs_func=get_analytical_cobb_douglas_mrs)
else:
    MRS = get_mrs_mat(x=np.arange(N), w=weights, mrs_func=get_analytical_ces_mrs, rho=RHO)

df = pd.DataFrame(np.concatenate([X, y], axis=1), columns=['user_id', 'item_id', 'rating'])
df, user_item_rating_map, item_rating_map, user_id_map, id_user_map, item_id_map, id_item_map, stats = preprocess_user_item_df(df)

X = df[['user_id', 'item_id']].astype(np.int64)
y = df['rating']
#X_train, X_test, y_train, y_test = split_train_test_user(X, y, random_seed=1990)




### Train Models

batch_size = 32
k = 5
h_dim = 16
n_epochs = 10
lr = 5e-5
loss_step = 10
eps = 0.01

output = {"linear": [],
          "vanilla": [],
          "utility": []}

for iter in range(N_SIM):

    # Train Linear Regression
    enc = OneHotEncoder(sparse=False)
    X_train_sparse = enc.fit_transform(X=X['item_id'].values.reshape(-1,1))

    linear = LinearRegression(fit_intercept=True)
    linear.fit(X_train_sparse, y)



    grad_linear = linear.coef_
    mrs_linear = compute_pariwise_mrs(grad_linear)
    l2_linear = mrs_error(MRS, mrs_linear)
    output['linear'].append(l2_linear)


    # Train Vanilla Wide&Deep
    item_encoder = UtilityEncoder(n_items=stats['n_items'], h_dim_size=h_dim)

    item_encoder.fit(X, y, batch_size, lr, n_epochs, loss_step, eps)

    grad_vanilla = item_encoder.get_input_grad(np.arange(N))
    mrs_vanilla = compute_pariwise_mrs(grad_vanilla.data.numpy())
    l2_vanilla = mrs_error(MRS, mrs_vanilla)
    output['vanilla'].append(l2_vanilla)



    # Train Neural Utility Function
    item_encoder_utility = UtilityEncoder(n_items=stats['n_items'], h_dim_size=h_dim)
    item_encoder_utility.fit_utility_loss(X, y, batch_size, lr, n_epochs, loss_step, eps, user_item_rating_map, \
                                item_rating_map, k, stats['n_items'])

    grad_utility = item_encoder_utility.get_input_grad(np.arange(N))
    mrs_utility = compute_pariwise_mrs(grad_utility.data.numpy())
    l2_utility = mrs_error(MRS, mrs_utility)
    output['utility'].append(l2_utility)



for k, v in output.items():

    print(k)


    v_clip = clip(v, k=2)

    print(v_clip)
    print("mean: {:.4f}".format(np.mean(v_clip)))
    print("std: {:.4f}".format(np.std(v_clip)))