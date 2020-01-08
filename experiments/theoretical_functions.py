import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.utils import preprocess_user_item_df
import numpy as np
import pandas as pd
from model.wide_and_deep import WideAndDeep
from preprocessing.utils import split_train_test_user
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

RANDOM_SEED = 1990
N_USERS = 1000
N = 10

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

    mse = mean_squared_error(m1, m2)
    return mse

def get_analytical_mrs(w1, w2):

    #return w1 + w2
    return -w1/w2

def mrs_mat(x, w):
    n = x.shape[0]
    mrs_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):


            mrs_mat[i, j] = get_analytical_mrs(w1=w[i], w2=w[j])

    return mrs_mat

def compute_pariwise_mrs(grad):

    n = len(grad)
    mrs_mat = np.zeros((n, n))

    for i, g_i in enumerate(grad):
        for j, g_j in enumerate(grad):

            mrs_mat[i, j] = - (g_i / g_j)

    return mrs_mat



def logit(x):
    return 1 / (1 + np.exp(-x))


def cobb_douglas(x, w):
    eps = 1.0
    log_x = np.log(x + eps)
    log_u = np.dot(log_x, w) + np.random.normal(0, 1, 1)[0]
    u = np.exp(log_u)

    return u



def gen_bundle(n, k):
    idx = np.random.randint(0, n, k)
    x = np.zeros(n)

    for i in idx:
        x[i] = 1.0

    return x, idx


def gen_dataset(n_items, n_users, n_reviews):

    n_samples = n_users * n_reviews

    X = np.zeros((n_samples, 2), dtype=np.int32)
    y = np.zeros((n_samples, 1), dtype=np.float32)

    cntr = 0
    for i in range(n_users):
        items_reviews_i = np.random.permutation(n_items)[:n_reviews]
        for j in items_reviews_i:

            x_vec, x_i = gen_bundle(N, 1)
            u = cobb_douglas(x_vec, weights)

            X[cntr, 0] = i


            X[cntr, 1] = j
            y[cntr] = u

            cntr += 1

    return X, y


X, y = gen_dataset(N, N_USERS, N // 2)

print(X.shape)

MRS = mrs_mat(x=np.arange(N), w=weights)

df = pd.DataFrame(np.concatenate([X, y], axis=1), columns=['user_id', 'item_id', 'rating'])
df, user_item_rating_map, item_rating_map, user_id_map, id_user_map, item_id_map, id_item_map, stats = preprocess_user_item_df(df)

X = df[['user_id', 'item_id']].astype(np.int64)
y = df['rating']
X_train, X_test, y_train, y_test = split_train_test_user(X, y, random_seed=1990)


### Train Models

batch_size = 32
k = 5
h_dim = 256
n_epochs = 10
lr = 5e-5
loss_step = 50
eps = 0.05


# Train Linear Regression
enc = OneHotEncoder(sparse=False)
X_train_sparse = enc.fit_transform(X=X_train['item_id'].values.reshape(-1,1))

linear = LinearRegression(fit_intercept=True)
linear.fit(X_train_sparse, y_train)



grad_linear = linear.coef_
mrs_linear = compute_pariwise_mrs(grad_linear)
l2_linear = mrs_error(MRS, mrs_linear)
print(l2_linear)

# Train Vanilla Wide&Deep
wide_deep = WideAndDeep(stats['n_items'],  h_dim_size=246, fc1=64, fc2=32)

wide_deep.fit(X_train, y_train, batch_size, lr, n_epochs, loss_step, eps)

grad_vanilla = wide_deep.get_input_grad(np.arange(N))
mrs_vanilla = compute_pariwise_mrs(grad_vanilla.data.numpy())
l2_vanilla = mrs_error(MRS, mrs_vanilla)

print(l2_vanilla)

# Train Neural Utility Function
wide_deep_utility = WideAndDeep(stats['n_items'], h_dim_size=246, fc1=64, fc2=32)
wide_deep_utility.fit_utility_loss(X_train, y_train, batch_size, lr, n_epochs, loss_step, eps, user_item_rating_map, \
                            item_rating_map, k, stats['n_items'])

grad_utility = wide_deep_utility.get_input_grad(np.arange(N))
mrs_utility = compute_pariwise_mrs(grad_utility.data.numpy())
l2_utility = mrs_error(MRS, mrs_utility)

print(l2_utility)