import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from model.embedding import EmbeddingGrad


def loss_mse(y_true, y_hat):
    err = y_true - y_hat
    mse = torch.mean(torch.pow(err, 2))

    return mse

def mrs_loss(y_hat, y_hat_c, y_hat_s, y_true, y_true_c, y_true_s):
    err = y_true - y_hat
    err_c = y_true_c - y_hat_c
    err_s = y_true_s - y_hat_s

    err_all = torch.cat((err.flatten(), err_c.flatten(), err_s.flatten()))

    return torch.mean(torch.pow(err_all, 2))

def get_input_layer(word_idx, n_items):
    x = torch.zeros((n_items,n_items)).float()
    x[word_idx, word_idx] = 1.0
    return x

def generate_sets(N, k):

    X_c = np.zeros((N, k, N))
    X_s = np.zeros((N, k, N))

    X_c_idx = np.zeros((N, k))
    X_s_idx = np.zeros((N, k))


    items = set(range(N))

    for i in range(N):

        items_i = items.difference(set([i]))

        c = np.random.choice(np.array(list(items_i)), size=k, replace=False)
        items_i = items_i.difference(set(c))
        s = np.random.choice(np.array(list(items_i)), size=k, replace=False)

        for j, idx in enumerate(c):
            X_c[i, j, idx] = 1.0

        for j, idx in enumerate(s):
            X_s[i, j, idx] = 1.0

        X_c_idx[i, :] = c
        X_s_idx[i, :] = s


    return X_c, X_c_idx, X_s, X_s_idx





class Encoder1(nn.Module):

    def __init__(self, n_items, h_dim_size):
        super(Encoder1, self).__init__()
        self.n_items = n_items
        self.h_dim_size = h_dim_size

        # item embedding
        #self.embedding = nn.Embedding(n_items, h_dim_size)
        #self.embedding = nn.Linear(n_items, h_dim_size)
        self.embedding = EmbeddingGrad(n_items, h_dim_size)
        self.weights = nn.Linear(h_dim_size, 1)


    def forward(self, x, x_c, x_s):
        e_i = self.embedding(x)
        e_c = self.embedding(x_c)
        e_s = self.embedding(x_s)

        y_hat = self.weights(e_i)
        y_hat_c = self.weights(e_c)
        y_hat_s = self.weights(e_s)


        return y_hat, y_hat_c, y_hat_s

    def get_input_grad(self, indices):
        """
        Get gradients with respect to inputs
        :param indices: (ndarray) array of item indices
        :return: (tensor) tensor of gradients with respect to inputs
        """
        if indices.ndim == 1:
            indices = indices.reshape(1, -1)


        dims = [d for d in indices.shape] + [1]
        idx_tensor = torch.LongTensor(torch.from_numpy(indices)).reshape(dims)

        grad = self.embedding.get_grad(indices)
        grad_at_idx = torch.gather(grad, -1, idx_tensor)
        return grad_at_idx



batch_size = 4
N = 10
k = 3
n_epochs = 200
lr = 1e-3

one_hot = OneHotEncoder(categories=[np.arange(N)], sparse=False)
X = torch.from_numpy(np.arange(N))
y = torch.from_numpy(np.random.randint(1, 5, N))

# generate complement/subsitute sets

X_c, X_c_idx, X_s, X_s_idx = generate_sets(N, k)


item_encoder = Encoder1(n_items=N, h_dim_size=8)
optimizer = optim.Adam(item_encoder.parameters(), lr=lr)

loss_arr = []

for i in range(n_epochs):

    batch_idx = np.random.choice(X, batch_size, replace=True)
    #x_batch = one_hot.fit_transform(batch_idx.reshape(-1, 1)).astype(np.float32)
    #x_batch = torch.tensor(x_batch, requires_grad=True)
    y_batch = y[batch_idx]

    #x_c_batch = torch.tensor(X_c[batch_idx, :, :].astype(np.float32), requires_grad=True)
    #x_s_batch = torch.tensor(X_s[batch_idx, :, :].astype(np.float32), requires_grad=True)
    x_c_batch = X_c_idx[batch_idx].astype(np.int64)
    x_s_batch = X_s_idx[batch_idx].astype(np.int64)

    #y_c = y[X_c_idx[batch_idx].flatten()].reshape((batch_size, k, 1))
    #y_s = y[X_s_idx[batch_idx].flatten()].reshape((batch_size, k, 1))

    y_c = y[x_c_batch.flatten()]
    y_s = y[x_s_batch.flatten()]


    #x_batch = torch.tensor(get_input_layer(batch_idx, N), requires_grad=True)
    y_hat, y_hat_c, y_hat_s = item_encoder.forward(batch_idx, x_c_batch, x_s_batch)

    loss = mrs_loss(y_hat, y_hat_c, y_hat_s, y_batch, y_c, y_s)
    loss.backward()
    optimizer.step()

    print(loss)
    loss_arr.append(loss)

    g = item_encoder.get_input_grad(batch_idx)
    print(g)
    g_c = item_encoder.get_input_grad(x_c_batch)
    print(g_c)
    g_s = item_encoder.get_input_grad(x_s_batch)
    print(g_s)

plt.plot(range(n_epochs), loss_arr)
plt.show()

