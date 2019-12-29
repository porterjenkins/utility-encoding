import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def loss_mse(y_true, y_hat):
    err = y_true - y_hat
    mse = torch.mean(torch.pow(err, 2))

    return mse

def get_input_layer(word_idx, n_items):
    x = torch.zeros((n_items,n_items)).float()
    x[word_idx, word_idx] = 1.0
    return x

def generate_sets(N, k):

    X_c = np.zeros((N, k))
    X_s = np.zeros((N, k))

    items = set(range(N))

    for i in range(N):

        items_i = items.difference(set([i]))

        c = np.random.choice(np.array(list(items_i)), size=k, replace=False)
        items_i = items_i.difference(set(c))
        s = np.random.choice(np.array(list(items_i)), size=k, replace=False)

        X_c[i, :] = np.array(c)
        X_s[k, :] = np.array(s)


    return X_c, X_s





class Encoder1(nn.Module):

    def __init__(self, n_items, h_dim_size):
        super(Encoder1, self).__init__()
        self.n_items = n_items
        self.h_dim_size = h_dim_size

        # item embedding
        #self.embedding = nn.Embedding(n_items, h_dim_size)
        self.embedding = nn.Linear(n_items, h_dim_size)
        self.weights = nn.Linear(h_dim_size, 1)


    def forward(self, x):
        e_i = self.embedding(x)

        y_hat = self.weights(e_i)


        return y_hat



batch_size = 4
N = 20
k = 5
n_epochs = 250
lr = 1e-3

one_hot = OneHotEncoder(categories=[np.arange(N)], sparse=False)
X = torch.from_numpy(np.arange(N))
y = torch.from_numpy(np.random.randint(1, 5, N))

# generate complement/subsitute sets

X_c, X_s = generate_sets(N, k)


item_encoder = Encoder1(n_items=N, h_dim_size=8)
optimizer = optim.Adam(item_encoder.parameters(), lr=lr)

loss_arr = []

for i in range(n_epochs):

    batch_idx = np.random.choice(X, batch_size, replace=True)
    x_batch = one_hot.fit_transform(batch_idx.reshape(-1, 1)).astype(np.float32)
    x_batch = torch.tensor(x_batch, requires_grad=True)
    y_batch = y[batch_idx]

    #x_batch = torch.tensor(get_input_layer(batch_idx, N), requires_grad=True)
    y_hat = item_encoder.forward(x_batch)

    loss = loss_mse(y_batch, y_hat)
    loss.backward()
    optimizer.step()

    print(loss)
    loss_arr.append(loss)


plt.plot(range(n_epochs), loss_arr)
plt.show()

