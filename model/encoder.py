import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model.embedding import EmbeddingGrad
from generator.generator import CoocurrenceGenerator
from preprocessing.utils import split_train_test_user, load_dict_output
import pandas as pd
from model.utils import write_embeddings


def loss_mse(y_true, y_hat):
    err = y_true - y_hat
    mse = torch.mean(torch.pow(err, 2))

    return mse

def utility_loss(y_hat, y_hat_c, y_hat_s, y_true, y_true_c, y_true_s):
    err = y_true - y_hat
    err_c = y_true_c - y_hat_c
    err_s = y_true_s - y_hat_s

    err_all = torch.cat((err.flatten(), err_c.flatten(), err_s.flatten()))
    return torch.mean(torch.pow(err_all, 2))

def mrs_loss(utility_loss, x_grad, x_c_grad, x_s_grad, lmbda=1):
    
    mrs_c = -(x_grad / x_c_grad)
    mrs_s = -(x_grad/ x_s_grad)

    c_norm = torch.norm(mrs_c, dim=1)
    s_norm = torch.log(torch.norm(mrs_s, dim=1))
    
    mrs_loss =  torch.mean(c_norm - s_norm)

    loss = mrs_loss + lmbda*utility_loss

    return loss


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





class UtilityEncoder(nn.Module):

    def __init__(self, n_items, h_dim_size):
        super(UtilityEncoder, self).__init__()
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


        return y_hat, torch.squeeze(y_hat_c), torch.squeeze(y_hat_s)

    def get_embedding_mtx(self):

        return np.transpose(item_encoder.embedding.weights.weight.data.numpy())

    def get_input_grad(self, indices):
        """
        Get gradients with respect to inputs
        :param indices: (ndarray) array of item indices
        :return: (tensor) tensor of gradients with respect to inputs
        """
        if indices.ndim == 1:
            indices = indices.reshape(-1, 1)


        dims = [d for d in indices.shape] + [1]
        idx_tensor = torch.LongTensor(indices).reshape(dims)

        grad = self.embedding.get_grad(indices)
        grad_at_idx = torch.gather(grad, -1, idx_tensor)
        return torch.squeeze(grad_at_idx)


if __name__ == "__main__":


    batch_size = 32
    k = 5
    d = 32
    n_epochs = 10
    lr = 1e-3
    loss_step = 20
    eps = .01

    data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"

    df = pd.read_csv(data_dir + "ratings.csv")

    X = df[['user_id', 'item_id']]
    y = df['rating']

    user_item_rating_map = load_dict_output(data_dir, "user_item_rating.json", True)
    item_rating_map = load_dict_output(data_dir, "item_rating.json", True)
    stats = load_dict_output(data_dir, "stats.json")

    X_train, X_test, y_train, y_test = split_train_test_user(X, y)

    gen = CoocurrenceGenerator(X=X_train.values, Y=y_train.values.reshape(-1, 1), batch_size=8, shuffle=True,
                               user_item_rating_map=user_item_rating_map, item_rating_map=item_rating_map, c_size=5,
                               s_size=5, n_item=stats['n_items'])



    item_encoder = UtilityEncoder(n_items=gen.n_item, h_dim_size=d)
    optimizer = optim.Adam(item_encoder.parameters(), lr=lr)

    loss_arr = []

    iter = 0
    cum_loss = 0
    prev_loss = -1

    while gen.epoch_cntr < n_epochs:


        x_batch, y_batch, x_c_batch, y_c, x_s_batch, y_s = gen.get_batch(as_tensor=True)

        # only consider items as features
        x_batch = x_batch[:, 1]

        y_hat, y_hat_c, y_hat_s = item_encoder.forward(x_batch, x_c_batch, x_s_batch)

        loss_u = utility_loss(y_hat, y_hat_c, y_hat_s, y_batch, y_c, y_s)
        loss_u.backward(retain_graph=True)


        x_grad = item_encoder.get_input_grad(x_batch)
        x_c_grad = item_encoder.get_input_grad(x_c_batch)
        x_s_grad = item_encoder.get_input_grad(x_s_batch)

        loss = mrs_loss(loss_u, x_grad.reshape(-1,1), x_c_grad, x_s_grad)
        cum_loss += loss
        loss.backward()
        optimizer.step()

        if iter % loss_step == 0:
            if iter == 0:
                avg_loss = cum_loss
            else:
                avg_loss = cum_loss / loss_step
            print("iteration: {} - loss: {}".format(iter, avg_loss))
            cum_loss = 0

            loss_arr.append(avg_loss)

            if abs(prev_loss - loss) < eps:
                print('early stopping criterion met. Finishing training')
                print("{} --> {}".format(prev_loss, loss))
                break
            else:
                prev_loss = loss


        iter += 1

    write_embeddings(item_encoder.get_embedding_mtx(), stats['n_items'], fname=cfg.vals['model_dir'] + '/embedding.txt')

    plt.plot(np.arange(len(loss_arr)), np.log(loss_arr))
    plt.savefig("learning_curve.pdf")


