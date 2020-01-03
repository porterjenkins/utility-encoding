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
from generator.generator import CoocurrenceGenerator, SimpleBatchGenerator
from preprocessing.utils import split_train_test_user, load_dict_output
import pandas as pd
from model.utils import write_embeddings


def loss_mse(y_true, y_hat):
    err = y_true - y_hat
    mse = torch.mean(torch.pow(err, 2))

    return mse

class WideAndDeep(nn.Module):

    def __init__(self, n_items, h_dim_size, fc1=64, fc2=32):
        super(WideAndDeep, self).__init__()
        self.n_items = n_items
        self.h_dim_size = h_dim_size


        self.embedding = EmbeddingGrad(n_items, h_dim_size)
        self.fc_1 = nn.Linear(h_dim_size, fc1)
        self.fc_2 = nn.Linear(fc1, fc2)



        self.output_layer = nn.Linear(n_items + fc2, 1)

    def _forward_set(self, x):
        h = self.embedding(x)
        h = self.fc_1(h)
        h = self.fc_2(h)

        wide = self.embedding._collect(x)
        h = torch.cat([h, wide], dim=1)

        y_hat = self.output_layer(h)
        return y_hat


    def forward(self, x):

        y_hat = self._forward_set(x)
        return y_hat




if __name__ == "__main__":
    batch_size = 32
    k = 5
    h_dim = 256
    n_epochs = 1
    lr = 1e-5
    loss_step = 50
    eps = 0


    df = pd.read_csv(cfg.vals['movielens_dir'] + "/preprocessed/ratings.csv")

    data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"
    stats = load_dict_output(data_dir, "stats.json")

    X = df[['user_id', 'item_id']]
    y = df['rating']

    X_train, X_test, y_train, y_test = split_train_test_user(X, y, random_seed=1990)



    wide_deep = WideAndDeep(stats['n_items'], h_dim_size=246, fc1=64, fc2=32)
    optimizer = optim.Adam(wide_deep.parameters(), lr=lr)

    gen = SimpleBatchGenerator(X_train.values, y_train.values.reshape(-1,1), batch_size=batch_size)

    loss_arr = []

    iter = 0
    cum_loss = 0
    prev_loss = -1

    while gen.epoch_cntr < n_epochs:


        x_batch, y_batch = gen.get_batch(as_tensor=True)
        # only consider items as features
        x_batch = x_batch[:, 1]

        y_hat = wide_deep.forward(x_batch)
        loss = loss_mse(y_true=y_batch, y_hat=y_hat)
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



    y_hat = wide_deep.forward(X_test.values[:, 1])
    rmse = np.sqrt(np.mean(np.power(y_test.values - y_hat.flatten().detach().numpy(), 2)))
    print(rmse)


