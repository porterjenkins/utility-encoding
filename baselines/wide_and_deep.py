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
from model._loss import loss_mse, utility_loss, mrs_loss




class WideAndDeepPretrained(nn.Module):

    def __init__(self, n_items, h_dim_size, wide, wide_dim, fc1=64, fc2=32,):
        super(WideAndDeepPretrained, self).__init__()
        self.n_items = n_items
        self.h_dim_size = h_dim_size


        self.embedding = EmbeddingGrad(n_items, h_dim_size)
        self.fc_1 = nn.Linear(h_dim_size, fc1)
        self.fc_2 = nn.Linear(fc1, fc2)

        self.wide = EmbeddingGrad(n_items, wide_dim, init_embed=wide)



        self.output_layer = nn.Linear(wide_dim + fc2, 1)

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


    def _forward_set(self, x):
        h = self.embedding(x)
        h = F.relu(self.fc_1(h))
        h = F.relu(self.fc_2(h))

        wide = self.wide(x)
        h = torch.cat([h, wide], dim=-1)

        y_hat = self.output_layer(h)
        return y_hat


    def forward(self, x, x_c=None, x_s=None):

        y_hat = self._forward_set(x)

        if x_c is not None and x_s is not None:

            y_hat_c = self._forward_set(x_c)
            y_hat_s = self._forward_set(x_s)

            return y_hat, torch.squeeze(y_hat_c), torch.squeeze(y_hat_s)

        else:

            return y_hat


    def fit(self, X_train, y_train, batch_size, k, lr, n_epochs, loss_step, eps):
        pass


    def predict(self, X_test):
        pass

class WideAndDeep(nn.Module):

    def __init__(self, n_items, h_dim_size, fc1=64, fc2=32, use_cuda=False, use_embedding=True):
        super(WideAndDeep, self).__init__()
        self.n_items = n_items
        self.h_dim_size = h_dim_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.use_embedding = use_embedding


        #self.embedding = nn.Embedding(n_items, h_dim_size).requires_grad_(True)
        self.embedding = EmbeddingGrad(n_items, h_dim_size, use_cuda=use_cuda)
        self.fc_1 = nn.Linear(h_dim_size, fc1)
        self.fc_2 = nn.Linear(fc1, fc2)

        self.output_layer = nn.Linear(fc2, 1)
        #self.output_layer = nn.Linear(n_items + fc2, 1)

        if use_cuda:
            self = self.cuda()



    def forward(self, users, items):

        h = self.embedding(items)
        h = F.relu(self.fc_1(h))
        h = F.relu(self.fc_2(h))

        #wide = self.embedding._collect(items)
        #h = torch.cat([h, items], dim=-1)

        y_hat = self.output_layer(h)

        return y_hat

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

        if self.use_embedding:
            grad = self.embedding.get_grad(indices)
        else:
            grad = self.backbone.embedding.get_grad(indices)

        grad_at_idx = torch.gather(grad, -1, idx_tensor)
        return torch.squeeze(grad_at_idx)


    def fit(self, X_train, y_train, batch_size, lr, n_epochs, loss_step, eps):

        optimizer = optim.Adam(self.parameters(), lr=lr)

        gen = SimpleBatchGenerator(X_train.values, y_train.values.reshape(-1, 1), batch_size=batch_size)

        loss_arr = []

        iter = 0
        cum_loss = 0
        prev_loss = -1

        while gen.epoch_cntr < n_epochs:

            x_batch, y_batch = gen.get_batch(as_tensor=True)
            # only consider items as features
            x_batch = x_batch[:, 1]

            y_hat = self.forward(x_batch)
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


    def fit_utility_loss(self, X_train, y_train, batch_size, lr, n_epochs, loss_step, eps, user_item_rating_map, item_rating_map, k, n_items):

        gen = CoocurrenceGenerator(X=X_train.values, Y=y_train.values.reshape(-1, 1), batch_size=batch_size, shuffle=True,
                                   user_item_rating_map=user_item_rating_map, item_rating_map=item_rating_map, c_size=5,
                                   s_size=k, n_item=n_items)



        optimizer = optim.Adam(self.parameters(), lr=lr)

        loss_arr = []

        iter = 0
        cum_loss = 0
        prev_loss = -1

        while gen.epoch_cntr < n_epochs:

            x_batch, y_batch, x_c_batch, y_c, x_s_batch, y_s = gen.get_batch(as_tensor=True)

            # only consider items as features
            x_batch = x_batch[:, 1]

            y_hat, y_hat_c, y_hat_s = self.forward(x_batch, x_c_batch, x_s_batch)

            loss_u = utility_loss(y_hat, y_hat_c, y_hat_s, y_batch, y_c, y_s)
            loss_u.backward(retain_graph=True)

            x_grad = self.get_input_grad(x_batch)
            x_c_grad = self.get_input_grad(x_c_batch)
            x_s_grad = self.get_input_grad(x_s_batch)

            loss = mrs_loss(loss_u, x_grad.reshape(-1, 1), x_c_grad, x_s_grad, lmbda=0.1)
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

    def predict(self, X_test):
        y_hat = self.forward(X_test)
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
    wide_deep.fit(X_train, y_train, batch_size, lr, n_epochs, loss_step, eps)
    y_hat = wide_deep.predict(X_test.values[:, 1])
    rmse = np.sqrt(np.mean(np.power(y_test.values - y_hat.flatten().detach().numpy(), 2)))
    print(rmse)

    """optimizer = optim.Adam(wide_deep.parameters(), lr=lr)

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
    print(rmse)"""


