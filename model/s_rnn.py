import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import config.config as cfg
from generator.generator import SimpleBatchGenerator
from model._loss import loss_mse
from preprocessing.utils import split_train_test_user, load_dict_output
from preprocessing.interactions import Interactions
import numpy as np
from model.embedding import EmbeddingGrad


class SRNNTrainer(object):
    def __init__(self, srnn, data, params, use_cuda=False):
        self.srnn = srnn
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.data = data

        self.batch_size = params['batch_size']

        self.h_dim = params['h_dim']
        self.n_epochs = params['n_epochs']
        self.lr = params['lr']
        self.loss_step = params['loss_step']
        self.eps = params['eps']

        self.optimizer = optim.Adam(srnn.parameters(), lr=self.lr)

    def train(self):
        X = self.data[0]
        y = self.data[1]
        X_train, X_test, y_train, y_test = split_train_test_user(X, y, random_seed=1990)
        gen = self.generator(X_train, y_train)

        loss_arr = []
        iter = 0
        cum_loss = 0
        prev_loss = -1

        while gen.epoch_cntr < self.n_epochs:
            # print('Start Epoch #', gen.epoch_cntr)

            train_loss = self.do_epoch(gen)
            cum_loss += train_loss
            train_loss.backward()
            self.optimizer.step()

            if iter % self.loss_step == 0:
                if iter == 0:
                    avg_loss = cum_loss
                else:
                    avg_loss = cum_loss / self.loss_step
                print("iteration: {} - loss: {}".format(iter, avg_loss))
                cum_loss = 0

                loss_arr.append(avg_loss)

                if abs(prev_loss - train_loss) < self.eps:
                    print('early stopping criterion met. Finishing training')
                    print("{} --> {}".format(prev_loss, train_loss))
                    break
                else:
                    prev_loss = train_loss

            iter += 1

        # h = self.srnn.init_hidden()
        # y_hat, h = srnn.forward(X_test.values[:, 1], h)
        # rmse = np.sqrt(np.mean(np.power(y_test.values - y_hat.flatten().detach().numpy(), 2)))
        # print(rmse)

    def generator(self, X_train, y_train):
        return SimpleBatchGenerator(X_train, y_train, batch_size=self.batch_size)

    def do_epoch(self, gen):

        h = self.srnn.init_hidden()
        x_batch, y_batch = gen.get_batch(as_tensor=True)
        # only consider items as features
        x_batch = x_batch[:, 1:]
        self.optimizer.zero_grad()
        x_batch = x_batch.to(self.device)
        h = h.to(self.device)
        y_hat, h = self.srnn.forward(x_batch, h)
        transpose = np.transpose(y_batch).to(self.device)
        loss = loss_mse(y_true=transpose, y_hat=y_hat)
        return loss


class SRNN(nn.Module):

    def __init__(self, n_items, h_dim_size, gru_hidden_size, n_layers=3, use_cuda=False, batch_size=32):
        super(SRNN, self).__init__()
        self.batch_size = batch_size
        self.n_items = n_items
        self.h_dim_size = h_dim_size
        self.gru_hidden_size = gru_hidden_size
        self.n_layers = n_layers
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.use_cuda = use_cuda
        self.gru = nn.GRU(input_size=self.h_dim_size, hidden_size=self.h_dim_size, num_layers=self.n_layers)
        self.activation = nn.Tanh()
        self.out = nn.Linear(h_dim_size, 1)
        self.embedding = EmbeddingGrad(n_items, h_dim_size, use_cuda=use_cuda)
        if use_cuda:
            self = self.cuda()

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = embedded.unsqueeze(0)
        o, h = self.gru(torch.transpose(torch.squeeze(embedded), 0, 1), hidden)
        # o = o.view(-1, o.size(-1))

        y_hat = torch.squeeze(self.activation(self.out(o)))
        if self.use_cuda:
            y_hat = y_hat.to(self.device)
            h = h.to(self.device)
        return y_hat, h

    def init_hidden(self):
        return torch.zeros(self.n_layers, self.batch_size, self.h_dim_size).to(self.device)

    def one_hot(self, input):
        self.one_hot_embedding.zero_()
        index = input.view(-1, 1)
        one_hot = self.one_hot_embedding.scatter_(1, index, 1)
        return one_hot

    def init_onehot_embedding(self):
        onehot = torch.FloatTensor(self.batch_size, self.n_items)
        onehot = onehot.to(self.device)
        return onehot


if __name__ == "__main__":
    params = {
        'batch_size': 32,
        'k': 5,
        'h_dim': 256,
        'n_epochs': 100,
        'lr': 1e-3,
        'loss_step': 10,
        'eps': 0
    }

    df = pd.read_csv(cfg.vals['movielens_dir'] + "/preprocessed/ratings.csv")
    data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"
    stats = load_dict_output(data_dir, "stats.json")

    interactions = Interactions(user_ids=df['user_id'].values,
                                item_ids=df['item_id'].values,
                                ratings=df['rating'].values,
                                timestamps=df['timestamp'].values,
                                num_users=stats['n_users'],
                                num_items=stats['n_items'])

    sequence_users, sequences, y, n_items = interactions.to_sequence(max_sequence_length=5, min_sequence_length=2)

    X = np.concatenate((sequence_users.reshape(-1, 1), sequences), axis=1)

    srnn = SRNN(stats['n_items'], h_dim_size=256, gru_hidden_size=32, n_layers=1, use_cuda=True)
    trainer = SRNNTrainer(srnn, [X, y], params, use_cuda=True)
    trainer.train()
