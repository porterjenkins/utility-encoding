import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.embedding import EmbeddingGrad
from preprocessing.utils import split_train_test_user, load_dict_output
import pandas as pd
from generator.generator import CoocurrenceGenerator, SimpleBatchGenerator
from model._loss import loss_mse, utility_loss, mrs_loss
from model.encoder import UtilityEncoder





class NeuralUtilityTrainer(object):

    def __init__(self, X_train, y_train, model, loss, n_epochs, batch_size, lr, loss_step_print, eps, use_cuda=False,
                 user_item_rating_map=None, item_rating_map=None, c_size=None, s_size=None, n_items=None,
                 checkpoint=False, model_path=None, model_name=None, X_val=None, y_val=None):
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.loss = loss
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.loss_step = loss_step_print
        self.eps = eps
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.user_item_rating_map = user_item_rating_map
        self.item_rating_map = item_rating_map
        self.c_size = c_size
        self.s_size = s_size
        self.n_items = n_items
        self.checkpoint = checkpoint
        self.model_path = model_path
        self.X_val = X_val
        self.y_val = y_val

        if model_name is None:
            self.model_name = 'model'
        else:
            self.model_name = model_name

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_item_user_indices(self, batch):


        ## user ids in first column, item id's in remaining
        user_ids = batch[:, 0]
        item_ids = batch[:, 1:]

        return user_ids, item_ids


    def get_generator(self, X_train, y_train, use_utility_loss):
        if use_utility_loss:
            return CoocurrenceGenerator(X_train, y_train, batch_size=self.batch_size,
                                        user_item_rating_map=self.user_item_rating_map,
                                        item_rating_map=self.item_rating_map, shuffle=True,
                                        c_size=self.c_size, s_size=self.s_size, n_item=self.n_items)
        else:
            return SimpleBatchGenerator(X_train, y_train, batch_size=self.batch_size)


    def checkpoint_model(self, suffix):

        if self.checkpoint:

            if self.model_path is None:
                fname = "{}_{}.pt".format(self.model_name, suffix)
            else:
                fname = "{}/{}_{}.pt".format(self.model_path, self.model_name, suffix)


            with open(fname, 'wb') as f:
                torch.save(self.model, f)

    def get_validation_loss(self, X_val, y_val):

        y_hat = self.model.forward(X_val)
        val_loss = self.loss(y_true=y_val, y_hat=y_hat)
        print("---> Validation Error: {:.4f}".format(val_loss.data.numpy()))
        return val_loss


    def fit(self):

        if self.X_val is not None:
            _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

        loss_arr = []

        iter = 0
        cum_loss = 0
        prev_loss = -1

        generator = self.get_generator(self.X_train, self.y_train, False)

        while generator.epoch_cntr < self.n_epochs:

            x_batch, y_batch = generator.get_batch(as_tensor=True)
            users, items = self.get_item_user_indices(x_batch)

            # zero gradient
            self.optimizer.zero_grad()

            y_hat = self.model.forward(users, items)
            loss = self.loss(y_true=y_batch, y_hat=y_hat)
            cum_loss += loss
            loss.backward()
            self.optimizer.step()

            if iter % self.loss_step == 0:
                if iter == 0:
                    avg_loss = cum_loss
                else:
                    avg_loss = cum_loss / self.loss_step
                print("iteration: {} - loss: {:.4f}".format(iter, avg_loss))
                cum_loss = 0

                loss_arr.append(avg_loss)

                if abs(prev_loss - loss) < self.eps:
                    print('early stopping criterion met. Finishing training')
                    print("{:.4f} --> {:.4f}".format(prev_loss, loss))
                    break
                else:
                    prev_loss = loss

            if generator.check():
                # Check if epoch is ending. Checkpoint and get evaluation metrics
                self.checkpoint_model(suffix=iter)
                if self.X_val is not None:
                    _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

            iter += 1


        return loss_arr



    def fit_utility_loss(self):

        if self.X_val is not None:
            _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

        loss_arr = []

        iter = 0
        cum_loss = 0
        prev_loss = -1

        generator = self.get_generator(self.X_train, self.y_train, True)

        while generator.epoch_cntr < self.n_epochs:

            x_batch, y_batch, x_c_batch, y_c, x_s_batch, y_s = generator.get_batch(as_tensor=True)

            users, items = self.get_item_user_indices(x_batch)

            y_hat = self.model.forward(users, items)
            y_hat_c = self.model.forward(users, x_c_batch)
            y_hat_s = self.model.forward(users, x_s_batch)


            # zero gradient
            self.optimizer.zero_grad()

            # TODO: Make this function flexible in the loss type (e.g., MSE, binary CE)
            loss_u = utility_loss(y_hat, torch.squeeze(y_hat_c), torch.squeeze(y_hat_s), y_batch, y_c, y_s)
            loss_u.backward(retain_graph=True)

            x_grad = self.model.get_input_grad(items)
            x_c_grad = self.model.get_input_grad(x_c_batch)
            x_s_grad = self.model.get_input_grad(x_s_batch)

            loss = mrs_loss(loss_u, x_grad.reshape(-1, 1), x_c_grad, x_s_grad, lmbda=0.1)
            cum_loss += loss
            loss.backward()
            self.optimizer.step()

            if iter % self.loss_step == 0:
                if iter == 0:
                    avg_loss = cum_loss
                else:
                    avg_loss = cum_loss / self.loss_step
                print("iteration: {} - loss: {:.4f}".format(iter, avg_loss))
                cum_loss = 0

                loss_arr.append(avg_loss)

                if abs(prev_loss - loss) < self.eps:
                    print('early stopping criterion met. Finishing training')
                    print("{:.4f} --> {:.4f}".format(prev_loss, loss))
                    break
                else:
                    prev_loss = loss

            if generator.check():
                # Check if epoch is ending. Checkpoint and get evaluation metrics
                self.checkpoint_model(suffix=iter)
                if self.X_val is not None:
                    _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

            iter += 1

        return loss_arr

