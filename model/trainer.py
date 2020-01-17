import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.optim as optim
from generator.generator import CoocurrenceGenerator, Generator
from model._loss import utility_loss, mrs_loss
from torch import nn
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class NeuralUtilityTrainer(object):

    def __init__(self, users, items, y_train, model, loss, n_epochs, batch_size, lr, loss_step_print, eps, use_cuda=False,
                 user_item_rating_map=None, item_rating_map=None, c_size=None, s_size=None, n_items=None,
                 checkpoint=False, model_path=None, model_name=None, X_val=None, y_val=None):
        self.users = users
        self.items = items
        self.y_train = y_train
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
        self.use_cuda = use_cuda
        self.n_gpu = torch.cuda.device_count()

        print(self.device)
        if self.use_cuda and self.n_gpu > 1:
            self.model = nn.DataParallel(model)  # enabling data parallelism
        else:
            self.model = model

        self.model.to(self.device)

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


    def get_generator(self, users, items, y_train, use_utility_loss):
        if use_utility_loss:
            return CoocurrenceGenerator(users, items, y_train, batch_size=self.batch_size,
                                        user_item_rating_map=self.user_item_rating_map,
                                        item_rating_map=self.item_rating_map, shuffle=True,
                                        c_size=self.c_size, s_size=self.s_size, n_item=self.n_items)
        else:
            return Generator(users, items, y_train, batch_size=self.batch_size, n_item=self.n_items, shuffle=True)

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


    def print_device_specs(self):

        if self.use_cuda:
            print("Training on GPU: {} devices".format(self.n_gpu))
        else:
            print("Training on CPU")


    def fit(self):

        self.print_device_specs()

        if self.X_val is not None:
            _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

        loss_arr = []

        iter = 0
        cum_loss = 0
        prev_loss = -1

        self.generator = self.get_generator(self.users, self.items, self.y_train, False)

        while self.generator.epoch_cntr < self.n_epochs:

            batch = self.generator.get_batch(as_tensor=True)

            batch['users'] = batch['users'].to(self.device)
            batch['items'] = batch['items'].to(self.device)
            batch['y'] = batch['y'].to(self.device)

            # zero gradient
            self.optimizer.zero_grad()

            y_hat = self.model.forward(batch['users'], batch['items']).to(self.device)
            loss = self.loss(y_true=batch['y'], y_hat=y_hat)

            if self.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            self.optimizer.step()
            loss = loss.detach()
            cum_loss += loss

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

            if self.generator.check():
                # Check if epoch is ending. Checkpoint and get evaluation metrics
                self.checkpoint_model(suffix=iter)
                if self.X_val is not None:
                    _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

            iter += 1


        return loss_arr

    def user_item_batch(self, input):
        x_user_batch = input[:, 0]
        x_batch = input[:, 1]
        return x_user_batch, x_batch

    def fit_utility_loss(self):

        self.print_device_specs()

        if self.X_val is not None:
            _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

        loss_arr = []

        iter = 0
        cum_loss = 0
        prev_loss = -1

        self.generator = self.get_generator(self.users, self.items, self.y_train, True)

        while self.generator.epoch_cntr < self.n_epochs:

            batch = self.generator.get_batch(as_tensor=True)

            batch['y'] = batch['y'].to(self.device)
            batch['y_c'] = batch['y_c'].to(self.device)
            batch['y_s'] = batch['y_s'].to(self.device)

            batch['items'] = batch['items'].requires_grad_(True).to(self.device)
            batch['x_c'] = batch['x_c'].requires_grad_(True).to(self.device)
            batch['x_s'] = batch['x_s'].requires_grad_(True).to(self.device)

            y_hat = self.model.forward(batch['users'], batch['items']).to(self.device)
            y_hat_c = self.model.forward(batch['users'], batch['x_c']).to(self.device)
            y_hat_s = self.model.forward(batch['users'], batch['x_s']).to(self.device)

            # TODO: Make this function flexible in the loss type (e.g., MSE, binary CE)
            loss_u = utility_loss(y_hat, torch.squeeze(y_hat_c), torch.squeeze(y_hat_s),
                                  batch['y'], batch['y_c'], batch['y_s'])


            if self.n_gpu > 1:
                loss_u = loss_u.mean()

            x_grad = torch.autograd.grad(loss_u, batch['items'], retain_graph=True)[0]
            x_c_grad = torch.autograd.grad(loss_u, batch['x_c'], retain_graph=True)[0]
            x_s_grad = torch.autograd.grad(loss_u, batch['x_s'], retain_graph=True)[0]

            x_grad = torch.sum(torch.mul(x_grad, batch['items']), dim=1)
            x_c_grad = torch.sum(torch.mul(x_c_grad, batch['x_c']), -1)
            x_s_grad = torch.sum(torch.mul(x_s_grad, batch['x_s']), -1)



            loss = mrs_loss(loss_u, x_grad.reshape(-1, 1), x_c_grad, x_s_grad, lmbda=0.1)

            if self.n_gpu > 1:
                loss = loss.mean()


            loss.backward()
            self.optimizer.step()
            loss = loss.detach()
            cum_loss += loss

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

            if self.generator.check():
                # Check if epoch is ending. Checkpoint and get evaluation metrics
                self.checkpoint_model(suffix=iter)
                if self.X_val is not None:
                    _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

            iter += 1

        return loss_arr


    def predict(self, users, items, y=None, batch_size=32):

        print("Getting predictions on device: {}".format(self.device))

        self.generator.update_data(users=users, items=items,
                                   y=y, shuffle=False,
                                   batch_size=batch_size)

        preds = list()

        while self.generator.epoch_cntr < 1:


            test = self.generator.get_batch(as_tensor=True)

            preds_batch = self.model.forward(test['users'], test['items'])
            preds.append(preds_batch)

        preds = torch.cat(preds, dim=0)


        return preds

