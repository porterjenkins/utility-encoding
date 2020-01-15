import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.optim as optim
from generator.generator import CoocurrenceGenerator, SimpleBatchGenerator
from model._loss import utility_loss, mrs_loss


class NeuralUtilityTrainer(object):

    def __init__(self, X_train, y_train, model, loss, n_epochs, batch_size, lr, loss_step_print, eps, use_cuda=False,
                 user_item_rating_map=None, item_rating_map=None, c_size=None, s_size=None, n_items=None):
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

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_generator(self, X_train, y_train, use_utility_loss):
        if use_utility_loss:
            return CoocurrenceGenerator(X_train, y_train, batch_size=self.batch_size,
                                        user_item_rating_map=self.user_item_rating_map,
                                        item_rating_map=self.item_rating_map, shuffle=True,
                                        c_size=self.c_size, s_size=self.s_size, n_item=self.n_items)
        else:
            return SimpleBatchGenerator(X_train, y_train, batch_size=self.batch_size)

    def fit(self):

        loss_arr = []

        iter = 0
        cum_loss = 0
        prev_loss = -1

        generator = self.get_generator(self.X_train, self.y_train, False)

        while generator.epoch_cntr < self.n_epochs:

            x_batch, y_batch = generator.get_batch(as_tensor=True)
            # only consider items as features
            x_batch = x_batch[:, 1]

            # zero gradient
            self.optimizer.zero_grad()

            y_hat = self.model.forward(x_batch)
            loss = self.loss(y_true=y_batch, y_hat=y_hat)
            cum_loss += loss
            loss.backward()
            self.optimizer.step()

            if iter % self.loss_step == 0:
                if iter == 0:
                    avg_loss = cum_loss
                else:
                    avg_loss = cum_loss / self.loss_step
                print("iteration: {} - loss: {}".format(iter, avg_loss))
                cum_loss = 0

                loss_arr.append(avg_loss)

                if abs(prev_loss - loss) < self.eps:
                    print('early stopping criterion met. Finishing training')
                    print("{} --> {}".format(prev_loss, loss))
                    break
                else:
                    prev_loss = loss

            iter += 1

        return loss_arr

    def user_item_batch(self, input):
        x_user_batch = input[:, 0]
        x_batch = input[:, 1]
        return x_user_batch, x_batch

    def fit_utility_loss(self):

        loss_arr = []

        iter = 0
        cum_loss = 0
        prev_loss = -1

        generator = self.get_generator(self.X_train, self.y_train, True)

        while generator.epoch_cntr < self.n_epochs:

            x_batch, y_batch, x_c_batch, y_c, x_s_batch, y_s = generator.get_batch(as_tensor=True)

            # only consider items as features
            user_indices, x_batch = self.user_item_batch(x_batch)

            y_hat = self.model.forward(user_indices, x_batch)

            for i in range(x_c_batch.shape[1]):
                # zero gradient
                y_hat_c = self.model.forward(user_indices, x_c_batch[:, i])
                y_hat_s = self.model.forward(user_indices, x_s_batch[:, i])

                # TODO: Make this function flexible in the loss type (e.g., MSE, binary CE)
                loss_u = utility_loss(y_hat, torch.squeeze(y_hat_c), torch.squeeze(y_hat_s), y_batch, y_c[:, i],
                                      y_s[:, i])
                loss_u.backward(retain_graph=True)
                self.optimizer.zero_grad()

            x_grad = self.model.get_input_grad(x_batch)
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
                print("iteration: {} - loss: {}".format(iter, avg_loss))
                cum_loss = 0

                loss_arr.append(avg_loss)

                if abs(prev_loss - loss) < self.eps:
                    print('early stopping criterion met. Finishing training')
                    print("{} --> {}".format(prev_loss, loss))
                    break
                else:
                    prev_loss = loss

            iter += 1

        return loss_arr
