import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import pandas as pd
import numpy as np
from utils.preprocessing import split_train_test_user, map_ids_to_idx
import torch

class Generator(object):

    def __init__(self, X, Y, batch_size, shuffle):

        assert Y.ndim > 1
        assert X.ndim > 1

        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = self.X.shape[0]
        self.curr_idx = 0
        self.epoch_cntr = 0
        if self.shuffle:
            self.idx = np.random.permutation(np.arange(self.n_samples))
        else:
            self.idx = np.arange(self.n_samples)


    def shuffle_idx(self):
        self.idx = np.random.permutation(np.arange(self.n_samples))

    def reset(self):
        self.curr_idx = 0
        self.epoch_cntr += 1
        if self.shuffle:
            self.shuffle_idx()

    def check(self):
        if self.curr_idx + self.batch_size >= self.n_samples:
            return True
        else:
            return False


    def update_curr_idx(self):
        self.curr_idx += self.batch_size

    def reset_epoch(self):
        self.epoch_cntr = 0


class SimpleBatchGenerator(Generator):

    def __init__(self, X, Y, batch_size, shuffle=True):
        super().__init__(X, Y, batch_size, shuffle)

    def get_batch(self):
        reset = self.check()
        if reset:
            self.reset()

        batch_idx = self.idx[self.curr_idx:(self.curr_idx + self.batch_size)]
        x_batch = self.X[batch_idx, :]
        y_batch = self.Y[batch_idx, :]
        self.update_curr_idx()

        return x_batch, y_batch



class CoocurrenceGenerator(Generator):
    """
    Class to generate minibatch samples, as well as complement and supplement sets
        - Assume that first column of X is user_id, second column is item_id
    """


    def __init__(self, X, Y, batch_size, shuffle, user_item_dict, c_size, s_size, n_item=None):
        super().__init__(X, Y, batch_size, shuffle)
        self.user_item_dict = user_item_dict
        self.c_size = c_size
        self.s_size = s_size

        if n_item is None:
            self.n_item = int(X[:, 1].max())
        else:
            self.n_item = n_item

    def get_complement_set(self, x_batch):
        X_c = np.zeros((x_batch.shape[0], self.c_size), dtype=np.int32)

        users = x_batch[:, 0]

        for i, user_id in enumerate(users):
            n, items = self.user_item_dict[user_id]
            idx = np.random.randint(0, n, self.c_size)
            c_set = items[idx]

            X_c[i, :] = c_set

        return X_c

    def get_supp_set(self, x_batch):
        X_s = np.zeros((x_batch.shape[0], self.s_size), dtype=np.int32)

        users = x_batch[:, 0]

        for i, user_id in enumerate(users):
            n, items = self.user_item_dict[user_id]
            supp_cntr = 0
            s_set = np.zeros(self.s_size)

            while supp_cntr < self.s_size:
                item = np.random.randint(0, self.n_item, 1)
                if item not in items:
                    s_set[supp_cntr] = item
                    supp_cntr +=1


            X_s[i, :] = s_set

        return X_s


    def get_batch(self, as_tensor=False):
        reset = self.check()
        if reset:
            self.reset()

        batch_idx = self.idx[self.curr_idx:(self.curr_idx + self.batch_size)]
        x_batch = self.X[batch_idx, :]
        y_batch = self.Y[batch_idx, :]

        X_c = self.get_complement_set(x_batch)
        X_s = self.get_supp_set(x_batch)

        self.update_curr_idx()

        if as_tensor:
            x_batch = torch.from_numpy(x_batch)
            y_batch = torch.from_numpy(y_batch)
            X_c = torch.from_numpy(X_c)
            X_s = torch.from_numpy(X_s)

        return x_batch, y_batch, X_c, X_s


    @classmethod
    def get_user_prod_dict(cls, df):

        user_item_dict = {}

        for user_id, user_dta in df.groupby('user_id'):

            items = user_dta['item_id'].unique()
            n = len(items)
            user_item_dict[user_id] = (n, items)

        return user_item_dict




if __name__ == "__main__":

    df = pd.read_csv(cfg.vals['movielens_dir'] + "/ratings.csv",nrows=1000)
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df.drop('timestamp', axis=1, inplace=True)

    X = df[['user_id', 'item_id']]
    y = df['rating']

    X, n_items, n_users = map_ids_to_idx(X)

    d = CoocurrenceGenerator.get_user_prod_dict(X)

    X_train, X_test, y_train, y_test = split_train_test_user(X, y)

    gen = CoocurrenceGenerator(X=X_train.values, Y=y_train.values.reshape(-1,1), batch_size=8, shuffle=True,
                               user_item_dict=d, c_size=5, s_size=5, n_item=n_items)

    while gen.epoch_cntr < 10:

        x_batch, y_batch, X_c, X_s = gen.get_batch()
        print(x_batch)

    print(gen.epoch_cntr)

