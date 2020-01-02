import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import pandas as pd
import numpy as np
from preprocessing.utils import split_train_test_user, load_dict_output
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


    def __init__(self, X, Y, batch_size, shuffle, user_item_rating_map, item_rating_map, c_size, s_size, n_item):
        super().__init__(X, Y, batch_size, shuffle)
        self.user_item_rating_map = user_item_rating_map
        self.item_rating_map = item_rating_map
        self.c_size = c_size
        self.s_size = s_size

        if n_item is None:
            self.n_item = int(X[:, 1].max())
        else:
            self.n_item = n_item

    def get_complement_set(self, x_batch):
        X_c = np.zeros((x_batch.shape[0], self.c_size), dtype=np.int64)
        y_c = np.zeros((x_batch.shape[0], self.c_size), dtype=np.float32)

        users = x_batch[:, 0]

        for i, user_id in enumerate(users):
            item_ratings = self.user_item_rating_map[user_id]
            items = np.random.choice(list(item_ratings.keys()), size=self.c_size, replace=True)

            X_c[i, :] = items

            for j, item in enumerate(items):
                y_c[i, j] = item_ratings[item]

        return X_c, y_c

    def get_supp_set(self, x_batch):
        X_s = np.zeros((x_batch.shape[0], self.s_size), dtype=np.int64)
        y_s = np.zeros((x_batch.shape[0], self.s_size), dtype=np.float32)

        users = x_batch[:, 0]

        for i, user_id in enumerate(users):
            user_items = list(self.user_item_rating_map[user_id].keys())

            supp_cntr = 0
            s_set = np.zeros(self.s_size, dtype=np.int64)
            y_s_set = np.zeros(self.s_size, dtype=np.float32)

            while supp_cntr < self.s_size:
                item = np.random.randint(0, self.n_item, 1)[0]
                if item not in user_items:
                    s_set[supp_cntr] = item

                    n_ratings = len(self.item_rating_map[item])
                    ratings_idx = np.random.randint(0, n_ratings, 1)[0]
                    y_s_set[supp_cntr] = self.item_rating_map[item][ratings_idx]

                    supp_cntr +=1


            X_s[i, :] = s_set
            y_s[i, :] = y_s_set

        return X_s, y_s


    def get_batch(self, as_tensor=False):
        reset = self.check()
        if reset:
            self.reset()

        batch_idx = self.idx[self.curr_idx:(self.curr_idx + self.batch_size)]
        x_batch = self.X[batch_idx, :]
        y_batch = self.Y[batch_idx, :]

        X_c, y_c = self.get_complement_set(x_batch)
        X_s, y_s = self.get_supp_set(x_batch)

        self.update_curr_idx()

        if as_tensor:
            x_batch = torch.from_numpy(x_batch)
            y_batch = torch.from_numpy(y_batch)
            X_c = torch.from_numpy(X_c)
            X_s = torch.from_numpy(X_s)

        return x_batch, y_batch, X_c, y_c, X_s, y_s





if __name__ == "__main__":

    data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"

    df = pd.read_csv(data_dir + "ratings.csv")

    X = df[['user_id', 'item_id']]
    y = df['rating']

    user_item_rating_map = load_dict_output(data_dir, "user_item_rating.json", True)
    item_rating_map = load_dict_output(data_dir, "item_rating.json", True)
    stats = load_dict_output(data_dir, "stats.json")



    X_train, X_test, y_train, y_test = split_train_test_user(X, y)

    gen = CoocurrenceGenerator(X=X_train.values, Y=y_train.values.reshape(-1,1), batch_size=8, shuffle=True,
                               user_item_rating_map=user_item_rating_map, item_rating_map=item_rating_map, c_size=5,
                               s_size=5, n_item=stats['n_items'])

    while gen.epoch_cntr < 10:

        x_batch, y_batch, X_c, y_c, X_s, y_s = gen.get_batch()
        print(x_batch)

    print(gen.epoch_cntr)

