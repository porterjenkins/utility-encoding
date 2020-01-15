import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import pandas as pd
from preprocessing.utils import split_train_test_user, load_dict_output
from model.trainer import NeuralUtilityTrainer
import numpy as np
from model._loss import loss_mse
from baselines.gmf import GMF




data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"

df = pd.read_csv(data_dir + "ratings.csv")

X = df[['user_id', 'item_id']].values.astype(np.int64)
y = df['rating'].values.reshape(-1, 1)

user_item_rating_map = load_dict_output(data_dir, "user_item_rating.json", True)
item_rating_map = load_dict_output(data_dir, "item_rating.json", True)
stats = load_dict_output(data_dir, "stats.json")

X_train, X_test, y_train, y_test = split_train_test_user(X, y)


c = {'num_users': stats['n_users'],
     'num_items': stats['n_items'],
     'latent_dim': 64}

model = GMF(config=c)

trainer = NeuralUtilityTrainer(X_train=X_train, y_train=y_train, model=model, loss=loss_mse, \
                               n_epochs=5, batch_size=32, lr=1e-3, loss_step_print=1, eps=.01,
                               item_rating_map=item_rating_map, user_item_rating_map=user_item_rating_map,
                               c_size=5, s_size=5, n_items=stats["n_items"])

#trainer.fit()
trainer.fit_utility_loss()