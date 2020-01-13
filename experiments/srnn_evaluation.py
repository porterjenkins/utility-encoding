import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import config.config as cfg
from preprocessing.utils import split_train_test_user, load_dict_output
from preprocessing.interactions import Interactions
import numpy as np
from model.s_rnn import SRNN, SRNNTrainer


params = {
    'batch_size': 32,
    'k': 5,
    'h_dim': 256,
    'n_epochs': 15,
    'lr': 1e-4,
    'loss_step': 10,
    'eps': 0
}

df = pd.read_csv(cfg.vals['movielens_dir'] + "/preprocessed/ratings.csv")
data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"
stats = load_dict_output(data_dir, "stats.json")
user_item_rating_map = load_dict_output(data_dir, "user_item_rating.json", True)
item_rating_map = load_dict_output(data_dir, "item_rating.json", True)

interactions = Interactions(user_ids=df['user_id'].values,
                            item_ids=df['item_id'].values,
                            ratings=df['rating'].values,
                            timestamps=df['timestamp'].values,
                            num_users=stats['n_users'],
                            num_items=stats['n_items'])

sequence_users, sequences, y, n_items = interactions.to_sequence(max_sequence_length=5, min_sequence_length=2)

X = np.concatenate((sequence_users.reshape(-1, 1), sequences), axis=1)

srnn = SRNN(stats['n_items'], h_dim_size=256, gru_hidden_size=32, n_layers=1)
trainer = SRNNTrainer(srnn, [X, y], params, use_utility_loss=False, user_item_rating_map=user_item_rating_map,
                      item_rating_map=item_rating_map, k=5)
trainer.train()
