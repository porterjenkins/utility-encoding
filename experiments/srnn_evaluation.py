import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import config.config as cfg
from preprocessing.utils import split_train_test_user, load_dict_output
from preprocessing.interactions import Interactions
import numpy as np
from baselines.s_rnn import SRNN, SRNNTrainer
from experiments.utils import get_eval_metrics


params = {
    'batch_size': 32,
    'k': 5,
    'h_dim': 256,
    'n_epochs': 10,
    'lr': 5e-4,
    'loss_step': 25,
    'eps': 0.05,
    'seq_len': 4
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

sequence_users, sequences, y, n_items = interactions.to_sequence(max_sequence_length=params['seq_len'],
                                                                 min_sequence_length=params['seq_len'])

X = np.concatenate((sequence_users.reshape(-1, 1), sequences), axis=1)

srnn = SRNN(stats['n_items'], h_dim_size=256, gru_hidden_size=32, n_layers=1)
trainer = SRNNTrainer(srnn, [X, y], params, use_utility_loss=False, user_item_rating_map=user_item_rating_map,
                      item_rating_map=item_rating_map, k=5)
srnn = trainer.train()


# trainer.X_test[:, 1:], hidden=trainer.h_init)
preds, _ = srnn.predict(trainer.X_test[:, 1:])

preds = preds.detach().numpy()


# compute evaluate metrics
x_test_user = np.transpose(trainer.X_test[: , 0])
x_test_user_flat = np.zeros((len(x_test_user)*params['seq_len'], 1))


i = 0
user_cntr = 0
while i < len(x_test_user)*params['seq_len']:
    user_id = x_test_user[user_cntr]
    x_test_user_flat[i, 0] = user_id

    i += 1

    if i % params['seq_len'] == 0:
        user_cntr += 1


output = pd.DataFrame(np.concatenate([x_test_user_flat,
                                      trainer.y_test.reshape(-1,1),
                                      preds.reshape(-1,1)], axis=1), columns = ['user_id', 'y_true', 'pred'])


output, rmse, dcg = get_eval_metrics(output, at_k=5)

print(rmse)
print(dcg)