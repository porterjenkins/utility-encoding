import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
from preprocessing.utils import split_train_test_user, load_dict_output
from model.trainer import NeuralUtilityTrainer
import numpy as np
from model._loss import loss_mse
from baselines.wide_and_deep import WideAndDeep
import torch
from experiments.utils import get_eval_metrics
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--loss", type = str, help="loss function to optimize", default='mse')
parser.add_argument("--cuda", type = bool, help="flag to run on gpu", default=False)
args = parser.parse_args()



params = {
            "h_dim_size": 256,
            "n_epochs": 10,
            "batch_size": 32,
            "lr": 5e-5,
            "eps": .1,
            "c_size": 5,
            "s_size": 5,
            "loss_step": 20,
            "eval_k": 5,
            "loss": args.loss
        }


print("Reading dataset")
data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"
df = pd.read_csv(data_dir + "ratings.csv")

X = df[['user_id', 'item_id']].values.astype(np.int64)
y = df['rating'].values.reshape(-1, 1).astype(np.float32)

del df

print("Dataset read complete...")

user_item_rating_map = load_dict_output(data_dir, "user_item_rating.json", True)
item_rating_map = load_dict_output(data_dir, "item_rating.json", True)
stats = load_dict_output(data_dir, "stats.json")

print("n users: {}".format(stats['n_users']))
print("n items: {}".format(stats['n_items']))

X_train, X_test, y_train, y_test = split_train_test_user(X, y)

wide_deep = WideAndDeep(stats['n_items'], h_dim_size=params["h_dim_size"], fc1=64, fc2=32,
                        use_cuda=args.cuda)


print("Model intialized")
print("Beginning Training...")

trainer = NeuralUtilityTrainer(users=X_train[:, 0].reshape(-1,1), items=X_train[:, 1:].reshape(-1,1),
                               y_train=y_train, model=wide_deep, loss=loss_mse,
                               n_epochs=params['n_epochs'], batch_size=params['batch_size'],
                               lr=params["lr"], loss_step_print=params["loss_step"],
                               eps=params["eps"], item_rating_map=item_rating_map,
                               user_item_rating_map=user_item_rating_map,
                               c_size=params["c_size"], s_size=params["s_size"],
                               n_items=stats["n_items"], use_cuda=args.cuda)


if params['loss'] == 'utility':
    print("utility loss")
    trainer.fit_utility_loss()
else:
    print("mse loss")
    trainer.fit()


def get_test_batch_size(n):

    b = 50

    while n % b > 0:
        b -= 1

    return b



users_test = X_test[:, 0].reshape(-1,1)
items_test = X_test[:, 1].reshape(-1,1)
y_test = y_test.reshape(-1,1)
test_batch_size = get_test_batch_size(users_test.shape[0])

preds = trainer.predict(users=users_test, items=items_test, y=y_test,
                        batch_size=test_batch_size).flatten().detach().numpy().reshape(-1,1)


output = pd.DataFrame(np.concatenate((users_test, preds, y_test), axis=1),
                      columns = ['user_id', 'pred', 'y_true'])

output, rmse, dcg = get_eval_metrics(output, at_k=params['eval_k'])

print("rmse: {:.4}".format(rmse))
print("dcg: {:.4}".format(dcg))
