import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
from preprocessing.utils import split_train_test_user, load_dict_output
from model.trainer import NeuralUtilityTrainer
import numpy as np
from model._loss import loss_mse
from baselines.gmf import GMF
from experiments.utils import get_eval_metrics
import argparse
import pandas as pd
from experiments.utils import get_test_sample_size


parser = argparse.ArgumentParser()
parser.add_argument("--loss", type = str, help="loss function to optimize", default='mse')
parser.add_argument("--cuda", type = bool, help="flag to run on gpu", default=False)
parser.add_argument("--checkpoint", type = bool, help="flag to run on gpu", default=True)
parser.add_argument("--dataset", type = str, help = "dataset to process: {amazon, movielens}", default="Movielens")

args = parser.parse_args()

MODEL_NAME = "gmf_{}_{}".format(args.dataset, args.loss)
MODEL_DIR = cfg.vals['model_dir']
TEST_BATCH_SIZE = 100

params = {
            "h_dim_size": 256,
            "n_epochs": 10,
            "batch_size": 32,
            "lr": 5e-5,
            "eps": .001,
            "c_size": 5,
            "s_size": 5,
            "loss_step": 50,
            "eval_k": 5,
            "loss": args.loss
        }


print("Reading dataset")
if args.dataset == "movielens":
    data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"
elif args.dataset == "amazon":
    data_dir = cfg.vals['amazon_dir'] + "/preprocessed/"
else:
    raise ValueError("--dataset must be 'amazon' or 'movielens'")

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
n_test = get_test_sample_size(X_test.shape[0], k=TEST_BATCH_SIZE)
X_test = X_test[:n_test, :]
y_test = y_test[:n_test, :]


gmf = GMF(n_users=stats['n_users'], n_items=stats['n_items'], h_dim_size=params["h_dim_size"], use_cuda=args.cuda)

print("Model intialized")
print("Beginning Training...")

trainer = NeuralUtilityTrainer(users=X_train[:, 0].reshape(-1,1), items=X_train[:, 1:].reshape(-1,1),
                               y_train=y_train, model=gmf, loss=loss_mse,
                               n_epochs=params['n_epochs'], batch_size=params['batch_size'],
                               lr=params["lr"], loss_step_print=params["loss_step"],
                               eps=params["eps"], item_rating_map=item_rating_map,
                               user_item_rating_map=user_item_rating_map,
                               c_size=params["c_size"], s_size=params["s_size"],
                               n_items=stats["n_items"], use_cuda=args.cuda,
                               model_name=MODEL_NAME, model_path=MODEL_DIR,
                               checkpoint=args.checkpoint)


if params['loss'] == 'utility':
    print("utility loss")
    trainer.fit_utility_loss()
else:
    print("mse loss")
    trainer.fit()


users_test = X_test[:, 0].reshape(-1,1)
items_test = X_test[:, 1].reshape(-1,1)
y_test = y_test.reshape(-1,1)

preds = trainer.predict(users=users_test, items=items_test, y=y_test,
                        batch_size=TEST_BATCH_SIZE).reshape(-1,1)


output = pd.DataFrame(np.concatenate((users_test, preds, y_test), axis=1),
                      columns = ['user_id', 'pred', 'y_true'])

output, rmse, dcg = get_eval_metrics(output, at_k=params['eval_k'])

print("rmse: {:.4f}".format(rmse))
print("dcg: {:.4f}".format(dcg))
