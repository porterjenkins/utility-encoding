import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import pandas as pd
from preprocessing.utils import split_train_test_user, load_dict_output
from model.trainer import NeuralUtilityTrainer
import numpy as np
from model._loss import loss_mse
import torch
from baselines.ncf_mlp import MLP
import pandas as pd
import argparse
from experiments.utils import get_eval_metrics
from model.neural_utility_function import NeuralUtility

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type = bool, help="flag to run on gpu", default=False)
args = parser.parse_args()

print("Use CUDA: {}".format(args.cuda))



params = {"loss": 'mse',
                "h_dim_size": 256,
                "n_epochs": 1,
                "batch_size": 32,
                "lr": 1e-3,
                "eps": 0.01,
                "c_size": 5,
                "s_size": 5,
                "loss_step": 1,
                "eval_k": 5
                }


data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"

df = pd.read_csv(data_dir + "ratings.csv")

X = df[['user_id', 'item_id']].values.astype(np.int64)
y = df['rating'].values.reshape(-1, 1)

user_item_rating_map = load_dict_output(data_dir, "user_item_rating.json", True)
item_rating_map = load_dict_output(data_dir, "item_rating.json", True)
stats = load_dict_output(data_dir, "stats.json")

X_train, X_test, y_train, y_test = split_train_test_user(X, y)

c = {'num_users': stats["n_users"],
     'num_items': stats['n_items'],
     'latent_dim': params['h_dim_size'],
     'layers': [params['h_dim_size']*2, 64, 32]}

model = NeuralUtility(backbone=MLP(c),
                      n_items=stats["n_items"], h_dim_size=params["h_dim_size"],
                      use_embedding=False)

trainer = NeuralUtilityTrainer(X_train=X_train, y_train=y_train, model=model, loss=loss_mse, \
                               n_epochs=params['n_epochs'], batch_size=params["batch_size"], lr=params["lr"],
                               loss_step_print=params["loss_step"], eps=params["eps"],
                               item_rating_map=item_rating_map, user_item_rating_map=user_item_rating_map,
                               c_size=params["c_size"], s_size=params["s_size"], n_items=stats["n_items"],
                               use_cuda=args.cuda)

if params['loss'] == 'utility':
    print("utility loss")
    trainer.fit_utility_loss()
else:
    print("mse loss")
    trainer.fit()


X_test = torch.from_numpy(X_test)
test_users, test_items = trainer.get_item_user_indices(X_test)
preds = model.predict(test_users, test_items).detach().numpy()

# compute evaluate metrics
x_test_user = X_test[:, 0].data.numpy()

output = pd.DataFrame(np.concatenate([x_test_user.reshape(-1,1), preds.reshape(-1,1), y_test.reshape(-1,1)], \
                                    axis=1), columns = ['user_id', 'pred', 'y_true'])

output, rmse, dcg = get_eval_metrics(output, at_k=params['eval_k'])

print("rmse: {:.4}".format(rmse))
print("dcg: {:.4}".format(dcg))

