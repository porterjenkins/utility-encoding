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
from baselines.wide_and_deep import WideAndDeep
from sklearn.metrics import mean_squared_error


train_config = {"loss": 'utility',
                "h_dim_size": 256,
                "n_epochs": 10,
                "batch_size": 32,
                "lr": 1e-3,
                "eps": 0.01,
                "c_size": 5,
                "s_size": 5,
                "loss_step": 20
                }


data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"
df = pd.read_csv(data_dir + "ratings.csv")


X = df[['user_id', 'item_id']].values.astype(np.int64)
y = df['rating'].values.reshape(-1, 1)


user_item_rating_map = load_dict_output(data_dir, "user_item_rating.json", True)
item_rating_map = load_dict_output(data_dir, "item_rating.json", True)
stats = load_dict_output(data_dir, "stats.json")

X_train, X_test, y_train, y_test = split_train_test_user(X, y)

wide_deep = WideAndDeep(stats['n_items'], h_dim_size=train_config["h_dim_size"], fc1=64, fc2=32)

trainer = NeuralUtilityTrainer(X_train=X_train, y_train=y_train, model=wide_deep, loss=loss_mse, \
                               n_epochs=train_config['n_epochs'], batch_size=train_config['batch_size'],
                               lr=train_config["lr"], loss_step_print=train_config["loss_step"],
                               eps=train_config["eps"], item_rating_map=item_rating_map,
                               user_item_rating_map=user_item_rating_map,
                               c_size=train_config["c_size"], s_size=train_config["batch_size"],
                               n_items=stats["n_items"])


if train_config['loss'] == 'utility':
    trainer.fit_utility_loss()
else:
    trainer.fit()


preds = wide_deep.forward(X_test.values[:, 1]).flatten().detach().numpy()

# compute evaluate metrics
x_test_user = X_test['user_id'].values

output = pd.DataFrame(np.concatenate([x_test_user.reshape(-1,1), preds.reshape(-1,1), y_test.values.reshape(-1,1)], \
                                    axis=1), columns = ['user_id', 'pred', 'y_true'])

output.sort_values(by=['user_id', 'pred'], inplace=True, ascending=False)
output = output.groupby('user_id').head(5)
output['rank'] = output[['user_id', 'pred']].groupby('user_id').rank(method='first', ascending=False).astype(float)
output['dcg'] = (np.power(2, output['y_true']) - 1) / np.log2(output['rank'] + 1)

rmse = np.sqrt(mean_squared_error(output.y_true, output.pred))

print(output)

avg_dcg = output.dcg.mean()
print(rmse)
print(avg_dcg)