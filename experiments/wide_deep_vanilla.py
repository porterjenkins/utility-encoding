import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import torch.optim as optim
import numpy as np
from generator.generator import CoocurrenceGenerator, SimpleBatchGenerator
from preprocessing.utils import split_train_test_user, load_dict_output
import pandas as pd
from model.wide_and_deep import WideAndDeep, WideAndDeepPretrained
from model._loss import loss_mse
from sklearn.metrics import mean_squared_error
from model.utils import load_embedding, embedding_to_df
import torch



batch_size = 32
k = 5
h_dim = 256
n_epochs = 10
lr = 1e-3
loss_step = 10
eps = 0


df = pd.read_csv(cfg.vals['movielens_dir'] + "/preprocessed/ratings.csv")

data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"
stats = load_dict_output(data_dir, "stats.json")

X = df[['user_id', 'item_id']].astype(np.int64)
y = df['rating']

X_train, X_test, y_train, y_test = split_train_test_user(X, y, random_seed=1990)



wide_deep = WideAndDeep(stats['n_items'], h_dim_size=256, fc1=64, fc2=32)

optimizer = optim.Adam(wide_deep.parameters(), lr=lr)

gen = SimpleBatchGenerator(X_train.values, y_train.values.reshape(-1 ,1), batch_size=batch_size)

loss_arr = []

iter = 0
cum_loss = 0
prev_loss = -1

while gen.epoch_cntr < n_epochs:


    x_batch, y_batch = gen.get_batch(as_tensor=True)
    # only consider items as features
    x_batch = x_batch[:, 1]

    y_hat = wide_deep.forward(x_batch)
    loss = loss_mse(y_true=y_batch, y_hat=y_hat)
    cum_loss += loss


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if iter % loss_step == 0:
        if iter == 0:
            avg_loss = cum_loss
        else:
            avg_loss = cum_loss / loss_step
        print("iteration: {} - loss: {}".format(iter, avg_loss))
        cum_loss = 0

        loss_arr.append(avg_loss)

        if abs(prev_loss - loss) < eps:
            print('early stopping criterion met. Finishing training')
            print("{} --> {}".format(prev_loss, loss))
            break
        else:
            prev_loss = loss

    iter += 1



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