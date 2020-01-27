import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime

def get_eval_metrics(output, at_k=5):

    output.sort_values(by=['user_id', 'pred'], inplace=True, ascending=False)
    output = output.groupby('user_id').head(at_k)
    output['rank'] = output[['user_id', 'pred']].groupby('user_id').rank(method='first', ascending=False).astype(float)
    output['dcg'] = (np.power(2, output['y_true']) - 1) / np.log2(output['rank'] + 1)

    rmse = np.sqrt(mean_squared_error(output.y_true, output.pred))



    avg_dcg = output.dcg.mean()

    return output, rmse, avg_dcg


def get_eval_metrics_sequential(users_test, preds, y_test, seq_len, eval_k):

    pred_cols = ["pred_{}".format(x) for x in range(seq_len)]
    true_cols = ["y_true_{}".format(x) for x in range(seq_len)]

    output = pd.DataFrame(np.concatenate((users_test, preds, y_test), axis=1),
                          columns=['user_id'] + pred_cols + true_cols)

    pred_long = pd.melt(output[['user_id'] + pred_cols], id_vars='user_id', value_vars=pred_cols, value_name='pred')
    true_long = pd.melt(output[['user_id'] + true_cols], id_vars='user_id', value_vars=true_cols, value_name='y_true')

    output = pd.concat([pred_long[['user_id', 'pred']], true_long['y_true']], axis=1)

    output, rmse, dcg = get_eval_metrics(output, at_k=eval_k)

    return output, rmse, dcg


def get_idcg(k):
    ideal = np.zeros(k)
    ideal[0] = 1
    rank = np.arange(1, k+1)

    idcg = (np.power(2, ideal) - 1) / np.log2(rank + 1)
    return idcg


def get_choice_eval_metrics(output, at_k=5):

    output.sort_values(by=['user_id', 'pred'], inplace=True, ascending=False)
    output = output.groupby('user_id').head(at_k)


    output['rank'] = output[['user_id', 'pred']].groupby('user_id').rank(method='first', ascending=False).astype(float)
    output['dcg'] = (np.power(2, output['y_true']) - 1) / np.log2(output['rank'] + 1)

    results = output[['user_id', 'y_true', 'dcg']].groupby("user_id").sum().mean()



    ndcg = results['dcg']
    hit_ratio = results['y_true']


    return output, hit_ratio, ndcg


def get_choice_eval_sequential(output, at_k=5):

    output.sort_values(by=['user_id', 'pred'], inplace=True, ascending=False)
    output = output.groupby('user_id').head(at_k)


    output['rank'] = output[['user_id', 'pred']].groupby('user_id').rank(method='first', ascending=False).astype(float)
    output['dcg'] = (np.power(2, output['y_true']) - 1) / np.log2(output['rank'] + 1)

    results = output[['user_id', 'y_true', 'dcg']].groupby("user_id").sum().mean()



    ndcg = results['dcg']
    hit_ratio = results['y_true']


    return output, hit_ratio, ndcg


def get_choice_eval_metrics_sequential(users_test, preds, y_test, seq_len, eval_k):

    pred_cols = ["pred_{}".format(x) for x in range(seq_len)]
    true_cols = ["y_true_{}".format(x) for x in range(seq_len)]

    output = pd.DataFrame(np.concatenate((users_test, preds, y_test), axis=1),
                          columns=['user_id'] + pred_cols + true_cols)

    pred_long = pd.melt(output[['user_id'] + pred_cols], id_vars='user_id', value_vars=pred_cols, value_name='pred')
    true_long = pd.melt(output[['user_id'] + true_cols], id_vars='user_id', value_vars=true_cols, value_name='y_true')

    output = pd.concat([pred_long[['user_id', 'pred']], true_long['y_true']], axis=1)

    output, hit_ratio, ndcg = get_choice_eval_sequential(output, at_k=eval_k)

    return output, hit_ratio, ndcg


def get_test_batch_size(n):

    b = 50

    while n % b > 0:
        b -= 1

    return b

def get_test_sample_size(n, k):

    floor = n // k
    n_update = floor*k

    return n_update


def read_train_test_dir(dir, drop_ts=True):

    x_train = pd.read_csv(dir + "/x_train.csv")
    x_test = pd.read_csv(dir + "/x_test.csv")

    if drop_ts:
        x_train = x_train[['user_id', 'item_id']].values.astype(np.int64)
        x_test = x_test[['user_id', 'item_id']].values.astype(np.int64)
    else:
        x_train = x_train.values.astype(np.int64)
        x_test = x_test.values.astype(np.int64)

    y_train = pd.read_csv(dir + "/y_train.csv").values.reshape(-1,1).astype(np.float32)
    y_test = pd.read_csv(dir + "/y_test.csv").values.reshape(-1,1).astype(np.float32)



    return x_train, x_test, y_train, y_test


def log_output(out_dir, model_name, params, output):

    log_dir = out_dir + "/log"

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    now = datetime.now()
    fname = "{}/{}-{}.txt".format(log_dir, model_name, now)

    with open(fname, 'w') as f:
        f.write("{} - {}\n".format(model_name, now))
        for name, val in params.items():
            f.write("{}: {}\n".format(name, val))

        for i in output:
            f.write("{:.4f}\n".format(i))
