import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def get_eval_metrics(output, at_k=5):

    output.sort_values(by=['user_id', 'pred'], inplace=True, ascending=False)
    output = output.groupby('user_id').head(at_k)
    output['rank'] = output[['user_id', 'pred']].groupby('user_id').rank(method='first', ascending=False).astype(float)
    output['dcg'] = (np.power(2, output['y_true']) - 1) / np.log2(output['rank'] + 1)

    rmse = np.sqrt(mean_squared_error(output.y_true, output.pred))



    avg_dcg = output.dcg.mean()

    return output, rmse, avg_dcg


def get_choice_eval_metrics(output, at_k=5):

    output.sort_values(by=['user_id', 'pred'], inplace=True, ascending=False)
    output = output.groupby('user_id').head(at_k)


    output['rank'] = output[['user_id', 'pred']].groupby('user_id').rank(method='first', ascending=False).astype(float)
    output['dcg'] = (np.power(2, output['y_true']) - 1) / np.log2(output['rank'] + 1)

    hit_ratio = output[['user_id', 'y_true']].groupby("user_id").sum().mean()



    cum_dcg = output.dcg.sum()

    return output, hit_ratio, cum_dcg


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
