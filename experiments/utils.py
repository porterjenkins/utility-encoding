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


def get_test_batch_size(n):

    b = 50

    while n % b > 0:
        b -= 1

    return b

def get_test_sample_size(n, k):

    floor = n // k
    n_update = floor*k

    return n_update
