import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import json
import os
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def get_one_hot_encodings(X, sparse=False):
    encoder = OneHotEncoder(sparse=sparse)
    if X.values.ndim == 1:
        one_hot = encoder.fit_transform(X.values.reshape(-1, 1))
    else:
        one_hot = encoder.fit_transform(X)
    cols = ['oh_{}'.format(x) for x in range(one_hot.shape[1])]
    one_hot = pd.DataFrame(one_hot, columns=cols)

    return pd.concat([X, one_hot], axis=1)


def split_train_test_user(X, y, test_size=.2, random_seed=None):
    assert 'user_id' in list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=X['user_id'],
                                                        random_state=random_seed)

    return X_train, X_test, y_train, y_test


def preprocess_user_item_df(df):

    n_rows = df.shape[0]

    user_item_rating_map = {}
    item_rating_map = {}

    user_id_map = {}
    item_id_map = {}

    id_user_map = {}
    id_item_map = {}

    user_cntr = 0
    item_cntr = 0

    row_cntr = 0
    for idx, row in df.iterrows():

        # update values of dataframe
        if row.user_id not in user_id_map:
            user_id_map[row.user_id] = user_cntr
            id_user_map[user_cntr] = row.user_id

            user_cntr += 1

        if row.item_id not in item_id_map:
            item_id_map[row.item_id] = item_cntr
            id_item_map[item_cntr] = row.item_id

            item_cntr += 1

        df.loc[idx, 'user_id'] = user_id_map[row.user_id]
        df.loc[idx, 'item_id'] = item_id_map[row.item_id]


        # update item_rating dicts

        if item_id_map[row.item_id] not in item_rating_map:
            item_rating_map[item_id_map[row.item_id]] = []

        item_rating_map[item_id_map[row.item_id]].append(row.rating)

        # update user_item_rating dict

        if user_id_map[row.user_id] not in user_item_rating_map:
            user_item_rating_map[user_id_map[row.user_id]] = {}



        user_item_rating_map[user_id_map[row.user_id]][item_id_map[row.item_id]] = row.rating

        row_cntr += 1

        print("progress: {:.2f}%".format((row_cntr / n_rows)*100), end='\r')

    stats = {
                'n_users': user_cntr,
                'n_items': item_cntr
             }



    return df, user_item_rating_map, item_rating_map, user_id_map, id_user_map, item_id_map, id_item_map, stats



def write_dict_output(out_dir, fname, data):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_path = out_dir + fname
    with open(out_path, 'w') as fp:
        json.dump(data, fp)

def load_dict_output(dir, fname, keys_to_int=False):

    fpath = dir + fname
    with open(fpath, 'r') as fp:
        data =json.load(fp)

    if keys_to_int:
        data_to_int = {}
        for k, v in data.items():

            if isinstance(v, dict):

                data_to_int[int(k)] = {int(k2): v2 for k2,v2 in v.items()}

            else:
                data_to_int[int(k)] = v

        return data_to_int


    else:
        return data
