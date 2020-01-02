import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import json
import os

def get_one_hot_encodings(X):
    item_encodings = pd.get_dummies(X['item_id'])
    user_encodings = pd.get_dummies(X['user_id'])

    return pd.concat([X, user_encodings, item_encodings], axis=1)



def split_train_test_user(X, y, test_size=.2):
    assert 'user_id' in list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=X['user_id'])

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



    return user_item_rating_map, item_rating_map, user_id_map, id_user_map, item_id_map, id_item_map, stats



def write_dict_output(out_dir, fname, data):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_path = out_dir + fname
    with open(out_path, 'w') as fp:
        json.dump(data, fp)
