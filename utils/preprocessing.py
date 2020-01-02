import pandas as pd
from sklearn.model_selection import train_test_split

def get_one_hot_encodings(X):
    item_encodings = pd.get_dummies(X['item_id'])
    user_encodings = pd.get_dummies(X['user_id'])

    return pd.concat([X, user_encodings, item_encodings], axis=1)



def split_train_test_user(X, y, test_size=.2):
    assert 'user_id' in list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=X['user_id'])

    return X_train, X_test, y_train, y_test


def map_ids_to_idx(df):
    user_map = {}
    item_map = {}

    user_cntr = 0
    item_cntr = 0



    for idx, row in df.iterrows():

        if row.user_id not in user_map:
            user_map[row.user_id] = user_cntr
            user_cntr += 1

        if row.item_id not in item_map:
            item_map[row.item_id] = item_cntr
            item_cntr += 1


        df.loc[idx, 'user_id'] = user_map[row.user_id]
        df.loc[idx, 'item_id'] = item_map[row.item_id]

    return df, item_cntr, user_cntr


