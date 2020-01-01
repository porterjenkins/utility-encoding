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