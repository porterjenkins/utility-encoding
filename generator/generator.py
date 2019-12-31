import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
from utils.utils import parse, pandas_df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class Generator(object):

    def __init__(self, X, y):
        self.x_cols = ['user_id', 'item_id']
        self.y_cols = ['rating']

        if not isinstance(X, pd.DataFrame):
            self.X = pd.DataFrame(X, columns=self.x_cols)
        else:
            self.X = X
        if not isinstance(y, pd.DataFrame):
            self.y = pd.DataFrame(y, columns=self.y_cols)
        else:
            self.y = y


    @classmethod
    def get_one_hot_encodings(cls, X):


        item_encodings = pd.get_dummies(X['item_id'])
        user_encodings = pd.get_dummies(X['user_id'])

        return pd.concat([X, user_encodings, item_encodings], axis=1)


    @classmethod
    def split_train_test_user(cls, X, y, test_size=.2):

        assert 'user_id' in list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify=X['user_id'])

        return X_train, X_test, y_train, y_test




if __name__ == "__main__":

    df = pd.read_csv(cfg.vals['movielens_dir'] + "/ratings.csv",nrows=1000)
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df.drop('timestamp', axis=1, inplace=True)

    X = df[['user_id', 'item_id']]
    y = df['rating']

    X_train, X_test, y_train, y_test = Generator.split_train_test_user(X, y)

    gen = Generator(X=X_train, y=y_train)