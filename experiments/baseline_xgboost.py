import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from generator.generator import Generator

df = pd.read_csv(cfg.vals['movielens_dir'] + "/ratings.csv",nrows=1000)
df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
df.drop('timestamp', axis=1, inplace=True)

X = df[['user_id', 'item_id']]
y = df['rating']

X = Generator.get_one_hot_encodings(X)

X_train, X_test, y_train, y_test = Generator.split_train_test_user(X, y)

X_train = X_train.drop(['user_id', 'item_id'], axis=1).values
X_test = X_test.drop(['user_id', 'item_id'], axis=1).values


xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=10)
xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(rmse)