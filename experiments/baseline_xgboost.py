import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from generator.generator import Generator
from utils.preprocessing import split_train_test_user, get_one_hot_encodings

df = pd.read_csv(cfg.vals['movielens_dir'] + "/ratings.csv", nrows=1000)
df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
df.drop('timestamp', axis=1, inplace=True)

X = df[['user_id', 'item_id']]
y = df['rating']

X = get_one_hot_encodings(X)

X_train, X_test, y_train, y_test = split_train_test_user(X, y)

X_train = X_train.drop(['user_id', 'item_id'], axis=1).values
x_test_user = X_test['user_id'].copy().values
X_test = X_test.drop(['user_id', 'item_id'], axis=1).values



xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=10)
print("training...")
xg_reg.fit(X_train, y_train)
print("complete")

preds = xg_reg.predict(X_test)

output = pd.DataFrame(np.concatenate([x_test_user.reshape(-1,1), preds.reshape(-1,1), y_test.values.reshape(-1,1)], \
                                    axis=1), columns = ['user_id', 'pred', 'y_true'])

output.sort_values(by=['user_id', 'pred'], inplace=True)
output = output.groupby('user_id').head(5)
output['rank'] = output[['user_id', 'pred']].groupby('user_id').rank(method='first').astype(float)
output['dcg'] = (np.power(2, output['y_true']) - 1) / np.log2(output['rank'] + 1)



rmse = np.sqrt(mean_squared_error(output.y_true, output.pred))

print(output)

avg_dcg = output.dcg.mean()
print(rmse)
print(avg_dcg)
