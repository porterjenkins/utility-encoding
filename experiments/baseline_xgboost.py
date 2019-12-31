import gzip
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


true = True
false = False
def parse(path):
    g = gzip.open(path, 'rb')
    for line in g:
        yield eval(line)
def pandas_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')
df = pandas_df("/home/ahmad/projects/utility-encoding/data/amazon/Grocery_and_Gourmet_Food_5.json.gz")
df = df.drop(columns=[
    "reviewerName",
    "reviewText",
    "summary",
    "unixReviewTime",
    "reviewTime",
    "verified",
    "vote",
    "style",
    "image"
])
sample_fraction = .1
df = df.sample(frac=sample_fraction).reset_index(drop=True)
combined = pd.concat([df, pd.get_dummies(df['asin']), pd.get_dummies(df['reviewerID'])], axis=1)
combined.drop(columns=['reviewerID', 'asin'], axis=1, inplace=true)
y = combined['overall']
X = combined.drop(['overall'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=10)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))