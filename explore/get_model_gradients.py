import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
from experiments.utils import read_train_test_dir
from preprocessing.utils import load_dict_output
import torch
import pandas as pd
import numpy as np
from model.trainer import NeuralUtilityTrainer
from model._loss import loss_mse
from sklearn.preprocessing import OneHotEncoder

# Set model and data paths manuall
# TODO: Ahmad to modify for his application
MODEL_PATH = cfg.vals["model_dir"] + "/encoder_movielens_mse_done.pt"
data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"


# Process data
X_train, X_test, y_train, y_test = read_train_test_dir(data_dir)
stats = load_dict_output(data_dir, "stats.json")

one_hot = OneHotEncoder(categories=[range(stats["n_items"])])
items_for_grad = one_hot.fit_transform(np.arange(stats["n_items"]).reshape(-1,1)).todense().astype(np.float32)


df = pd.DataFrame(np.concatenate([X_train, y_train], axis=1), columns=["user_id", "item_id", "rating"])
item_means = df[['item_id', 'rating']].groupby("item_id").mean()

users=None
items_for_grad = torch.from_numpy(items_for_grad)
y_true = torch.from_numpy(item_means.values.reshape(-1,1))


# Load Model
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))


# compute gradietns
gradients =  NeuralUtilityTrainer.get_gradient(model, loss_mse, users, items_for_grad, y_true)
print(gradients)