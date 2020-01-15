import argparse

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.utils import preprocess_user_item_df, write_dict_output
import pandas as pd
import config.config as cfg
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--nrows", type = int, help="limit number of rows")
args = parser.parse_args()




df = pd.read_csv(cfg.vals['movielens_dir'] + "/ratings.csv", nrows=args.nrows)
df.columns = ['user_id', 'item_id', 'rating', 'timestamp']

X = df[['user_id', 'item_id', 'rating']]

arr, user_item_rating_map, item_rating_map, user_id_map, id_user_map, item_id_map, id_item_map, stats = preprocess_user_item_df(X)

out_dir = cfg.vals['movielens_dir'] + "/preprocessed/"


write_dict_output(out_dir, "user_item_rating.json", user_item_rating_map)
write_dict_output(out_dir, "item_rating.json", item_rating_map)
write_dict_output(out_dir, "user_id_map.json", user_id_map)
write_dict_output(out_dir, "id_user_map.json", id_user_map)
write_dict_output(out_dir, "item_id_map.json", item_id_map)
write_dict_output(out_dir, "id_item_map.json", id_item_map)
write_dict_output(out_dir, "stats.json", stats)


arr = np.concatenate([arr, df[['rating', 'timestamp']]], axis=1)

df = pd.DataFrame(arr, columns=['user_id', 'item_id', 'rating', 'timestamp'])

df.to_csv(out_dir + "ratings.csv", index=False)