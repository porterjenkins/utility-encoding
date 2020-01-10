import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.utils import preprocess_user_item_df, write_dict_output
import pandas as pd
import config.config as cfg




df = pd.read_csv(cfg.vals['movielens_dir'] + "/ratings.csv", nrows=10000)
df.columns = ['user_id', 'item_id', 'rating', 'timestamp']



df, user_item_rating_map, item_rating_map, user_id_map, id_user_map, item_id_map, id_item_map, stats = preprocess_user_item_df(df)

out_dir = cfg.vals['movielens_dir'] + "/preprocessed/"


write_dict_output(out_dir, "user_item_rating.json", user_item_rating_map)
write_dict_output(out_dir, "item_rating.json", item_rating_map)
write_dict_output(out_dir, "user_id_map.json", user_id_map)
write_dict_output(out_dir, "id_user_map.json", id_user_map)
write_dict_output(out_dir, "item_id_map.json", item_id_map)
write_dict_output(out_dir, "id_item_map.json", id_item_map)
write_dict_output(out_dir, "stats.json", stats)



df.to_csv(out_dir + "ratings.csv", index=False)