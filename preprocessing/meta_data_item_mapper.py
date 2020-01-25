import os
import pickle
import sys

from preprocessing.utils import load_dict_output, pandas_df

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def create_meta_mapping(name, pp_data_dir, meta_data_file):
    records = get_amazon_metadata(meta_data_file)
    item_mapping = load_dict_output(pp_data_dir, 'id_item_map.json')
    asin_mapping = dict()
    data_dir = pp_data_dir
    name = name
    for record in records:
        asin_mapping[record['asin']] = record['title']

    item_mapping.update(asin_mapping)
    with open(data_dir + '/' + name + '_meta_map.pt', 'wb') as f:
        pickle.dump(item_mapping, f)


@staticmethod
def get_amazon_metadata(m_file):
    print("getting data from {}".format(m_file))
    df = pandas_df(m_file)
    df = df[['title', 'asin']]
    return df.to_dict('record')


def load_mapping_for(data_dir, name):
    with open(data_dir + '/' + name + '_meta_map.pt', 'rb') as f:
        mm = pickle.load(f)
    return mm

# dir = cfg.vals['amazon_dir']
# m_file = dir + "meta/meta_Home_and_Kitchen.json.gz"
# p_file = dir + "preprocessed_home_kitchen/"
# create_meta_mapping('home_kitchen', p_file, m_file)
