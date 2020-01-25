from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

from preprocessing.meta_data_item_mapper import get_meta_config, MetaDataMap

config = get_meta_config()

mm = MetaDataMap(get_meta_config())


def load_model_for(name):
    model_dir = config[name]['model']
    model = torch.load(model_dir + '/item_encoder_amazon_utility_done.pt')
    return model


def load_items(name):
    df = pd.read_csv(config[name]['pp'] + "/x_train.csv")
    df = df['item_id']
    # dedup
    items = set(df.values.tolist())
    return items


def map_id_to_title(name, idx):
    nm = mm.get_id_asin(name)
    return nm[nm[(str(int(idx)))]]  # lol float -> int -> string -> asin -> title


def map_asin_to_title(asin):
    nm = mm.get_all()
    return nm[asin]


def map_asin_id(name, asin):
    return mm.get_idx_for_asin(name, asin)


def map_id_to_cat(name, idx):
    asin = mm.get_id_asin(name)[(str(int(idx)))]
    nm = mm.get_cat(name)
    return nm[asin][0]


def get_also_bought(name, idx):
    asin = mm.get_id_asin(name)[(str(int(idx)))]
    bought = mm.get_bought(name)
    if asin in bought:
        return bought[asin]
    else:
        return None


def do_explore(name):
    items = load_items(name)
    model = load_model_for(name)

    arr = np.empty((0, 256))
    weights = model.embedding.weights.weight.data.to('cpu')
    labels = []
    i = 0
    items = sample(items, config['num_items'])

    for val in items:
        vec = weights[:, int(val)]
        arr = np.append(arr, [vec.numpy()], axis=0)
        labels.append(map_id_to_cat(name, val))

    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()


# hack
def do_explore_bought(name):
    items = load_items(name)
    model = load_model_for(name)

    weights = model.embedding.weights.weight.data.to('cpu')
    labels = []
    i = 0
    arr = np.empty((0, 256))
    while len(labels) < 2:
        items = sample(items, config['num_items'])
        arr = np.empty((0, 256))
        labels = []
        for val in items:
            vec = weights[:, int(val)]
            also_bought = get_also_bought(name, val)
            arr = np.append(arr, [vec.numpy()], axis=0)
            labels.append(map_id_to_title(name, val))

            if also_bought is not None:
                for b in also_bought:
                    b_id = map_asin_id(name, b)
                    if b_id is not None:
                        vec2 = weights[:, int(b_id)]
                        arr = np.append(arr, [vec2.numpy()], axis=0)
                        labels.append(map_asin_to_title(b))

    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()


do_explore_bought('grocery')
