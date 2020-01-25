import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

import config.config as cfg
from preprocessing.meta_data_item_mapper import load_mapping_for

config = {
    'num_items': 100,
    'pantry': {
        'pp': cfg.vals['amazon_dir'] + "/preprocessed_pantry/",
        'model': cfg.vals['model_dir'] + "/pantry_ahmad/"
    },
    'home_kitchen': {
        'pp': cfg.vals['amazon_dir'] + "/preprocessed_home_kitchen/",
        'model': cfg.vals['model_dir'] + "/home_kitchen_ahmad/"
    },
    'grocery': {
        'pp': cfg.vals['amazon_dir'] + "/preprocessed_grocery/",
        'model': cfg.vals['model_dir'] + "/grocery_ahmad/"
    }
}


def load_mm_model_for(name):
    model_dir = config[name]['model']
    mm_dir = config[name]['pp']
    model = torch.load(model_dir + '/item_encoder_amazon_utility_done.pt')
    mm = load_mapping_for(mm_dir, name)
    return mm, model


def load_items(name):
    df = pd.read_csv(config[name]['pp'] + "/x_train.csv")
    df = df['item_id']
    # dedup
    items = set(df.values.tolist())
    return items


def map_id_to_title(mm, idx):
    return mm[mm[(str(int(idx)))]]  # lol float -> int -> string -> asin -> title


def do_explore(name):
    items = load_items(name)
    mapping, model = load_mm_model_for(name)

    arr = np.empty((0, 256))
    weights = model.embedding.weights.weight.data.to('cpu')
    labels = []
    i = 0

    for val in items:
        vec = weights[:, int(val)]
        arr = np.append(arr, [vec.numpy()], axis=0)
        labels.append(map_id_to_title(mapping, val))
        i += 1
        if i == config['num_items']:
            break

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


# do_explore('grocery')
