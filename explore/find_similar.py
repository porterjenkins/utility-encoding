from typing import List, Tuple

import numpy
import pandas as pd
import torch
from numpy.linalg import norm
from scipy import spatial

from preprocessing.meta_data_item_mapper import MetaDataMap, get_meta_config

Vector = numpy.ndarray
config = get_meta_config()


class Item:
    def __init__(self, asin: str, idx: int, title: str, vector: Vector) -> None:
        if "var aPage" in title:
            self.title = "omitted html"
        else:
            self.title = title
        self.asin = asin
        self.idx = idx
        self.vector = vector


def print_tuple(tup):
    return tup[0] + " " + str(tup[1])


def dot_product(a: Vector, b: Vector) -> float:
    return norm(a) * norm(b)


def cosine_similarity(a: Vector, b: Vector) -> float:
    return 1 - spatial.distance.cosine(a, b)
    # return dot_product(a, b) / (norm(a) * norm(b))


def vector(asin: str) -> Vector:
    return find_item(items, asin).vector


def load_model_for(name):
    model_dir = config[name]['model']
    model = torch.load(model_dir + '/item_encoder_amazon_utility_done.pt')
    return model


def by_similarity(items: List[Item], start: Vector) -> List[Tuple[float, Item]]:
    item_distances = [(cosine_similarity(start, i.vector), i) for i in items]
    return sorted(item_distances, key=lambda t: t[0], reverse=True)


def print_related(items: List[Item], asin: str) -> None:
    start = find_item(items, asin)
    found = [
        print_tuple([item.title, dist]) for (dist, item) in
        by_similarity(items, start.vector)
        if item.asin.lower() != start.asin.lower()
    ]
    print('\n'.join(found[:100]))


def find_item(items: List[Item], asin: str) -> Item:
    return next(i for i in items if asin == i.asin)


def load_items(name):
    df = pd.read_csv(config[name]['pp'] + "/x_train.csv")
    df = df['item_id']
    # dedup
    items = set(df.values.tolist())
    return items


def create_items(meta_name):
    mm = MetaDataMap(get_meta_config())
    nm = mm.get_all()

    model = load_model_for(meta_name)
    weights = model.embedding.weights.weight.data.to('cpu')
    item_map = load_items(meta_name)
    items = []
    for val in item_map:
        vec = weights[:, int(val)]
        asin = nm[str(int(val))]
        title = nm[nm[str(int(val))]]
        items.append(Item(asin=asin, idx=int(val), vector=vec, title=title))
    return items


def add(v1: Vector, v2: Vector) -> Vector:
    return numpy.add(v1, v2)


def sub(v1: Vector, v2: Vector) -> Vector:
    return numpy.subtract(v1, v2)


def closest_analogies(
        left2: Item, left1: Item, right2: Item, items: List[Item]
) -> List[Tuple[float, Item]]:
    v = add(sub(left1.vector, left2.vector), right2.vector)
    closest = by_similarity(items, v)[:150]

    return closest


def get_analogy(left2: str, left1: str, right2: str, items: List[Item]) -> List[Tuple[float, Item]]:
    item_left1 = find_item(items, left1)
    item_left2 = find_item(items, left2)
    item_right2 = find_item(items, right2)
    analogies = closest_analogies(item_left2, item_left1, item_right2, items)

    print(f"\n\n{item_left2.title}-{item_left1.title} is like {item_right2.title}- ?????\n\n")

    for i in range(0, len(analogies)):
        (dist, w) = analogies[i]
        print(f"{w.title} with score {dist}")
    return analogies


DATASET = 'grocery'
items = create_items(DATASET)
# print_related(items, 'B00DX53P4M')
print("Similar Loading")

a = get_analogy('B00DX53P4M', 'B00ODEN5M4', 'B005HPTV9Y', items)
