import numpy
import pandas as pd
import pymongo

from config import config
from preprocessing.meta_data_item_mapper import get_meta_config

Vector = numpy.ndarray

config = get_meta_config()


def load_items(name):
    df = pd.read_csv(config[name]['pp'] + "/x_train.csv")
    df = df['item_id']
    # dedup
    items = set(df.values.tolist())
    return items


def load_meta_into(name):
    class Item:
        def __init__(self, asin: str, idx: int, title: str, vector: Vector = None, cats=None) -> None:
            if cats is None:
                cats = []
            try:
                if "var aPage" in title:
                    self.title = "omitted html"
            except:
                self.title = "exception"
            else:
                self.title = title
            self.asin = asin
            self.idx = idx
            self.vector = vector
            self.cats = cats

    def create_items(meta_name):
        from preprocessing.meta_data_item_mapper import MetaDataMap
        mm = MetaDataMap(get_meta_config())
        nm = mm.get_all()
        item_map = load_items(meta_name)
        catmap = mm.get_cat(meta_name)
        items = []
        for val in item_map:
            # vec = weights[:, int(val)]
            asin = nm[str(int(val))]
            try:
                title = nm[nm[str(int(val))]]
            except:
                title = 'error'
            if asin in catmap:
                cat = catmap[asin]
            else:
                cat = []
            items.append(Item(asin=asin, idx=int(val), title=title, cats=cat))
        return items

    client = pymongo.MongoClient("localhost", 27017)
    db = client['utility']
    db.drop_collection(name)
    collection = db[name]
    items = create_items(name)

    def create_item_record(data: Item):
        return {
            "asin": data.asin,
            "title": data.title.strip(),
            "idx": data.idx,
            "cats": data.cats,
        }

    def insert(data):
        collection.insert_many(data)

    records = []
    for item in items:
        records.append(create_item_record(item))

    insert(records)
    return records


# items = load_meta_into('home_kitchen')

def search_collection_for_term(collection, term):
    results = []
    for res in collection.find({"title": {'$regex': term}}):
        results.append(res)
    return results


def get_db():
    client = pymongo.MongoClient("localhost", 27017)
    db = client['utility']
    return db


def get_collection(name):
    client = pymongo.MongoClient("localhost", 27017)
    db = client['utility']
    return db[name]


col = get_collection('home_kitchen')

