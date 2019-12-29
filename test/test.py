import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import pandas as pd
import json
import gzip
from utils.utils import read_ratings
import numpy as np


fname = cfg.vals["amazon_dir"] + "/grocery_and_gourmet_food/Grocery_and_Gourmet_Food.csv"


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

