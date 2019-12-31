import numpy as np
import gzip
import pandas as pd

true = True
false = False



def read_ratings(fname):

    raw_data = []
    with open(fname, 'r') as f:

        for line in f:
            row = line.split(",")
            raw_data.append(row)

    return np.array(raw_data)


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