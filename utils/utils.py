import numpy as np


def read_ratings(fname):

    raw_data = []
    with open(fname, 'r') as f:

        for line in f:
            row = line.split(",")
            raw_data.append(row)

    return np.array(raw_data)
