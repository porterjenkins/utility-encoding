import pandas as pd
import numpy as np


def write_embeddings(arr, n_nodes, fname):

    h_dim = arr.shape[1]

    with open(fname, 'w') as f:
        f.write("{} {} \n".format(n_nodes, h_dim))

        # for cntr, embedding_vector in enumerate(arr):
        for region_idx in range(n_nodes):
            embedding_vector = arr[region_idx, :]
            f.write("{} ".format(region_idx))

            cntr = 0
            for element in embedding_vector:
                if cntr == (h_dim - 1):
                    f.write("{}".format(element))
                else:
                    f.write("{} ".format(element))

                cntr += 1

            f.write("\n")


def load_embedding(fname):
    """
    Load embedding matrix from text file
        - assumes first row details the dimensions of the matrix (e.g., 2500 x 64)
        - assumes that first element of each row is an int denoting the nod index
        - read data, put into dataframe, and sort by index
    :param fname: str
    :return: (np.array) Sorted 2-d array of embedding vectors
    """
    deepwalk_features = list()
    idx = list()

    with open(fname, 'rb') as f:
        cntr = 0
        for line in f:
            if cntr > 0:
                row = line.decode('utf-8').split(" ")
                # skip row if no id
                if row[0] == '':
                    continue
                row_float = []
                for i, element in enumerate(row):
                    # skip 0th element - tract id
                    if i == 0:
                        idx.append(int(element))
                    else:
                        row_float.append(float(element))
                deepwalk_features.append(row_float)

            cntr += 1

    feature_mtx = pd.DataFrame(deepwalk_features, index=idx)
    feature_mtx.sort_index(inplace=True)
    return feature_mtx.values


def embedding_to_df(embedding_arr):
    dims = embedding_arr.shape
    ids = np.arange(dims[0])
    cols = ['id'] + ["e_{}".format(x) for x in range(dims[1])]
    X = np.concatenate((ids.reshape(-1,1), embedding_arr), axis=1)
    df = pd.DataFrame(X, columns=cols)
    df['id'] = df['id'].astype(int)
    return df