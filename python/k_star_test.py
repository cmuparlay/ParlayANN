# %%
import pandas as pd
from datetime import datetime
import time
from collections import defaultdict

import os
import _ParlayANNpy as pann
import numpy as np
import wrapper as wp
from scipy.sparse import csr_matrix

# %%
# DATA_DIR = "/scratch/diversity/"
# BASE = os.path.join(DATA_DIR, "base_uk.ivec")
# LABELS = os.path.join(DATA_DIR, "base_uk_seller_csr.spmat")
# QUERIES = os.path.join(DATA_DIR, "query_uk_popular.ivec")
# INDEX_CACHE = "index_cache/diversity/"

DATA_DIR = "/ssd1/anndata/bigann/data/yfcc100M/"
BASE = os.path.join(DATA_DIR, "base.10M.u8bin.crop_nb_10000000")
LABELS = os.path.join(DATA_DIR, "base.metadata.10M.spmat")
QUERIES = os.path.join(DATA_DIR, "query.public.100K.u8bin")
INDEX_CACHE = "index_cache/yfcc100M/"
# %%
GRAPH_NAME = "vamana_yfcc100M"
# build vamana graph

start = time.time()
# wp.build_vamana_index("Euclidian", "uint8", BASE, GRAPH_NAME, 64, 100, 1.2)
print(f"build time: {time.time() - start:.2f}s")

# %%
# load vamana graph
index = wp.load_vamana_index("Euclidian", "uint8", BASE, GRAPH_NAME, 10_000_000, 192)

# %%

print("loaded index")

# %%
print("building k_star index")
start = time.time()
index.k_star_build_index(64, 100, 1.2, 3)
# %%
# load labels
index.load_filters(LABELS)

# %%
print("loaded labels")
# %%
print("loading queries")

def ivec_to_numpy(file):
    X = np.fromfile(file, dtype=np.int32)
    n = X[0]
    d = X[1]
    return np.frombuffer(X, dtype=np.uint8, offset=8).reshape(n, d)

queries = ivec_to_numpy(QUERIES)

print(queries.shape)
# %%
neighbors, distances = index.batch_search(queries, queries.shape[0], 10, 100)



# %%
kstar_neighbors, kstar_distances = index.k_star_batch_query(queries, queries.shape[0], 10, 100, 3)

print(neighbors[0, :])
print(kstar_neighbors[0, :])
print(distances[0, :])
print(kstar_distances[0, :])

# %%
# exit()

# %%
print(f"are the sets of neighbors the same? {'yes' if np.all(neighbors == kstar_neighbors) else 'no'}")
print(f"on how many rows are the sets of neighbors different? {np.sum(np.any(neighbors != kstar_neighbors, axis=1))} ({np.sum(np.any(neighbors != kstar_neighbors, axis=1)) / queries.shape[0] * 100:.2f}%)")
print(f"on how many of the vanilla distance rows does the distance decrease? {np.sum(np.any(kstar_distances < distances, axis=1))} ({np.sum(np.any(kstar_distances < distances, axis=1)) / queries.shape[0] * 100:.2f}%)")
print(f"how much farther proportionally are the kstar neighbors? {np.mean(kstar_distances) / np.mean(distances) * 100:.2f}%")
# %%
# we load the metadata to check the diversity of the neighbors

def mmap_sparse_matrix_fields(fname):
    """ mmap the fields of a CSR matrix without instantiating it """
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
    ofs = sizes.nbytes
    indptr = np.memmap(fname, dtype='int64', mode='r',
                       offset=ofs, shape=nrow + 1)
    ofs += indptr.nbytes
    indices = np.memmap(fname, dtype='int32', mode='r', offset=ofs, shape=nnz)
    ofs += indices.nbytes
    data = np.memmap(fname, dtype='float32', mode='r', offset=ofs, shape=nnz)
    return data, indices, indptr, ncol


def read_sparse_matrix_fields(fname):
    """ read the fields of a CSR matrix without instantiating it """
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype='int64', count=nrow + 1)
        assert nnz == indptr[-1]
        indices = np.fromfile(f, dtype='int32', count=nnz)
        assert np.all(indices >= 0) and np.all(indices < ncol)
        data = np.fromfile(f, dtype='float32', count=nnz)
        return data, indices, indptr, ncol


def read_sparse_matrix(fname, do_mmap=False):
    """ read a CSR matrix in spmat format, optionally mmapping it instead """
    if not do_mmap:
        data, indices, indptr, ncol = read_sparse_matrix_fields(fname)
    else:
        data, indices, indptr, ncol = mmap_sparse_matrix_fields(fname)

    return csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, ncol))

metadata = read_sparse_matrix(LABELS)
print(metadata.shape)
# %%
# get first nonzero in each row
first_label = metadata.indices[metadata.indptr[:-1]]
# print(first_label)
# %%
neighbors_first_label = [first_label[[x for x in neighbors[i, :] if x < len(first_label)]] for i in range(neighbors.shape[0])]
kstar_neighbors_first_label = [first_label[[x for x in kstar_neighbors[i, :] if x < len(first_label)]] for i in range(kstar_neighbors.shape[0])]

# %%
print(f"average unique labels per query for beam search: {np.mean([len(np.unique(x)) for x in neighbors_first_label])} (max count for label: {np.max([np.max(np.unique(x, return_counts=True)[1]) for x in neighbors_first_label])})")
print(f"average unique labels per query for k-star search: {np.mean([len(np.unique(x)) for x in kstar_neighbors_first_label])} (max count for label: {np.max([np.max(np.unique(x, return_counts=True)[1]) for x in kstar_neighbors_first_label])})")