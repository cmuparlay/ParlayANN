import pandas as pd
from datetime import datetime
import time
from collections import defaultdict

import os
import _ParlayANNpy as pann
import numpy as np
import wrapper as wp
from scipy.sparse import csr_matrix


def mmap_sparse_matrix_fields(fname):
    """ mmap the fields of a CSR matrix without instanciating it """
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
    """ read the fields of a CSR matrix without instanciating it """
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


print(dir(pann))

FERN_DATA_DIR = "/ssd1/anndata/bigann/"
AWARE_DATA_DIR = "/ssd1/data/bigann/"

DATA_DIR = FERN_DATA_DIR

BUILD = True

SMALL_R = 32
SMALL_L = 100
SMALL_ALPHA = 1.175

LARGE_R = 64
LARGE_L = 100
LARGE_ALPHA = 1.175

QUERY_BEAM_WIDTH = 100
CUT = 1.35
SEARCH_LIMIT = 10_000_000
QUERY_LIMIT = 10_000_000

HYBRID_CUTOFF = 20_000

# sv_index = wp.init_stitched_vamana_index("Euclidian", "uint8")
sv_index = wp.init_hybrid_stitched_vamana_index("Euclidian", "uint8")

sv_index.set_cutoff(HYBRID_CUTOFF)

sv_index.set_build_params_small(SMALL_R, SMALL_L, SMALL_ALPHA)
sv_index.set_build_params_large(LARGE_R, LARGE_L, LARGE_ALPHA)

sv_index.set_query_params(wp.QueryParams(10, QUERY_BEAM_WIDTH, CUT, SEARCH_LIMIT, QUERY_LIMIT))


if BUILD:
    print("----- Building Stitched Vamana Index-----")

    sv_index.fit_from_filename(DATA_DIR + "data/yfcc100M/base.10M.u8bin.crop_nb_10000000", DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat')

    sv_index.save("index_cache/")
else:
    print("----- Loading Stitched Vamana Index-----")

    sv_index.load_from_filename("index_cache/", DATA_DIR + "data/yfcc100M/base.10M.u8bin.crop_nb_10000000", DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat')

print("----- Querying Stitched Vamana Index-----")

X = np.fromfile(DATA_DIR + "data/yfcc100M/query.public.100K.u8bin",
                dtype=np.uint8)[8:].reshape((100_000, 192))
filters = read_sparse_matrix(
    DATA_DIR + 'data/yfcc100M/query.metadata.public.100K.spmat')

rows, cols = filters.nonzero()
filter_dict = defaultdict(list)

for row, col in zip(rows, cols):
    filter_dict[row].append(col)

filters = [None] * len(filter_dict.keys())
for i in filter_dict.keys():
    filters[i] = wp.QueryFilter(*filter_dict[i])

sq_filters = [f for f in filters if not f.is_and()]
sq_queries = X[[i for i, f in enumerate(filters) if not f.is_and()]]

NQ = len(sq_queries)

print(f"Number of queries: {NQ}")
# print(f"Number of filters: {len(sq_filters)}")
# print(sq_queries.shape)

start = time.time()
neighbors, distances = sv_index.batch_filter_search(sq_queries, sq_filters, NQ, 10)
end = time.time()

print(f"Time taken: {end - start}s")
print(f"QPS: {NQ / (end - start)}")

print(neighbors.shape)
print(neighbors[:10, :])
print(distances[:10, :])

# print(f"Average distance comparisons: {sv_index.get_dist_comparisons() / NQ}")

GROUND_TRUTH_DIR = DATA_DIR + "data/yfcc100M/GT.public.ibin"

def retrieve_ground_truth(fname):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
    f = open(fname, "rb")
    f.seek(4+4)
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return I, D


I, D = retrieve_ground_truth(GROUND_TRUTH_DIR)

sq_I = I[[i for i, f in enumerate(filters) if not f.is_and()]]
sq_D = D[[i for i, f in enumerate(filters) if not f.is_and()]]

recall = 0

for i in range(NQ):
    ground_truth = set(I[i][:10])
    ann = set(neighbors[i][:10])
    local_recall = len(ground_truth & ann)

    recall += local_recall / 10

recall /= NQ
print(f"Recall: {recall * 100:.4f}% (1 / {1 / recall:.2f})")

