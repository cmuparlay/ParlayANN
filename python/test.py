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
    indptr = np.memmap(fname, dtype='int64', mode='r', offset=ofs, shape=nrow + 1)
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

# print("!!! FILTERED IVF !!!")
# fivf = wp.init_filtered_ivf_index("Euclidian", "uint8")
# fivf.fit_from_filename(DATA_DIR + "data/yfcc100M/base.10M.u8bin.crop_nb_10000000", DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat', 1000)

# print("----- Querying Filtered IVF Index... -----")

# X = np.fromfile(DATA_DIR + "data/yfcc100M/query.public.100K.u8bin", dtype=np.uint8)[8:].reshape((100_000, 192))
# filters = [wp.QueryFilter(3432, 3075) for _ in range(50_000)] + [wp.QueryFilter(23) for _ in range(50_000)]

# neighbors, distances = fivf.batch_search(X, filters, 100_000, 10, 100)
# print(neighbors.shape)
# print(neighbors[:10, :])
# print(distances[:10, :])
#                           UNCOMMENT
print("----- Building Squared IVF index... -----")

CUTOFF = 20_000
CLUSTER_SIZE = 1000
NQ = 100_000
TARGET_POINTS = 20_000
SQ_TARGET_POINTS = 5000
TINY_CUTOFF = 1000

start = time.time()

index = wp.init_squared_ivf_index("Euclidian", "uint8")
index.fit_from_filename(DATA_DIR + "data/yfcc100M/base.10M.u8bin.crop_nb_10000000", DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat', CUTOFF, CLUSTER_SIZE, "index_cache/", (75_000, 400_000))

print(f"Time taken: {time.time() - start:.2f}s")

print("----- Querying Squared IVF Index... -----")
start = time.time()

X = np.fromfile(DATA_DIR + "data/yfcc100M/query.public.100K.u8bin", dtype=np.uint8)[8:].reshape((100_000, 192))
filters = read_sparse_matrix(DATA_DIR + 'data/yfcc100M/query.metadata.public.100K.spmat')

rows, cols = filters.nonzero()
filter_dict = defaultdict(list)

for row, col in zip(rows, cols):
    filter_dict[row].append(col)

# filters = [wp.QueryFilter(*filters[i]) for i in filters.keys()]
filters = [None] * len(filter_dict.keys())
for i in filter_dict.keys():
    filters[i] = wp.QueryFilter(*filter_dict[i])



index.set_target_points(TARGET_POINTS)
index.set_sq_target_points(SQ_TARGET_POINTS)
index.set_tiny_cutoff(TINY_CUTOFF)

neighbors, distances = index.batch_filter_search(X, filters, NQ, 10)

index.print_stats()
#                           UNCOMMENT
all_filters = wp.csr_filters(DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat').transpose()

if NQ <= 10:
    print(filters[:NQ])

    print([(all_filters.point_count(i.a), all_filters.point_count(i.b)) if i.is_and() else all_filters.point_count(i.a) for i in filters[:NQ]])
else:
    print(filters[:10])

    print([(all_filters.point_count(i.a), all_filters.point_count(i.b)) if i.is_and() else all_filters.point_count(i.a) for i in filters[:10]])

print(neighbors.shape)
print(neighbors[:10, :])
print(distances[:10, :])

elapsed = time.time() - start
print(f"Time taken: {elapsed:.2f}s")
print(f"QPS: {NQ / elapsed:,.2f}")

# Calculate and print average recall
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
print(len(I))
print(len(D))

total_recall = 0.0
for i in range(100000):
    ground_truth = set()
    ann = set()
    for j in range(10):
        ground_truth.add(I[i][j])
        ann.add(neighbors[i][j])
    local_recall = len(ground_truth & ann)
    total_recall += local_recall/10
avg_recall = total_recall / 100000

print(f"Average recall is {100*avg_recall:.2f}%.")


#print("----- Building 2 Stage Filtered IVF... -----")
#start = time.time()
#
#fivf2 = wp.init_2_stage_filtered_ivf_index("Euclidian", "uint8")
#fivf2.fit_from_filename(DATA_DIR + "data/yfcc100M/base.10M.u8bin.crop_nb_10000000", DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat', 1000)
#
#print(f"Time taken: {time.time() - start:.2f}s")
#
#print("----- Querying 2 Stage Filtered IVF Index... -----")
#start = time.time()
#
#NQ = 10_000
#
#X = np.fromfile(DATA_DIR + "data/yfcc100M/query.public.100K.u8bin", dtype=np.uint8)[8:].reshape((100_000, 192))
#filters = read_sparse_matrix(DATA_DIR + 'data/yfcc100M/query.metadata.public.100K.spmat')
#
#rows, cols = filters.nonzero()
#filter_dict = defaultdict(list)
#
#for row, col in zip(rows, cols):
#    filter_dict[row].append(col)
#
## filters = [wp.QueryFilter(*filters[i]) for i in filters.keys()]
#filters = [None] * len(filter_dict.keys())
#for i in filter_dict.keys():
#    filters[i] = wp.QueryFilter(*filter_dict[i])
#
#if NQ < 10:
#    print(filters[:NQ])
#
#
#N_LISTS = 100
#CUTOFF = 20_000
#
#print(f"n_lists: {N_LISTS:,}")
#print(f"cutoff: {CUTOFF:,}")
#
#neighbors, distances = fivf2.batch_search(X, filters, NQ, 10, N_LISTS, CUTOFF)
#print(neighbors.shape)
#print(neighbors[:10, :])
#print(distances[:10, :])
#
#elapsed = time.time() - start
#print(f"Time taken: {elapsed:.2f}s")
#print(f"QPS: {NQ / elapsed:.2f}s")
#
#
#
#print("----- Building IVF Index... -----")
#start = time.time()
#
#ivf = wp.init_ivf_index("Euclidian", "uint8")
## ivf.fit_from_filename(DATA_DIR + "base.1B.u8bin.crop_nb_1000000", 1000)
#ivf.fit_from_filename(DATA_DIR + "data/yfcc100M/base.10M.u8bin.crop_nb_10000000", 1000)
#ivf.print_stats()
#
#
#
## # neighbors, distances = Index.batch_search_from_string(DATA_DIR + "query.public.10K.u8bin", 10000, 10, 100)
#
## query = np.fromfile(DATA_DIR + "query.public.10K.u8bin", dtype=np.uint8)[8:].reshape((10000, 128))
## print(query.shape)
## neighbors, distances = ivf.batch_search(query, 10000, 10, 100)
#
## print(neighbors.shape)
## print(neighbors[:10, :])
## print(distances[:10, :])
## # Index.check_recall(DATA_DIR + "bigann-1M", neighbors, 10)
#
#print("----- Testing Filters... -----")
#
#filters = wp.csr_filters(DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat')
#
#print(filters.first_label(42))
#print(filters.match(42, 6))
#print(filters.match(42, 2))
#
#print(f"Point count of filter 23: {filters.filter_count(23):,}")
#
#print(f"Filter count of point 42: {filters.point_count(42):,}")
#print(f"Point count of filter 6: {filters.filter_count(6):,}")
#
#print("Transposing... (from python)")
#
#filters.transpose_inplace()
#filters_t = filters
#
## filters_t = filters.transpose()
#
#print("Transposed! (from python)")
#
#print(filters_t.first_label(6)) # should be 42
#print(filters_t.match(6, 42)) # should be True
#print(filters_t.match(2, 42)) # should be False
#
#print(f"Filter count of point 42: {filters_t.filter_count(42):,}")
#print(f"Point count of filter 6: {filters_t.point_count(6):,}")
#
## wp.build_vamana_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/parlayann", 64, 128, 1.2)
#
## Index = wp.load_vamana_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/parlayann", 1000000, 128)
#
#
#
## print("Filter Query")
#
## query_a = wp.QueryFilter(1)
## query_b = wp.QueryFilter(1, 2)
#
## print(query_a.is_and()) # should be False
## print(query_b.is_and()) # should be True
#
#
#
