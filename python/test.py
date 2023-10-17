import _ParlayANNpy as pann
import wrapper as wp
import numpy as np
import time
from scipy.sparse import csr_matrix
from collections import defaultdict

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

print("----- Building 2 Stage Filtered IVF... -----")
start = time.time()

fivf2 = wp.init_2_stage_filtered_ivf_index("Euclidian", "uint8")
fivf2.fit_from_filename(DATA_DIR + "data/yfcc100M/base.10M.u8bin.crop_nb_10000000", DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat', 1000)

print(f"Time taken: {time.time() - start:.2f}s")

print("----- Querying 2 Stage Filtered IVF Index... -----")
start = time.time()

NQ = 2

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

if NQ < 10:
    print(filters[:NQ])


N_LISTS = 1000
CUTOFF = 10_000_000

print(f"n_lists: {N_LISTS:,}")
print(f"cutoff: {CUTOFF:,}")

neighbors, distances = fivf2.batch_search(X, filters, NQ, 10, N_LISTS, CUTOFF)
print(neighbors.shape)
print(neighbors[:10, :])
print(distances[:10, :])

elapsed = time.time() - start
print(f"Time taken: {elapsed:.2f}s")
print(f"QPS: {NQ / elapsed:.2f}s")

print("----- Building IVF Index... -----")
start = time.time()

ivf = wp.init_ivf_index("Euclidian", "uint8")
# ivf.fit_from_filename(DATA_DIR + "base.1B.u8bin.crop_nb_1000000", 1000)
ivf.fit_from_filename(DATA_DIR + "data/yfcc100M/base.10M.u8bin.crop_nb_10000000", 1000)
ivf.print_stats()

# # neighbors, distances = Index.batch_search_from_string(DATA_DIR + "query.public.10K.u8bin", 10000, 10, 100)

# query = np.fromfile(DATA_DIR + "query.public.10K.u8bin", dtype=np.uint8)[8:].reshape((10000, 128))
# print(query.shape)
# neighbors, distances = ivf.batch_search(query, 10000, 10, 100)

# print(neighbors.shape)
# print(neighbors[:10, :])
# print(distances[:10, :])
# # Index.check_recall(DATA_DIR + "bigann-1M", neighbors, 10)

print("----- Testing Filters... -----")

filters = wp.csr_filters(DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat')

print(filters.first_label(42))
print(filters.match(42, 6))
print(filters.match(42, 2))

print(f"Point count of filter 23: {filters.filter_count(23):,}")

print(f"Filter count of point 42: {filters.point_count(42):,}")
print(f"Point count of filter 6: {filters.filter_count(6):,}")

print("Transposing... (from python)")

filters.transpose_inplace()
filters_t = filters

# filters_t = filters.transpose()

print("Transposed! (from python)")

print(filters_t.first_label(6)) # should be 42
print(filters_t.match(6, 42)) # should be True
print(filters_t.match(2, 42)) # should be False

print(f"Filter count of point 42: {filters_t.filter_count(42):,}")
print(f"Point count of filter 6: {filters_t.point_count(6):,}")

# wp.build_vamana_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/parlayann", 64, 128, 1.2)

# Index = wp.load_vamana_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/parlayann", 1000000, 128)



# print("Filter Query")

# query_a = wp.QueryFilter(1)
# query_b = wp.QueryFilter(1, 2)

# print(query_a.is_and()) # should be False
# print(query_b.is_and()) # should be True



