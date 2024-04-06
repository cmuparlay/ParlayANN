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

ALT_FERN_DATA_DIR = "/ssd2/ben/nhqdatasets/"

DATA_DIR = FERN_DATA_DIR

AUDIO = True

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

print("----- Building Squared IVF index... -----")

if AUDIO:
    CUTOFF = 0
    CLUSTER_SIZE = 500
    NQ = 200
    WEIGHT_CLASSES = (5_000, 10_000)
    MAX_DEGREES = (8, 10, 12)

    TINY_CUTOFF = 1000 
    TARGET_POINTS = 500
    BEAM_WIDTHS = (85, 85, 85)
    SEARCH_LIMITS = (int(WEIGHT_CLASSES[0] * 0.2), int(WEIGHT_CLASSES[1] * 0.5), int(3_000_000 * 0.5))

    ALPHA = 0.11

    BITVECTOR_CUTOFF = 1000
else:
    CUTOFF = 5_000
    CLUSTER_SIZE = 5000
    NQ = 100_000
    WEIGHT_CLASSES = (100_000, 400_000)
    MAX_DEGREES = (8, 10, 12)

    TINY_CUTOFF = 60000
    TARGET_POINTS = 15_000
    BEAM_WIDTHS = (85, 85, 85)
    SEARCH_LIMITS = (int(WEIGHT_CLASSES[0] * 0.2), int(WEIGHT_CLASSES[1] * 0.5), int(3_000_000 * 0.5))

    ALPHA = 1.175

    BITVECTOR_CUTOFF = 10000



start = time.time()

if not os.path.exists("index_cache/"):
    os.mkdir("index_cache/")

if AUDIO:
    if not os.path.exists("index_cache/audio/"):
        os.mkdir("index_cache/audio/")
    CACHE_DIR = "index_cache/audio/"
else:
    if not os.path.exists("index_cache/yfcc100M/"):
        os.mkdir("index_cache/yfcc100M/")
    CACHE_DIR = "index_cache/yfcc100M/"

if AUDIO:
    index = wp.init_squared_ivf_index("mips", "float")
else:
    index = wp.init_squared_ivf_index("Euclidian", "uint8")

for i in range(3):
    index.set_build_params(wp.BuildParams(MAX_DEGREES[i], 200, ALPHA), i)
    index.set_query_params(wp.QueryParams(10, BEAM_WIDTHS[i], 1.35, SEARCH_LIMITS[i], MAX_DEGREES[i]), i)

index.set_bitvector_cutoff(BITVECTOR_CUTOFF)

if AUDIO:
    index.fit_from_filename(ALT_FERN_DATA_DIR + "audio/audio_base.fvec", ALT_FERN_DATA_DIR + "audio/label_audio_base.spmat", CUTOFF, CLUSTER_SIZE, CACHE_DIR, WEIGHT_CLASSES, True)
else:
    index.fit_from_filename(DATA_DIR + "data/yfcc100M/base.10M.u8bin.crop_nb_10000000", DATA_DIR +
                            'data/yfcc100M/base.metadata.10M.spmat', CUTOFF, CLUSTER_SIZE, CACHE_DIR, WEIGHT_CLASSES, True)
    


print(f"Time taken: {time.time() - start:.2f}s")

print("----- Querying Squared IVF Index... -----")
start = time.time()

if AUDIO:
    X = np.fromfile(ALT_FERN_DATA_DIR + "audio/audio_query.fvec", dtype=np.float32)
    filters = read_sparse_matrix(
        ALT_FERN_DATA_DIR + 'audio/label_audio_query.spmat')

    def restrict_csr_to_two_nonzeros(csr):
        rows, cols = csr.shape
        new_data = []
        new_indices = []
        new_indptr = [0]
        for i in range(rows):
            row = csr.getrow(i)
            # Get indices and values of the non-zeros
            nz_indices = row.indices
            nz_values = row.data
            # If there are more than two non-zeros, retain only the two highest
            if len(nz_values) > 2:
                # argsort returns indices that would sort the array, so take the last two for the highest values
                top_two_idx = np.argsort(nz_values)[-2:]
                new_data.extend(nz_values[top_two_idx])
                new_indices.extend(nz_indices[top_two_idx])
            else:
                new_data.extend(nz_values)
                new_indices.extend(nz_indices)
            new_indptr.append(len(new_data))

        return csr_matrix((new_data, new_indices, new_indptr), shape=(rows, cols))
    
    filters = restrict_csr_to_two_nonzeros(filters)
else:
    X = np.fromfile(DATA_DIR + "data/yfcc100M/query.public.100K.u8bin",
                    dtype=np.uint8)[8:].reshape((100_000, 192))
    filters = read_sparse_matrix(
        DATA_DIR + 'data/yfcc100M/query.metadata.public.100K.spmat')

rows, cols = filters.nonzero()
filter_dict = defaultdict(list)

for row, col in zip(rows, cols):
    filter_dict[row].append(col)

# filters = [wp.QueryFilter(*filters[i]) for i in filters.keys()]
filters = [None] * len(filter_dict.keys())
for i in filter_dict.keys():
    filters[i] = wp.QueryFilter(*filter_dict[i])


index.set_target_points(TARGET_POINTS)
# index.set_sq_target_points(SQ_TARGET_POINTS)
index.set_tiny_cutoff(TINY_CUTOFF)

neighbors, distances = index.batch_filter_search(X, filters, NQ, 10)

elapsed = time.time() - start

index.print_stats()

all_filters = wp.csr_filters(
    DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat').transpose()

# if NQ <= 10:
#     print(filters[:NQ])

#     print([(all_filters.point_count(i.a), all_filters.point_count(i.b))
#           if i.is_and() else all_filters.point_count(i.a) for i in filters[:NQ]])
# else:
#     print(filters[:10])

#     print([(all_filters.point_count(i.a), all_filters.point_count(i.b))
#           if i.is_and() else all_filters.point_count(i.a) for i in filters[:10]])

# print(neighbors.shape)
print(neighbors[:10, :])
# print(distances[:10, :])

print(f"Time taken: {elapsed:.2f}s")
print(f"QPS: {NQ / elapsed:,.2f}")

# Calculate and print average recall for each case
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
# print(len(I))
# print(len(D))
print(I[:10, :])

total_recall = 0.0
query_cases = defaultdict(float)
case_counts = defaultdict(int)
filters2 = wp.csr_filters(DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat')
filters2.transpose_inplace()

case_comparisons = defaultdict(int)
case_time = defaultdict(float)

case_sort_map = {"t": 0, "s": 1, "l": 2}

query_recall = [0] * NQ

cases_list = []
log = index.get_log()  # should be a list of (id, comparisons, time) tuples

print(len(log))

for i in range(NQ):
    ground_truth = set(I[i][:10])
    ann = set(neighbors[i][:10])
    
    local_recall = len(ground_truth & ann)

    case = ""

    # sort size to get smaller filter first
    filter_a_size = filters2.point_count(filters[i].a)
    filter_b_size = 0
    if filters[i].b != -1:
        filter_b_size = filters2.point_count(filters[i].b)
    if filter_b_size > filter_a_size:
        filter_a_size, filter_b_size = filter_b_size, filter_a_size

    # build case string
    if filter_a_size <= TINY_CUTOFF:
        case += "tiny"
    elif filter_a_size <= CUTOFF:
        case += "small"
    else:
        case += "large"

    if filters[i].b != -1:
        if filter_b_size <= TINY_CUTOFF:
            case += "xtiny"
        elif filter_b_size <= CUTOFF:
            case += "xsmall"
        else:
            case += "xlarge"

    # tinyxsmall is the same as tinyxlarge
    if case == "tinyxsmall" or case == "smallxtiny" or case == "largextiny":
        case = "tinyxlarge"

    # tinyxtiny is the same as smallxsmall
    if case == "tinyxtiny":
        case = "smallxsmall"

    # casing on weight classes
    if case == "large":
        if filter_a_size <= WEIGHT_CLASSES[0]:
            case += "_" + str(MAX_DEGREES[0])
        elif filter_a_size <= WEIGHT_CLASSES[1]:
            case += "_" + str(MAX_DEGREES[1])
        else:
            case += "_" + str(MAX_DEGREES[2])

    # investigating
    # if local_recall < 5:
    #     print(f"Query {i} has recall {local_recall} and case {case}.")

    cases_list.append((case, local_recall))

    case_comparisons[case] += log[i][1]
    case_time[case] += log[i][2]

    query_cases[case] += local_recall/10
    case_counts[case] += 1
    query_recall[i] = local_recall
    total_recall += local_recall/10
avg_recall = total_recall / NQ

print()
print(f"Average total recall is {100*avg_recall:.2f}%.")
print()

# print the average recall for each case in a readable format

lst = []
for case, v in case_counts.items():
    lst.append((case, 100*query_cases[case]/v))

# sorting by case size makes more sense than what I had in mind
lst.sort(key=lambda x: case_counts[x[0]], reverse=True)

print(f"{'Case':>12}  {'Average Recall':>16} {'Case Count':>12} {'Total Comparisons':>17} {'Total Time':>11} {'Avg. Comparisons':>16} {'Avg. QPS':>10}")
for case, recall in lst:
    print(
        f"{case:>12} {recall:>16.2f}% {case_counts[case]:>12,} {case_comparisons[case]:>17,} {case_time[case]:>10.2f}s {case_comparisons[case]/case_counts[case]:>16,.2f} {case_counts[case] / case_time[case]:>10.2f}")


if not os.path.exists("logs/"):
    os.mkdir("logs/")

CSV_PATH = f"logs/cutoff{CUTOFF}_clustersize{CLUSTER_SIZE}_targetpoints{TARGET_POINTS}_tinycutoff{TINY_CUTOFF}_weightclasses{WEIGHT_CLASSES[0]}-{WEIGHT_CLASSES[1]}.csv"


# print(log[:10])

# building a list of dictionary that can be trivially converted to a pandas dataframe
log_dicts = []

for i in range(len(log)):
    log_dicts.append({
        "id": log[i][0],
        "comparisons": log[i][1],
        "time": log[i][2],
        "filter_count_a": all_filters.point_count(filters[i].a),
        "filter_count_b": all_filters.point_count(filters[i].b) if filters[i].is_and() else 0,
        "recall": query_recall[i],
        "case": cases_list[i][0],
    })

# print(log_dicts[:10])


df = pd.DataFrame(log_dicts)

if os.path.exists(CSV_PATH):
    df.to_csv(CSV_PATH, mode='a', header=False)
else:
    df.to_csv(CSV_PATH)

print(df.head())

build_config = f"""{{"cluster_size": {CLUSTER_SIZE}, 
              "T": 8,
              "cutoff": {CUTOFF},
              "max_iter": 40,
              "weight_classes": [{WEIGHT_CLASSES[0]}, {WEIGHT_CLASSES[1]}],
              "build_params": [{{"max_degree": {MAX_DEGREES[0]},
                                "limit": 500,
                                "alpha": 1.175}},
                              {{"max_degree": {MAX_DEGREES[1]},
                               "limit": 500,
                               "alpha": 1.175}},
                              {{"max_degree": {MAX_DEGREES[2]},
                               "limit": 500,
                               "alpha": 1.175}}]
            }}"""

print(build_config)

# print("----- Building 2 Stage Filtered IVF... -----")
# start = time.time()
#
# fivf2 = wp.init_2_stage_filtered_ivf_index("Euclidian", "uint8")
# fivf2.fit_from_filename(DATA_DIR + "data/yfcc100M/base.10M.u8bin.crop_nb_10000000", DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat', 1000)
#
# print(f"Time taken: {time.time() - start:.2f}s")
#
# print("----- Querying 2 Stage Filtered IVF Index... -----")
# start = time.time()
#
# NQ = 10_000
#
# X = np.fromfile(DATA_DIR + "data/yfcc100M/query.public.100K.u8bin", dtype=np.uint8)[8:].reshape((100_000, 192))
# filters = read_sparse_matrix(DATA_DIR + 'data/yfcc100M/query.metadata.public.100K.spmat')
#
# rows, cols = filters.nonzero()
# filter_dict = defaultdict(list)
#
# for row, col in zip(rows, cols):
#    filter_dict[row].append(col)
#
# filters = [wp.QueryFilter(*filters[i]) for i in filters.keys()]
# filters = [None] * len(filter_dict.keys())
# for i in filter_dict.keys():
#    filters[i] = wp.QueryFilter(*filter_dict[i])
#
# if NQ < 10:
#    print(filters[:NQ])
#
#
# N_LISTS = 100
# CUTOFF = 20_000
#
# print(f"n_lists: {N_LISTS:,}")
# print(f"cutoff: {CUTOFF:,}")
#
# neighbors, distances = fivf2.batch_search(X, filters, NQ, 10, N_LISTS, CUTOFF)
# print(neighbors.shape)
# print(neighbors[:10, :])
# print(distances[:10, :])
#
# elapsed = time.time() - start
# print(f"Time taken: {elapsed:.2f}s")
# print(f"QPS: {NQ / elapsed:.2f}s")
#
#
#
# print("----- Building IVF Index... -----")
# start = time.time()
#
# ivf = wp.init_ivf_index("Euclidian", "uint8")
# ivf.fit_from_filename(DATA_DIR + "base.1B.u8bin.crop_nb_1000000", 1000)
# ivf.fit_from_filename(DATA_DIR + "data/yfcc100M/base.10M.u8bin.crop_nb_10000000", 1000)
# ivf.print_stats()
#
#
#
# neighbors, distances = Index.batch_search_from_string(DATA_DIR + "query.public.10K.u8bin", 10000, 10, 100)
#
# query = np.fromfile(DATA_DIR + "query.public.10K.u8bin", dtype=np.uint8)[8:].reshape((10000, 128))
# print(query.shape)
# neighbors, distances = ivf.batch_search(query, 10000, 10, 100)
#
# print(neighbors.shape)
# print(neighbors[:10, :])
# print(distances[:10, :])
# Index.check_recall(DATA_DIR + "bigann-1M", neighbors, 10)
#
# print("----- Testing Filters... -----")
#
# filters = wp.csr_filters(DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat')
#
# print(filters.first_label(42))
# print(filters.match(42, 6))
# print(filters.match(42, 2))
#
# print(f"Point count of filter 23: {filters.filter_count(23):,}")
#
# print(f"Filter count of point 42: {filters.point_count(42):,}")
# print(f"Point count of filter 6: {filters.filter_count(6):,}")
#
# print("Transposing... (from python)")
#
# filters.transpose_inplace()
# filters_t = filters
#
# filters_t = filters.transpose()
#
# print("Transposed! (from python)")
#
# print(filters_t.first_label(6)) # should be 42
# print(filters_t.match(6, 42)) # should be True
# print(filters_t.match(2, 42)) # should be False
#
# print(f"Filter count of point 42: {filters_t.filter_count(42):,}")
# print(f"Point count of filter 6: {filters_t.point_count(6):,}")
#
# wp.build_vamana_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/parlayann", 64, 128, 1.2)
#
# Index = wp.load_vamana_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/parlayann", 1000000, 128)
#
#
#
# print("Filter Query")
#
# query_a = wp.QueryFilter(1)
# query_b = wp.QueryFilter(1, 2)
#
# print(query_a.is_and()) # should be False
# print(query_b.is_and()) # should be True
#
#
#
