# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import wrapper as wp
import os, sys
import optuna
from datetime import datetime
import time
from collections import defaultdict
from scipy.sparse import csr_matrix
import argparse
from math import log

# %%
parser = argparse.ArgumentParser(description='Run parameter search for ParlayIVF')
parser.add_argument('--recall', type=float, default=0.75, help='Recall cutoff for the parameter search')
parser.add_argument('--dataset', type=str, default='audio', help='dataset name (audio, crawl, enron, gist, msong, sift, uqv)')
parser.add_argument('--output', type=str, default=None, help='output file name for the parameter search')
parser.add_argument('--build', action='store_true', help='run the build phase')
parser.add_argument('--search', action='store_true', help='run the search phase')
parser.add_argument('--threads', type=int, default=8, help='number of threads to use')


args = parser.parse_args()

# %%
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


nhq_properties = {
    "crawl" : {"nb" : 1989995, "d" : 300, "nq" : 10000, "dtype" : "float32", "distance" : "euclidian"},
    "enron" : {"nb" : 94987, "d" : 1369, "nq" : 200, "dtype" : "float32", "distance" : "euclidian"},
    "gist" : {"nb" : 1000000, "d" : 960, "nq" : 1000, "dtype" : "float32", "distance" : "euclidian"},
    "msong" : {"nb" : 992272, "d" : 420, "nq" : 200, "dtype" : "float32", "distance" : "euclidian"},
    "audio" : {"nb" : 53387, "d" : 192, "nq" : 200, "dtype" : "float32", "distance" : "euclidian"},
    "sift" : {"nb" : 1000000, "d" : 128, "nq" : 10000, "dtype" : "float32", "distance" : "euclidian"},
    "uqv" : {"nb" : 1000000, "d" : 256, "nq" : 10000, "dtype" : "float32", "distance" : "euclidian"},
}

class NHQ_Dataset:
    def __init__(self, name):
        self.name = name
        self.nb = nhq_properties[name]["nb"]
        self.d = nhq_properties[name]["d"]
        self.nq = nhq_properties[name]["nq"]
        self.dtype = nhq_properties[name]["dtype"]
        self.distance = nhq_properties[name]["distance"]

        self.ds_fn = f"{name}_base.fvec"
        self.qs_fn = f"{name}_query.fvec"
        self.ds_metadata_fn = f"label_{name}_base.spmat"
        self.qs_metadata_fn = f"label_{name}_query.spmat"
        self.gt_fn = f"{name}_gt.ibin"

if args.dataset in nhq_properties:
    dataset = NHQ_Dataset(args.dataset)
else:
    raise ValueError(f"Dataset {args.dataset} not found")

# %%
FERN_DATA_DIR = "/ssd1/anndata/bigann/"
AWARE_DATA_DIR = "/ssd1/data/bigann/"
AWS_DATA_DIR = "../../big-ann-benchmarks/"

DATA_DIR = AWS_DATA_DIR # incidentally, this works elsewhere if you have the repo set up

CUTOFF = 10_000
CLUSTER_SIZE = 5_000
NQ = 100_000
WEIGHT_CLASSES = (100_000, 400_000)
MAX_DEGREES = (8, 10, 12)

TINY_CUTOFF = 35000
TARGET_POINTS = 7500
BEAM_WIDTHS = (55, 55, 55)
SEARCH_LIMITS = (int(WEIGHT_CLASSES[0] * 0.2), int(WEIGHT_CLASSES[1] * 0.5), int(3_000_000 * 0.5))
MAX_ITER = 10

ALPHA = 1.175

# %%
if not os.path.exists("index_cache/"):
    os.mkdir("index_cache/")

os.environ['PARLAY_NUM_THREADS'] = str(args.threads)
# INDEX_DIR = '../../big-ann-benchmarks/data/indices/filter/parlayivf/YFCC100MDataset-10000000/parlayivf_Euclidian_uint8'
INDEX_DIR = f"index_cache/{dataset.name}/"
if not os.path.exists(INDEX_DIR):
    os.mkdir(INDEX_DIR)

# %%
# def parse_framework_output(data):
#     lines = [line.strip() for line in data.split("\n") if line.strip()]
#     entries = []
#     entries = []
#     for line in lines:
#         if line.startswith("Computing"):
#             continue
#         match = re.search(r'ParlayIVF\((.*?)\)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
#         params = match.group(1)
#         recall = match.group(2)
#         qps = match.group(3)

#         param_dict = {}
#         for param in params.split(","):
#             if "=" in param:
#                 key, val = param.split("=")
#                 key = key.strip()
#                 # Convert numbers with commas to integers
#                 if ',' in val:
#                     val = int(val.replace(',', ''))
#                 # Convert lists to tuples
#                 elif '[' in val and ']' in val:
#                     val = tuple(map(int, re.findall(r'\d+', val)))
#                 # Convert tuple strings to actual tuples
#                 elif '(' in val and ')' in val:
#                     val = tuple(map(int, re.findall(r'\d+', val)))
#                 else:
#                     val = val.strip()
#                 param_dict[key] = val
        
#         param_dict['recall'] = float(recall)
#         param_dict['qps'] = float(qps)
#         entries.append(param_dict)

#     return pd.DataFrame(entries)



# # %%
# framework_output = """Computing knn metrics
#   0: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=20,000, tiny_cutoff=0, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[128, 128, 128], search_limits=[100000, 400000, 3000000])        0.926    13284.645
# Computing knn metrics
#   1: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=15,000, tiny_cutoff=0, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[90, 90, 90], search_limits=[100000, 400000, 3000000])        0.905    16296.918
# Computing knn metrics
#   2: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=20,000, tiny_cutoff=0, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[90, 90, 90], search_limits=[100000, 400000, 3000000])        0.914    14097.419
# Computing knn metrics
#   4: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=15,000, tiny_cutoff=1,500, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[128, 128, 128], search_limits=[100000, 400000, 3000000])        0.918    15059.208
# Computing knn metrics
#   5: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=20,000, tiny_cutoff=1,000, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[90, 90, 90], search_limits=[100000, 400000, 3000000])        0.915    13754.467
# Computing knn metrics
#   7: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=20,000, tiny_cutoff=0, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[80, 80, 80], search_limits=[100000, 400000, 3000000])        0.909    14194.552
# Computing knn metrics
#   8: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=20,000, tiny_cutoff=1,000, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[100, 100, 100], search_limits=[100000, 400000, 3000000])        0.919    13741.418
# Computing knn metrics
#   9: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=20,000, tiny_cutoff=1,000, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[80, 80, 80], search_limits=[100000, 400000, 3000000])        0.910    14084.299
# Computing knn metrics
#  10: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=15,000, tiny_cutoff=0, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[100, 100, 100], search_limits=[100000, 400000, 3000000])        0.908    16085.959
# Computing knn metrics
#  11: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=15,000, tiny_cutoff=0, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[80, 80, 80], search_limits=[100000, 400000, 3000000])        0.900    16608.533
# Computing knn metrics
#  12: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=20,000, tiny_cutoff=1,000, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[128, 128, 128], search_limits=[100000, 400000, 3000000])        0.927    12812.983
# Computing knn metrics
#  14: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=15,000, tiny_cutoff=1,500, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[80, 80, 80], search_limits=[100000, 400000, 3000000])        0.902    16298.183
# Computing knn metrics
#  15: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=15,000, tiny_cutoff=1,500, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[100, 100, 100], search_limits=[100000, 400000, 3000000])        0.911    15766.889
# Computing knn metrics
#  16: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=15,000, tiny_cutoff=0, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[128, 128, 128], search_limits=[100000, 400000, 3000000])        0.916    15482.303
# Computing knn metrics
#  17: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=20,000, tiny_cutoff=0, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[100, 100, 100], search_limits=[100000, 400000, 3000000])        0.918    13811.775
# Computing knn metrics
#  18: ParlayIVF(metric=Euclidian, dtype=uint8, T=8, cluster_size=2,500, cutoff=20,000, target_points=15,000, tiny_cutoff=1,500, max_iter=40, weight_classes=(100000, 400000), max_degrees=(8, 10, 12), beam_widths=[90, 90, 90], search_limits=[100000, 400000, 3000000])        0.907    16175.226"""

# # %%
# framework_df = parse_framework_output(framework_output)
# framework_df.sort_values(by=['qps'], inplace=True)

# # %%
# framework_df

# %%
def build_with_params(max_degrees, weight_classes, cutoff, cluster_size, bitvector_cutoff, max_iter):
    dtype = dataset.dtype
    if dtype == "float32":
        dtype = "float"

    index = wp.init_squared_ivf_index('Euclidian', dtype)
    for i in range(3):
        index.set_build_params(wp.BuildParams(max_degrees[i], 200, 1.175), i)
    index.set_bitvector_cutoff(bitvector_cutoff)
    index.set_max_iter(max_iter)
    index.fit_from_filename(os.path.join(DATA_DIR, 'data', dataset.name, dataset.ds_fn), os.path.join(DATA_DIR, 'data', dataset.name, dataset.ds_metadata_fn), cutoff, cluster_size, INDEX_DIR, weight_classes, True)
    return index

# %%

def update_search_params(index, target_points, tiny_cutoff, beam_widths, search_limits):
    index.set_target_points(target_points)
    index.set_tiny_cutoff(tiny_cutoff)
    for i in range(3):
        index.set_query_params(wp.QueryParams(10, beam_widths[i], 1.35, search_limits[i], 12), i)
    return index

# %%

# GROUND_TRUTH_DIR = DATA_DIR + "data/yfcc100M/GT.public.ibin"
GROUND_TRUTH_DIR = os.path.join(DATA_DIR, 'data', dataset.name, dataset.gt_fn)

def retrieve_ground_truth(fname):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
    f = open(fname, "rb")
    f.seek(4+4)
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return I, D

# %%
I, D = retrieve_ground_truth(GROUND_TRUTH_DIR)

def recall(I, I_gt):
    I_gt = I_gt[:,:10]
    # if (len(I) and len(I_gt)):
    #     print()
    #     print("Computed neighbors:", I[0])
    #     print()
    #     print("Actual neighbors", I_gt[0])
    return np.mean([len(np.intersect1d(I[i], I_gt[i], assume_unique=True)) / min(10, len(I[i])) for i in range(len(I))])
# %%
# assumed here that the dtype is 32 bits 
X = np.fromfile(os.path.join(DATA_DIR, 'data', dataset.name, dataset.qs_fn), dtype=dataset.dtype)[2:].reshape((dataset.nq, dataset.d))

filters = read_sparse_matrix(
    os.path.join(DATA_DIR, 'data', dataset.name, dataset.qs_metadata_fn))

rows, cols = filters.nonzero()
filter_dict = defaultdict(list)

for row, col in zip(rows, cols):
    filter_dict[row].append(col)

# filters = [wp.QueryFilter(*filters[i]) for i in filters.keys()]
filters = [None] * len(filter_dict.keys())
for i in filter_dict.keys():
    filters[i] = wp.QueryFilter(*filter_dict[i])

# %%
def run_index(index, I_gt, nq, runs=4, recall_cutoff=0.9):
    best_run = (0, 0, 0)
    for i in range(runs):
        start = time.time()
        I, D = index.batch_filter_search(X, filters, nq, 10)
        end = time.time()
        dcmps = index.get_dcmps()
        index.reset()
        r = recall(I, I_gt)
        qps = nq / (end - start)
        if (r > recall_cutoff or best_run == (0, 0, 0)) and qps > best_run[1]:
            best_run = (r, qps, dcmps)
        
    return best_run

# %%

# index = build_with_params(MAX_DEGREES, WEIGHT_CLASSES, CUTOFF, CLUSTER_SIZE, 10_000, MAX_ITER)
# %%
def search_objective(trial, recall_cutoff=0.75):
    tiny_cutoff = trial.suggest_int('tiny_cutoff', 10_000, 100_000, step=1_000)
    target_points = trial.suggest_int('target_points', 5_000, 40_000, step=1_000)
    beam_width_s = trial.suggest_int('beam_width_s', 30, 130, step=5)
    beam_width_m = trial.suggest_int('beam_width_m', 30, 130, step=5)
    beam_width_l = trial.suggest_int('beam_width_l', 30, 130, step=5)
    # beam_width = trial.suggest_int('beam_width', 30, 130, step=5)
    search_limit = trial.suggest_int('search_limit', 30, 1000, step=5)

    update_search_params(index, target_points, tiny_cutoff, (beam_width_s, beam_width_m, beam_width_s), (search_limit, search_limit, search_limit))

    r, qps, dcmps = run_index(index, I, dataset.nq, runs=4, recall_cutoff=recall_cutoff)

    if r > recall_cutoff:
        return round(qps) + r
    else:
        return r
    
def build_objective(trial, recall_cutoff=0.75):
    max_degree = trial.suggest_int('max_degree', 5, 24)
    cutoff = trial.suggest_int('cutoff', 5_000, 50_000, step=500)
    cluster_size = trial.suggest_int('cluster_size', 500, 50_000, step=500)

    bitvector_cutoff = 10_000
    max_iter = 15
    weight_classes = (100_000, 400_000)

    index = build_with_params((max_degree, max_degree, max_degree), weight_classes, cutoff, cluster_size, bitvector_cutoff, max_iter)

    update_search_params(index, TARGET_POINTS, TINY_CUTOFF, BEAM_WIDTHS, SEARCH_LIMITS)

    r, qps, dcmps = run_index(index, I, dataset.nq, runs=5)

    if r > recall_cutoff:
        return round(qps) + r
    else:
        return r


    
# %%

if __name__ == "__main__":
    recall_cutoff = args.recall
    
    print(f"Running with recall cutoff {recall_cutoff}")

    if args.build:
        print("Running build phase")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: build_objective(trial, recall_cutoff), n_trials=100)
        best_params = study.best_params
        print(f"Best build parameters: {best_params}")
        index = build_with_params((best_params['max_degree'], best_params['max_degree'], best_params['max_degree']), WEIGHT_CLASSES, best_params['cutoff'], best_params['cluster_size'], 10_000, 15)
    else:
        print("Skipping build phase")
        index = build_with_params(MAX_DEGREES, WEIGHT_CLASSES, CUTOFF, CLUSTER_SIZE, 10_000, 15)

    if args.search:
        print("Running search phase")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: search_objective(trial, recall_cutoff), n_trials=500)
        best_params = study.best_params
        print(f"Best search parameters: {best_params}")
# %%
# def build_objective(trial):
#     max_degrees = (trial.suggest_int('max_degree_s', 5, 8), trial.suggest_int('max_degree_m', 6, 10), trial.suggest_int('max_degree_l', 6, 12))
#     weight_classes = (trial.suggest_int('weight_classes_s', 60_000, 300_000, step=10_000), trial.suggest_int('weight_classes_m', 300_000, 800_000, step=10_000))
#     cutoff = trial.suggest_int('cutoff', 7_000, 20_000, step=500)
#     cluster_size = trial.suggest_int('cluster_size', 1_000, 10_000, step=500)
#     bitvector_cutoff = trial.suggest_int('bitvector_cutoff', 0, 20_000, step=1_000)

#     index = build_with_params(max_degrees, weight_classes, cutoff, cluster_size, bitvector_cutoff, 10)

#     update_search_params(index, TARGET_POINTS, TINY_CUTOFF, BEAM_WIDTHS, SEARCH_LIMITS)

#     r, qps, dcmps = run_index(index, I, NQ, runs=5)

#     if r > 0.9:
#         return dcmps
#     else:
#         return 0

# # %%
# study = optuna.create_study(direction='minimize')

# study.enqueue_trial({'max_degree_s': 8, 'max_degree_m': 10, 'max_degree_l': 12, 'weight_classes_s': 100_000, 'weight_classes_m': 400_000, 'cutoff': 10_000, 'cluster_size': 5_000, 'bitvector_cutoff': 10_000, 'max_iter': 10})

# study.optimize(build_objective, n_trials=1000)
# %%
