import os
import _ParlayANNpy as pann
import numpy as np
import wrapper as wp
import time
from collections import defaultdict

import h5py

def parse_ann_benchmarks_hdf5(data_path):
    with h5py.File(data_path, "r") as file:
        gt_neighbors = np.array(file["neighbors"])
        queries = np.array(file["test"])
        data = np.array(file["train"])

        return data, queries, gt_neighbors


def pareto_front(x, y):
    sorted_indices = sorted(range(len(y)), key=lambda k: -y[k])
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    pareto_front_x = [x_sorted[0]]
    pareto_front_y = [y_sorted[0]]

    for i in range(1, len(x_sorted)):
        if x_sorted[i] > pareto_front_x[-1]:
            pareto_front_x.append(x_sorted[i])
            pareto_front_y.append(y_sorted[i])

    return pareto_front_x, pareto_front_y


data_dir = "/ssd1/anndata/ann-benchmarks/"

for dataset_name in ["glove-100-angular", "sift-128-euclidean"]:
    data_path = os.path.join(data_dir, f"{dataset_name}.hdf5")
    filter_path = os.path.join(data_dir, f"{dataset_name}_filters.npy")

    data = parse_ann_benchmarks_hdf5(data_path)[0]
    filter_values = np.load(filter_path)
    queries = parse_ann_benchmarks_hdf5(data_path)[1]

    if 'angular' in dataset_name:
        metric = "mips"
        # normalize data
        data = data / np.linalg.norm(data, axis=-1)[:, np.newaxis]
    else:
        metric = "Euclidian"

    # do prefiltering for ground truth
    print("prefiltering index build")
    prefilter_constructor = wp.prefilter_index_constructor(metric, 'float')
    prefilter_build_start = time.time()
    prefilter_index = prefilter_constructor(data, filter_values)
    prefilter_build_end = time.time()
    prefilter_build_time = prefilter_build_end - prefilter_build_start
    print(f"prefiltering index build time: {prefilter_build_time:.3f}s")

    # build index
    constructor = wp.flat_range_filter_index_constructor(metric, 'float')
    print("building index")
    index_build_start = time.time()
    index = constructor(data, filter_values)
    index_build_end = time.time()
    index_build_time = index_build_end - index_build_start
    print(f"index build time: {index_build_time:.3f}s")

    # run experiment
    top_k = 10
    output_file = f"results/{dataset_name}_experiment.txt"

    with open(output_file, "a") as f:
        f.write("filter_width,method,recall,average_time\n")

    for filter_width in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
        print(f"filter width: {filter_width}")
        run_results = defaultdict(list)
        raw_filters = np.random.uniform(filter_width / 2, 1 - filter_width / 2, size=len(queries))

        filters = np.array([(x - filter_width / 2, x + filter_width / 2) for x in raw_filters])

        print("prefilter querying")
        prefiltering_start = time.time()
        prefilter_results = prefilter_index.batch_query(queries, filters, queries.shape[0], top_k)
        prefiltering_end = time.time()
        prefiltering_time = prefiltering_end - prefiltering_start
        print(f"prefiltering time: {prefiltering_time:.3f}s")

        print("index querying")
        start = time.time()
        




