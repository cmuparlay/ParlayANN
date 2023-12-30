import os
import _ParlayANNpy as pann
import numpy as np
import wrapper as wp
import time

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

    if 'angular' in dataset_name:
        metric = "mips"
        # normalize data
        data = data / np.linalg.norm(data, axis=-1)[:, np.newaxis]
    else:
        metric = "euclidean"

    # build index
    constructor = wp.flat_range_filter_index_constructor(metric, 'float')
    print("building index")
    index = constructor(data, filter_values)
    print("index built")
    # run experiment

