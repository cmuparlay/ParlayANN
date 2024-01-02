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

def compute_recall(gt_neighbors, results, top_k):
    recall = 0
    for i in range(len(gt_neighbors)): # for each query
        gt = set(gt_neighbors[i])
        res = set(results[i][:top_k])
        recall += len(gt.intersection(res)) / len(gt)
    return recall / len(gt_neighbors) # average recall per query


data_dir = "/ssd1/anndata/ann-benchmarks/"

THREADS = 144
os.environ["PARLAY_NUM_THREADS"] = str(THREADS)

# FILTER_WIDTHS = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
FILTER_WIDTHS = [0.01, 0.1, 0.5]

for dataset_name in ["glove-100-angular", "sift-128-euclidean"]:
    data_path = os.path.join(data_dir, f"{dataset_name}.hdf5")
    filter_path = os.path.join(data_dir, f"{dataset_name}_filters.npy")

    data = parse_ann_benchmarks_hdf5(data_path)[0]
    filter_values = np.load(filter_path)

    # filter_values = np.array(list(range(data.shape[0], 0, -1)), dtype=np.float32)
    # filter_values = np.array(list(range(data.shape[0])), dtype=np.float32)
    # filter_values.sort(reverse=True)

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
    index = constructor(data, filter_values, 1_000)
    index_build_end = time.time()
    index_build_time = index_build_end - index_build_start
    print(f"index build time: {index_build_time:.3f}s")

    # run experiment
    top_k = 10
    output_file = f"results/{dataset_name}_experiment.txt"

    # with open(output_file, "a") as f:
    #     f.write("filter_width,method,recall,average_time,qps,threads\n")

    # only write header if file doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, "a") as f:
            f.write("filter_width,method,recall,average_time,qps,threads\n")


    for filter_width in FILTER_WIDTHS:
        print(f"filter width: {filter_width}")
        run_results = defaultdict(list)
        raw_filters = np.random.uniform(filter_width / 2, 1 - filter_width / 2, size=len(queries))

        filters = np.array([(x - filter_width / 2, x + filter_width / 2) for x in raw_filters])

        # filters = np.array([((x - filter_width / 2) * data.shape[0], (x + filter_width / 2) * data.shape[0]) for x in raw_filters])

        

        print("prefilter querying")
        prefiltering_start = time.time()
        prefilter_results = prefilter_index.batch_query(queries, filters, queries.shape[0], top_k)
        prefiltering_end = time.time()
        prefiltering_time = prefiltering_end - prefiltering_start
        print(f"prefiltering time: {prefiltering_time:.3f}s")

        # print(prefilter_results[0][:10])
        # print(prefilter_results[1][:10])


        print("index querying")
        start = time.time()
        index_results = index.batch_filter_search(queries, filters, queries.shape[0], top_k)
        end = time.time()
        index_time = end - start
        print(f"index time: {index_time:.3f}s")

        # print(index_results[0][:10])
        # print(index_results[1][:10])

        RAND_QUERY = 9878
        print(f"filter: {filters[RAND_QUERY]}")
        print(f"prefilter results: {[filter_values[x] for x in prefilter_results[0][RAND_QUERY]]}")
        print(f"index results: {[filter_values[x] for x in index_results[0][RAND_QUERY]]}")

        max_distance_out_of_range = []
        for i in range(len(queries)):
            filter_range = filters[i]
            knn_filter_values = [filter_values[x] for x in index_results[0][i] if x != -1]
            min_filter_value = min(knn_filter_values)
            max_filter_value = max(knn_filter_values)

            max_distance_out_of_range.append(max(filter_range[0] - min_filter_value, max_filter_value - filter_range[1]))

        print(f"farthest out of range prefilter result: {max(max_distance_out_of_range)}")

        # compute recall
        index_recall = compute_recall(prefilter_results[0], index_results[0], top_k)
        print(f"index recall: {index_recall*100:.2f}%")

        # compute average time
        index_average_time = index_time / queries.shape[0]
        prefilter_average_time = prefiltering_time / queries.shape[0]

        # compute qps
        index_qps = queries.shape[0] / index_time
        prefilter_qps = queries.shape[0] / prefiltering_time

        # write results
        with open(output_file, "a") as f:
            f.write(f"{filter_width},index,{index_recall},{index_average_time},{index_qps},{THREADS}\n")
            f.write(f"{filter_width},prefilter,{index_recall},{prefilter_average_time},{prefilter_qps},{THREADS}\n")




