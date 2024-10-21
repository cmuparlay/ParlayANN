import _ParlayANNpy as pann
import wrapper as wp
import time

NAME = "sift-128-euclidean"
DATA_DIR = "data/" + NAME + "/"
metric = "Euclidian"

# wp.build_vamana_index(metric, "float", DATA_DIR + NAME + "_base.fbin", DATA_DIR + "outputs/" + NAME + "_64", 64, 128, 1.15, True)
# wp.build_vamana_index(metric, "float", DATA_DIR + NAME + "_base.fbin", DATA_DIR + "outputs/" + NAME + "_40", 40, 80, 1.15, True)
# wp.build_vamana_index(metric, "float", DATA_DIR + NAME + "_base.fbin", DATA_DIR + "outputs/" + NAME + "_32", 32, 64, 1.15, True)

Index = wp.load_index(metric, "float", DATA_DIR + NAME + "_base.fbin", DATA_DIR + "outputs/" + NAME + "_64")

for Q in [14, 23] :
    for x in range(5) :
        start = time.time()
        neighbors, distances = Index.batch_search_from_string(DATA_DIR + NAME + "_query.fbin", 10, Q, True, 1000)
        end = time.time()
        print(neighbors.size)
        print("QPS: ", f'{neighbors.shape[0]/(end - start):.6f})
        
    Index.check_recall(DATA_DIR + NAME + "_query.fbin", DATA_DIR + NAME + "_groundtruth", neighbors, 10)
