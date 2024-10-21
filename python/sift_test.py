import _ParlayANNpy as pann
import wrapper as wp
import time

NAME = "sift-128-euclidean"
DATA_DIR = "data/" + NAME + "/"
metric = "Euclidian"

# wp.build_vamana_index(metric, "float", DATA_DIR + "base.fbin", DATA_DIR + "graphs/graph_" + "64_1.1x", 64, 128, 1.1, True)

Index = wp.load_index(metric, "float", DATA_DIR + "base.fbin", DATA_DIR + "graphs/graph_" + "64_1.1x")

for Q in [16, 25] :
    for x in range(3) :
        start = time.time()
        neighbors, distances = Index.batch_search_from_string(DATA_DIR + "query.fbin", 10, Q, True, 1000)
        end = time.time()
        print("QPS: ", neighbors.shape[0]/(end - start))
        
    Index.check_recall(DATA_DIR + "query.fbin", DATA_DIR + "groundtruth", neighbors, 10)
