import _ParlayANNpy as pann
import wrapper as wp
import time

NAME = "fashion-mnist-784-euclidean"
DATA_DIR = "data/" + NAME + "/"
metric = "Euclidian"

wp.build_vamana_index(metric, "float", DATA_DIR + NAME + "_base.fbin", DATA_DIR + "outputs/" + NAME, 32, 64, 1.15, True)

Index = wp.load_index(metric, "float", DATA_DIR + NAME + "_base.fbin", DATA_DIR + "outputs/" + NAME)

for (Q,limit) in [(10,10), (10,13), (10,16)] :
    for x in range(5) :
        start = time.time()
        neighbors, distances = Index.batch_search_from_string(DATA_DIR + NAME + "_query.fbin", 10, Q, True, limit)
        end = time.time()
        print("QPS: ", neighbors.shape[0]/(end - start))
        
    Index.check_recall(DATA_DIR + NAME + "_query.fbin", DATA_DIR + NAME + "_groundtruth", neighbors, 10)
