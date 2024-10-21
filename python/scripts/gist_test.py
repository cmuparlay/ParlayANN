import _ParlayANNpy as pann
import wrapper as wp
import time

NAME = "gist"
DATA_DIR = "data/" + NAME + "/"
metric = "Euclidian"

# wp.build_vamana_index(metric, "float", DATA_DIR + NAME + "_base.fbin", DATA_DIR + "outputs/" + NAME, 100, 200, 1.1, True)

Index = wp.load_index(metric, "float", DATA_DIR + NAME + "_base.fbin", DATA_DIR + "outputs/" + NAME)

for Q in [19, 36, 65] :
    for x in range(5) :
        start = time.time()
        neighbors, distances = Index.batch_search_from_string(DATA_DIR + NAME + "_query.fbin", 10, Q, True, 1000)
        end = time.time()
        print("QPS: ", neighbors.shape[0]/(end - start))
        
    Index.check_recall(DATA_DIR + NAME + "_query.fbin", DATA_DIR + NAME + "-1M", neighbors, 10)
