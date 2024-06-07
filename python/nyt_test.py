import _ParlayANNpy as pann
import wrapper as wp
import time

NAME = "nytimes-256-angular"
DATA_DIR = "data/" + NAME + "/"

# wp.build_vamana_index("mips", "float", DATA_DIR + NAME + "_base.fbin", DATA_DIR + "outputs/" + NAME, 150, 300, .85, True)

Index = wp.load_index("mips", "float", DATA_DIR + NAME + "_base.fbin", DATA_DIR + "outputs/" + NAME)

for Q in [17, 60] :
    for x in range(5) :
        start = time.time()
        neighbors, distances = Index.batch_search_from_string(DATA_DIR + NAME + "_query.fbin", 10, Q, True, 1000)
        end = time.time()
        print("QPS: ", neighbors.shape[0]/(end - start))
        
    Index.check_recall(DATA_DIR + NAME + "_query.fbin", DATA_DIR + NAME + "_groundtruth", neighbors, 10)
