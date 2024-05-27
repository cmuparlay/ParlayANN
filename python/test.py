import _ParlayANNpy as pann
import wrapper as wp

print(dir(pann))

# FERN_DATA_DIR = "/ssd1/anndata/bigann/"
# AWARE_DATA_DIR = "/ssd1/data/bigann/"

# DATA_DIR = AWARE_DATA_DIR

# wp.build_pynndescent_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/pynn", 40, 10, 100, 1.2, .05)

# Index = wp.load_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/pynn", 1000000, 128)
# neighbors, distances = Index.batch_search_from_string(DATA_DIR + "query.public.10K.u8bin", 10000, 10, 100)

# Index.check_recall(DATA_DIR + "bigann-1M", neighbors, 10)

DATA_DIR="/ssd1/data/FB_ssnpp/"

# wp.build_vamana_index("Euclidian", "uint8", DATA_DIR+"FB_ssnpp_database.u8bin.crop_nb_1000000", DATA_DIR+"python_output", 64, 128, 1.0, False)

Index = wp.load_index("Euclidian", "uint8", DATA_DIR+"FB_ssnpp_database.u8bin.crop_nb_1000000", DATA_DIR+"python_output", 1000000, 256)

(lims, (neighbors, distances)) = Index.batch_range_search_from_string(DATA_DIR + "FB_ssnpp_public_queries.u8bin", 100000, 96237, 100)

Index.check_range_recall(DATA_DIR + "ssnpp-1M", lims)



