import _ParlayANNpy as pann
import wrapper as wp

print(dir(pann))

FERN_DATA_DIR = "/ssd1/anndata/bigann/"
AWARE_DATA_DIR = "/ssd1/data/bigann/"

DATA_DIR = FERN_DATA_DIR

wp.build_vamana_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/parlayann", 64, 128, 1.2)

Index = wp.load_vamana_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/parlayann", 1000000, 128)
neighbors, distances = Index.batch_search_from_string(DATA_DIR + "query.public.10K.u8bin", 10000, 10, 100)

# Index.check_recall(DATA_DIR + "bigann-1M", neighbors, 10)