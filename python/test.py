import _ParlayANNpy as pann
import wrapper as wp

print(dir(pann))

FERN_DATA_DIR = "/ssd1/anndata/bigann/"
AWARE_DATA_DIR = "/ssd1/data/bigann/"

DATA_DIR = FERN_DATA_DIR

print("Testing pynndescent...")

wp.build_pynndescent_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/pynn", 40, 10, 100, 1.2, .05)

Index = wp.load_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/pynn")
neighbors, distances = Index.batch_search_from_string(DATA_DIR + "query.public.10K.u8bin", 10, 10, True, 10000)

Index.check_recall(DATA_DIR + "query.public.10K.u8bin", DATA_DIR + "bigann-1M", neighbors, 10)

print("Testing vamana...")

wp.build_vamana_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/vamana", 40, 100, 1.2, False)

Index = wp.load_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/vamana")
neighbors, distances = Index.batch_search_from_string(DATA_DIR + "query.public.10K.u8bin", 10, 10, True, 10000)

Index.check_recall(DATA_DIR + "query.public.10K.u8bin", DATA_DIR + "bigann-1M", neighbors, 10)

print("Testing hcnng...")

wp.build_hcnng_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/hcnng", 40, 20, 1000)

Index = wp.load_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/hcnng")
neighbors, distances = Index.batch_search_from_string(DATA_DIR + "query.public.10K.u8bin", 10, 10, True, 10000)

Index.check_recall(DATA_DIR + "query.public.10K.u8bin", DATA_DIR + "bigann-1M", neighbors, 10)