import _ParlayANNpy as pann
import wrapper as wp
import numpy as np

print(dir(pann))

FERN_DATA_DIR = "/ssd1/anndata/bigann/"
AWARE_DATA_DIR = "/ssd1/data/bigann/"

DATA_DIR = FERN_DATA_DIR

print("!!! FILTERED IVF !!!")
fivf = wp.init_filtered_ivf_index("Euclidian", "uint8")
fivf.fit_from_filename(DATA_DIR + "data/yfcc100M/base.10M.u8bin.crop_nb_10000000", DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat', 1000)

print("----- Building IVF Index... -----")

ivf = wp.init_ivf_index("Euclidian", "uint8")
# ivf.fit_from_filename(DATA_DIR + "base.1B.u8bin.crop_nb_1000000", 1000)
ivf.fit_from_filename(DATA_DIR + "data/yfcc100M/base.10M.u8bin.crop_nb_10000000", 1000)
ivf.print_stats()

# # neighbors, distances = Index.batch_search_from_string(DATA_DIR + "query.public.10K.u8bin", 10000, 10, 100)

# query = np.fromfile(DATA_DIR + "query.public.10K.u8bin", dtype=np.uint8)[8:].reshape((10000, 128))
# print(query.shape)
# neighbors, distances = ivf.batch_search(query, 10000, 10, 100)

# print(neighbors.shape)
# print(neighbors[:10, :])
# print(distances[:10, :])
# # Index.check_recall(DATA_DIR + "bigann-1M", neighbors, 10)

print("----- Testing Filters... -----")

filters = wp.csr_filters(DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat')

print(filters.first_label(42))
print(filters.match(42, 6))
print(filters.match(42, 2))

print(f"Filter count of point 42: {filters.point_count(42)}")
print(f"Point count of filter 6: {filters.filter_count(6)}")

print("Transposing... (from python)")

filters.transpose_inplace()

filters_t = filters

print("Transposed! (from python)")

print(filters_t.first_label(6)) # should be 42
print(filters_t.match(6, 42)) # should be True
print(filters_t.match(2, 42)) # should be False

print(f"Filter count of point 42: {filters_t.filter_count(42)}")
print(f"Point count of filter 6: {filters_t.point_count(6)}")

# wp.build_vamana_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/parlayann", 64, 128, 1.2)

# Index = wp.load_vamana_index("Euclidian", "uint8", DATA_DIR + "base.1B.u8bin.crop_nb_1000000", DATA_DIR + "outputs/parlayann", 1000000, 128)



# print("Filter Query")

# query_a = wp.QueryFilter(1)
# query_b = wp.QueryFilter(1, 2)

# print(query_a.is_and()) # should be False
# print(query_b.is_and()) # should be True



