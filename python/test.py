import _ParlayANNpy as pann

# pann.build_vamana_uint8_euclidian_index("Euclidian", "/ssd1/data/bigann/base.1B.u8bin.crop_nb_1000000", "/ssd1/data/bigann/outputs/parlayann", 64, 128, 1.2)

Index = pann.VamanaUInt8EuclidianIndex("/ssd1/data/bigann/base.1B.u8bin.crop_nb_1000000", "/ssd1/data/bigann/outputs/parlayann", 1000000, 128)
neighbors, distances = Index.batch_search_from_string("/ssd1/data/bigann/query.public.10K.u8bin", 10000, 10, 100)
Index.check_recall("/ssd1/data/bigann/bigann-1M", neighbors, 10)