import _ParlayANNpy as pann
import wrapper as wp

print(dir(pann))

FERN_DATA_DIR = "/ssd1/anndata/text2image1B/"
AWARE_DATA_DIR = "/ssd1/data/text2image1B/"

DATA_DIR = AWARE_DATA_DIR

base_dir = DATA_DIR + "base.1B.fbin.crop_nb_1000000"
sample_dir = DATA_DIR + "query_rs_10000.fbin"
output_dir = DATA_DIR + "outputs/parlayann"
secondary_output_dir = DATA_DIR + "outputs/parlayann.secondary"
secondary_gt_dir = DATA_DIR + "outputs/parlayann.secondary.gt"


wp.build_vamana_index("mips", "float", base_dir, sample_dir, output_dir, secondary_output_dir, secondary_gt_dir, 64, 128, 1.0, False)

Index = wp.load_vamana_index("mips", "float", base_dir, sample_dir, output_dir, secondary_output_dir, secondary_gt_dir, 1000000, 128)
neighbors, distances = Index.batch_search_from_string(DATA_DIR + "query_rs_1000.fbin", 1000, 10, 100)

# Index.check_recall(DATA_DIR + "bigann-1M", neighbors, 10)