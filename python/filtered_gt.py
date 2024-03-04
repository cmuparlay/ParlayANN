import pandas as pd
import numpy as np
from datetime import datetime
import time
from collections import defaultdict

import os
import _ParlayANNpy as pann
import numpy as np
import wrapper as wp
from scipy.sparse import csr_matrix
from tqdm import tqdm

FERN_DATA_DIR = "/ssd1/anndata/"

SIZE_STR = "35M"

DATA_DIR = FERN_DATA_DIR
POINTS_PATH = DATA_DIR + f"wiki_sentence/base.{SIZE_STR}.4k.fbin"
FILTER_PATH = DATA_DIR + f'wiki_sentence/base.{SIZE_STR}.4k.metadata.spmat'

QUERY_FILTER_DIR = DATA_DIR + "wiki_sentence/msr_queries/"
QUERY_POINTS_PATH = DATA_DIR + "wiki_sentence/msr_queries/wikipedia_simple_query_embeddings.bin"

def numpy_to_gt_ibin(ids, distances, output_fname):
    with open(output_fname, "wb") as out_f:
        # Write the header (n and d) to the output file
        np.array([ids.shape[0], ids.shape[1]], dtype="uint32").tofile(out_f)
        ids.astype("int32").tofile(out_f)
        distances.astype("float32").tofile(out_f)

base_dataset = wp.FilteredDataset(POINTS_PATH, FILTER_PATH)

print("base dataset loaded")

# for each query dataset, load it and compute the filtered ground truth

# get all filter files
query_files = [f for f in os.listdir(QUERY_FILTER_DIR) if f.endswith(".spmat") and "triple" not in f and ("random" not in f or "double" in f)]

for query_file in tqdm(query_files):
    query_dataset = wp.FilteredDataset(QUERY_POINTS_PATH, QUERY_FILTER_DIR + query_file)

    print("loaded query dataset", query_file)

    I, D = base_dataset.filtered_groundtruth(query_dataset, 100)

    gt_filename = QUERY_FILTER_DIR + query_file[:-6] + f"_gt.{SIZE_STR}.ibin"

    numpy_to_gt_ibin(I, D, gt_filename)
    print("done with", query_file)

print("done with all queries")