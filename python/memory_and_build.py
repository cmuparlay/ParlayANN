import pandas as pd
from datetime import datetime
import time
from collections import defaultdict

import os
import _ParlayANNpy as pann
import numpy as np
import wrapper as wp
from scipy.sparse import csr_matrix
import yaml
import psutil
from itertools import product
import gc
import sys


DATA_DIR = "../../big-ann-benchmarks/data/"

def mmap_sparse_matrix_fields(fname):
    """ mmap the fields of a CSR matrix without instanciating it """
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
    ofs = sizes.nbytes
    indptr = np.memmap(fname, dtype='int64', mode='r',
                       offset=ofs, shape=nrow + 1)
    ofs += indptr.nbytes
    indices = np.memmap(fname, dtype='int32', mode='r', offset=ofs, shape=nnz)
    ofs += indices.nbytes
    data = np.memmap(fname, dtype='float32', mode='r', offset=ofs, shape=nnz)
    return data, indices, indptr, ncol


def read_sparse_matrix_fields(fname):
    """ read the fields of a CSR matrix without instanciating it """
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype='int64', count=nrow + 1)
        assert nnz == indptr[-1]
        indices = np.fromfile(f, dtype='int32', count=nnz)
        assert np.all(indices >= 0) and np.all(indices < ncol)
        data = np.fromfile(f, dtype='float32', count=nnz)
        return data, indices, indptr, ncol


def read_sparse_matrix(fname, do_mmap=False):
    """ read a CSR matrix in spmat format, optionally mmapping it instead """
    if not do_mmap:
        data, indices, indptr, ncol = read_sparse_matrix_fields(fname)
    else:
        data, indices, indptr, ncol = mmap_sparse_matrix_fields(fname)

    return csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, ncol))

vector_extensions = {
    'yfcc-10M': "yfcc100M/base.10M.u8bin.crop_nb_10000000",
    'wiki_sentence': "wiki_sentence/base.35M.4k.fbin",
    'uqv': "uqv/uqv_base.fvec",
    'audio': "audio/audio_base.fvec",
    'crawl': "crawl/crawl_base.fvec",
    'sift': "sift/sift_base.fvec",
    'gist': "gist/gist_base.fvec",
    'msong': "msong/msong_base.fvec"
}

metadata_extensions = {
    'yfcc-10M': "yfcc100M/base.metadata.10M.spmat",
    'wiki_sentence': "wiki_sentence/base.35M.4k.metadata.spmat",
    'uqv': "uqv/label_uqv_base.spmat",
    'audio': "audio/label_audio_base.spmat",
    'crawl': "crawl/label_crawl_base.spmat",
    'sift': "sift/label_sift_base.spmat",
    'gist': "gist/label_gist_base.spmat",
    'msong': "msong/label_msong_base.spmat"
}

query_vector_extensions = {
    'yfcc-10M': "yfcc100M/query.public.100K.u8bin",
    "wiki_sentence": "wiki_sentence/wikipedia_simple_query_embeddings.bin",
    'uqv': "uqv/uqv_query.fvec",
    'audio': "audio/audio_query.fvec",
    'crawl': "crawl/crawl_query.fvec",
    'sift': "sift/sift_query.fvec",
    'gist': "gist/gist_query.fvec",
    'msong': "msong/msong_query.fvec"
}

query_metadata_extensions = {
    'yfcc-10M': "yfcc100M/query.metadata.public.100K.spmat",
    "wiki_sentence": "wiki_sentence/wikipedia_simple_query_labels_numeric_double_common.spmat",
    'uqv': "uqv/label_uqv_query.spmat",
    'audio': "audio/label_audio_query.spmat",
    'crawl': "crawl/label_crawl_query.spmat",
    'sift': "sift/label_sift_query.spmat",
    'gist': "gist/label_gist_query.spmat",
    'msong': "msong/label_msong_query.spmat"
}

types = {
    'yfcc-10M': ('Euclidian', 'uint8'),
    'wiki_sentence': ('mips', 'float'),
    'uqv': ('mips', 'float'),
    'audio': ('mips', 'float'),
    'crawl': ('mips', 'float'),
    'sift': ('mips', 'float'),
    'gist': ('mips', 'float'),
    'msong': ('mips', 'float')
}

with open('../../big-ann-benchmarks/neurips23/filter/parlayivf/config.yaml') as f:
    config = yaml.unsafe_load(f)

def get_build_config_dict(dataset, index_name):
    configs = config[dataset][index_name]['run-groups']['base']['args']
    if type(configs) is dict:
        return configs
    elif type(configs) is list:
        return configs[0]
    elif type(configs) is str:
        configs = yaml.unsafe_load(configs)
        if type(configs) is dict:
            return configs
        elif type(configs) is list:
            return configs[0]
        else:
            raise Exception("Invalid config type " + str(type(configs)))
    else:
        raise Exception("Invalid config type " + str(type(configs)))
        

def build_and_measure(dataset, index_name):
    """builds a ParlayIVF index and measures the time it takes + footprint"""
    if not os.path.exists("tmp_index_cache/"):
        os.makedirs("tmp_index_cache/")

    build_config = get_build_config_dict(dataset, index_name)

    index = wp.init_squared_ivf_index(*(types[dataset]))

    index.set_max_iter(build_config['max_iter'])
    index.set_bitvector_cutoff(build_config['bitvector_cutoff'])
    index.set_materialized_joins(bool(build_config['materialize_joins']))

    for i, bp in enumerate(build_config['build_params']):
        bp = wp.BuildParams(bp['max_degree'], bp['limit'], bp['alpha'])
        index.set_build_params(bp, i)

    index.set_materialized_join_cutoff(build_config['join_cutoff'])

    start = time.time()

    index.fit_from_filename(DATA_DIR + vector_extensions[dataset], DATA_DIR + metadata_extensions[dataset], build_config['cutoff'], build_config['cluster_size'], "tmp_index_cache/", build_config['weight_classes'], False)

    build_time = time.time() - start
    footprint = index.footprint()

    # getting the memory usage of the whole process
    process = psutil.Process(os.getpid())
    total_memory = process.memory_info().rss

    # getting the cumulative size of all generated index files
    index_files = os.listdir("tmp_index_cache/")
    index_files_size = sum(os.path.getsize("tmp_index_cache/" + file) for file in index_files)

    return build_time, footprint, total_memory, index_files_size

def get_dataset_size(dataset):
    """the size of the vectors + metadata files in bytes"""
    vector_size = os.path.getsize(DATA_DIR + vector_extensions[dataset])
    metadata_size = os.path.getsize(DATA_DIR + metadata_extensions[dataset])
    return vector_size + metadata_size

# ABLATIONS = ['parlayivf-no-bitvector', 'parlayivf']

# DATASETS = ['audio']

RESULT_FILE = 'footprint_build_results.csv'

if not os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, 'w') as f:
        f.write('dataset,index_name,build_time,footprint,total_memory,index_files_size,dataset_size, run_timestamp\n')

# results = pd.read_csv(RESULT_FILE)

if __name__ == '__main__':
    index = sys.argv[1]
    dataset = sys.argv[2]

    build_time, footprint, total_memory, index_files_size = build_and_measure(dataset, index)
    with open(RESULT_FILE, 'a') as f:
        f.write(f"{dataset},{index},{build_time},{footprint},{total_memory},{index_files_size},{get_dataset_size(dataset)},{datetime.now()}\n")
    print(f"Built {index} on {dataset} in {build_time} seconds, with a footprint of {footprint} bytes, memory usage of {total_memory} bytes and index files size of {index_files_size} bytes")

    # delete all files in the tmp cache
    for file in os.listdir("tmp_index_cache/"):
        os.remove("tmp_index_cache/" + file)

    # run gc
    # gc.collect()
    # time.sleep(1)


