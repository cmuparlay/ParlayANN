# %%
import wrapper as wp
import os, sys
import numpy as np
import pandas as pd
from time import time
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter
from math import ceil

# %%
DATA_DIR = '../data/diversity/'

if len(sys.argv) > 1:
    PARTITION_TYPE = sys.argv[1]
else:
    PARTITION_TYPE = 'vamana' # random, seller-aware, seller, vamana
    
PARTITIONS = 16 # ignored if irrelevant to type

BASE_FILENAME = os.path.join(DATA_DIR, 'base_uk.ivec')
if PARTITION_TYPE in ['seller', 'vamana']:
    INDEX_CACHE = f'index_cache/{PARTITION_TYPE}/'
    PARTITIONS = 1
    
    METADATA_FILENAME = os.path.join(INDEX_CACHE, 'base_uk_seller_csr.spmat')
elif PARTITION_TYPE in ['seller-aware', 'random']:
    INDEX_CACHE = f'index_cache/{PARTITION_TYPE}_{PARTITIONS}/'
    
    
    METADATA_FILENAME = os.path.join(INDEX_CACHE, f"{PARTITION_TYPE}_{PARTITIONS}.spmat")
else:
    raise ValueError(f"Invalid partition type {PARTITION_TYPE}")

if not os.path.exists(INDEX_CACHE):
    os.makedirs(INDEX_CACHE)

OFFER_METADATA = os.path.join(DATA_DIR, 'base_uk_metadata.parquet')


# %%
CUTOFF = 5_000
CLUSTER_SIZE = 5000
NQ = 200_000
WEIGHT_CLASSES = (100_000, 400_000)
MAX_DEGREES = (16, 32, 64)

TINY_CUTOFF = 0
TARGET_POINTS = 15_000
BEAM_WIDTHS = (1100, 1100, 1100) # (85, 85, 85)
SEARCH_LIMITS = (25_000_000, 25_000_000, 25_000_000) #(int(WEIGHT_CLASSES[0] * 0.2), int(WEIGHT_CLASSES[1] * 0.5), int(3_000_000 * 0.5))

ALPHA = 1.175

BITVECTOR_CUTOFF = 1000000000
# %%
# make our own partitioning sparse matrix
def write_csr(matrix, filename):
    with open(filename, 'wb') as f:
        np.array([matrix.shape[0], matrix.shape[1], matrix.nnz], dtype=np.int64).tofile(f)
        matrix.indptr.astype(np.int64).tofile(f)
        matrix.indices.astype(np.int32).tofile(f)
        matrix.data.astype(np.int32).tofile(f)
        
def assign_partitions(sellers, n_partitions, P):
    """
    Assigns sellers to partitions based on the number of items they have.

    Args:
        sellers (list): A list of (seller_id, seller_n_items) tuples.
            seller_n_items is assumed to be a positive integer.
        n_partitions (int): The number of partitions to assign sellers to.
            It's assumed that partitions are unbounded, such that all sellers can be assigned to a partition if the partitioning is approximately balanced.
        P (int): The maximum number of items from a given seller in a partition.

    Returns:
        dict: A dictionary mapping seller_id to a list of partition indices.
    """

    assignments = defaultdict(list)  # dictionary mapping seller to list of partitions
    partitions = defaultdict(list)  # dictionary mapping partition to list of (seller, n_items) tuples

    loads = [0] * n_partitions  # list of loads for each partition

    # we sort the sellers in descending order by the size of the chunks being assigned
    # this key resolves to 1 if
    sellers.sort(key=lambda x: x[1] / ceil(x[1] / P), reverse=True)

    for seller_id, seller_n_items in sellers:
        chunk_count = ceil(seller_n_items / P)
        chunk_size = ceil(seller_n_items / chunk_count)

        for c in range(chunk_count):
            # we assign the seller to the partition with the smallest load not already containing the seller
            min_load = min([(load, i) for i, load in enumerate(loads) if i not in assignments[seller_id]])[1]

            loads[min_load] += chunk_size  # update relevant partition's load
            partitions[min_load].append((seller_id, chunk_size))  # add seller to partition

            assignments[seller_id].append(min_load)  # add partition to those assigned to seller

    return assignments, partitions

def place_item(item_id, seller_id, assignments):
    """Returns the partition index where the item should be placed.

    Args:
        item_id (any): unique identifier for an item
        seller_id (any): unique identifier for a seller
        assignments (dict): dictionary mapping seller_id to list of partition indices
        
    Returns:
        int: the partition index where the item should be placed
    """
    if len(assignments[seller_id]) == 1:
        return assignments[seller_id][0]
    else:
        return assignments[seller_id][hash(item_id) % len(assignments[seller_id])]
        
# %%
print("Loading metadata...")
metadata = pd.read_parquet(OFFER_METADATA)
metadata.head()

# %%
if PARTITION_TYPE == 'random':
    print(f"Building random metadata csr matrix with {PARTITIONS} partitions...")
    # randomly divide offers between PARTITIONS partitions
    np.random.seed(0)
    partition = np.random.randint(0, PARTITIONS, metadata.shape[0])
    
elif PARTITION_TYPE == 'seller-aware':
    print(f"Building seller-aware metadata csr matrix with {PARTITIONS} partitions...")
    # divide offers between PARTITIONS partitions based on water-filling algorithm
    print("\tCounting occurences of each seller...")
    seller_counts = Counter(metadata['seller_id'])
    seller_counts = [(seller_id, count) for seller_id, count in seller_counts.items()]
    print('\tAssigning sellers to partitions...')
    assignments, partitions = assign_partitions(seller_counts, PARTITIONS, 3_000_000)
    
    partition = np.array([place_item(item_id, seller_id, assignments) for item_id, seller_id in zip(metadata.index, metadata['seller_id'])])
    
if PARTITION_TYPE in ['random', 'seller-aware']:
    label_csr = csr_matrix((np.ones(metadata.shape[0], dtype=np.int32), (np.arange(metadata.shape[0]), partition)), shape=(metadata.shape[0], PARTITIONS))

    write_csr(label_csr, METADATA_FILENAME)

    MAX_LABEL = PARTITIONS

    partition_counts = np.bincount(partition)
    partition_counts
    

# %%
index = wp.init_squared_ivf_index('Euclidian', 'uint8')

# %%
for i in range(3):
    index.set_build_params(wp.BuildParams(MAX_DEGREES[i], 500, ALPHA), i)
    index.set_query_params(wp.QueryParams(10, BEAM_WIDTHS[i], 1.35, SEARCH_LIMITS[i], MAX_DEGREES[i]), i)

index.set_bitvector_cutoff(BITVECTOR_CUTOFF)
index.set_materialized_joins(False)

# %%
index.fit_from_filename(BASE_FILENAME, METADATA_FILENAME, CUTOFF, CLUSTER_SIZE, INDEX_CACHE, WEIGHT_CLASSES, True)

# %%
# import queries
QUERY_FILENAME = os.path.join(DATA_DIR, 'query_uk_random_metadata.parquet')
GT_FILENAME = os.path.join(DATA_DIR, 'uk_random_GT.bin')

# %%

queries = pd.read_parquet(QUERY_FILENAME)

query_embedding_matrix = np.array(queries['embedding'].tolist(), dtype=np.uint8)

def single_label_query_filters(i, nq):
    return [wp.QueryFilter(i)] * nq

nq = query_embedding_matrix.shape[0]

# %%
# run query
start = time()
if PARTITION_TYPE == 'vamana':
    neighbors, distances = index.unsorted_batch_filter_search(query_embedding_matrix, single_label_query_filters(0, nq), nq, 1000)
elif PARTITION_TYPE in ['random', 'seller-aware']:
    # want to save a 2D array of neighbors and distances for each query, one row for each partition, so end up with a 3D array
    neighbors = []
    distances = []
    for i in range(MAX_LABEL):
        neighbors_i, distances_i = index.unsorted_batch_filter_search(query_embedding_matrix, single_label_query_filters(i, nq), nq, 1000)
        neighbors.append(neighbors_i)
        distances.append(distances_i)
    neighbors = np.stack(neighbors, axis=1)
    distances = np.stack(distances, axis=1)
    print(neighbors.shape) # (nq, MAX_LABEL, n_neighbors)

end = time()

print(f"qps: {nq / (end - start)} ({nq} queries in {end - start} seconds)")

# %%
# shrink the last dimension of neighbors and distances to remove padding
# max_nonzero = np.max(np.count_nonzero(neighbors, axis=-1))
# print(max_nonzero)
# neighbors = neighbors[:, :, :max_nonzero]
# distances = distances[:, :, :max_nonzero]


# %%   
# we make a tuple of tuples for each query, where the outer tuple is shards and the inner tuple is neighbors or distances
# print("making tuples of neighbors")
# tuple_neighbors = [tuple(tuple(n) for n in neighbors_i) for neighbors_i in neighbors]
# print("making tuples of distances")
# tuple_distances = [tuple(tuple(d) for d in distances_i) for distances_i in distances] 

# print(f"query df memory usage: {queries.memory_usage(deep=True).sum() / 1024**3} GB")

# print("assigning queries to df")
# queries[f'{PARTITION_TYPE}_{PARTITIONS}_neighbors'] = tuple_neighbors
# print(f"query df memory usage: {queries.memory_usage(deep=True).sum() / 1024**3} GB")
# print("assigning distances to df")
# queries[f'{PARTITION_TYPE}_{PARTITIONS}_distances'] = tuple_distances
# print(f"query df memory usage: {queries.memory_usage(deep=True).sum() / 1024**3} GB")

# %%
# average length of neighbors
# np.mean([np.count_nonzero(n, axis=-1) for n in neighbors])

# %%
# QUERY_OUTPUT_FILENAME = os.path.join(DATA_DIR, f'query_metadata_{PARTITION_TYPE}_{PARTITIONS}.parquet')
# # QUERY_OUTPUT_FILENAME = QUERY_FILENAME

# print(f"Writing queries to {QUERY_OUTPUT_FILENAME}")
# queries.to_parquet(QUERY_OUTPUT_FILENAME)


# %%
def read_gt(gt_file):
    """reads the ibin ground truth to a numpy array"""
    file = np.fromfile(gt_file, dtype=np.uint32)
    n = file[0]
    k = file[1]
    gt = file[2:2 + n * k].reshape(n, k)
    distances = file[2 + n * k:].reshape(n, k).view(np.float32)
    
    return gt, distances

def write_gt(gt_file, gt, distances):
    """writes the ground truth to a ibin file"""
    n, k = gt.shape
    file = np.concatenate([np.array([n, k], dtype=np.uint32), gt.flatten(), distances.flatten().view(np.uint32)])
    file.tofile(gt_file)

# we write the output from the shards to gt files
OUTPUT_DIR = os.path.join(DATA_DIR, f"output_{PARTITION_TYPE}_{PARTITIONS}")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for i in range(PARTITIONS): 
    neighbors_i = neighbors[:, i]
    distances_i = distances[:, i]
    write_gt(os.path.join(OUTPUT_DIR, f"shard_{i}.ibin"), neighbors_i, distances_i)

# # %%
# gt, gt_distances = read_gt(GT_FILENAME)

# # %%
# queries['gt'] = gt.tolist()
# queries['gt_distances'] = gt_distances.tolist()
# queries

# # %%
# queries.to_parquet(QUERY_OUTPUT_FILENAME)

