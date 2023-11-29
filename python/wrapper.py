from _ParlayANNpy import *

def build_vamana_index(metric, dtype, data_dir, index_dir, R, L, alpha, two_pass):
    if metric == 'Euclidian':
        if dtype == 'uint8':
            build_vamana_uint8_euclidian_index(metric, data_dir, index_dir, R, L, alpha, two_pass)
        elif dtype == 'int8':
            build_vamana_int8_euclidian_index(metric, data_dir, index_dir, R, L, alpha, two_pass)
        elif dtype == 'float':
            build_vamana_float_euclidian_index(metric, data_dir, index_dir, R, L, alpha, two_pass)
        else:
            raise Exception('Invalid data type ' + dtype)
    elif metric == 'mips':
        if dtype == 'uint8':
            build_vamana_uint8_mips_index(metric, data_dir, index_dir, R, L, alpha, two_pass)
        elif dtype == 'int8':
            build_vamana_int8_mips_index(metric, data_dir, index_dir, R, L, alpha, two_pass)
        elif dtype == 'float':
            build_vamana_float_mips_index(metric, data_dir, index_dir, R, L, alpha, two_pass)
        else:
            raise Exception('Invalid data type ' + dtype)
    else:
        raise Exception('Invalid metric ' + metric)
    


def build_hcnng_index(metric, dtype, data_dir, index_dir, mst_deg, num_clusters, cluster_size):
    if metric == 'Euclidian':
        if dtype == 'uint8':
            build_hcnng_uint8_euclidian_index(metric, data_dir, index_dir, mst_deg, num_clusters, cluster_size)
        elif dtype == 'int8':
            build_hcnng_int8_euclidian_index(metric, data_dir, index_dir, mst_deg, num_clusters, cluster_size)
        elif dtype == 'float':
            build_hcnng_float_euclidian_index(metric, data_dir, index_dir, mst_deg, num_clusters, cluster_size)
        else:
            raise Exception('Invalid data type ' + dtype)
    elif metric == 'mips':
        if dtype == 'uint8':
            build_hcnng_uint8_mips_index(metric, data_dir, index_dir, mst_deg, num_clusters, cluster_size)
        elif dtype == 'int8':
            build_hcnng_int8_mips_index(metric, data_dir, index_dir, mst_deg, num_clusters, cluster_size)
        elif dtype == 'float':
            build_hcnng_float_mips_index(metric, data_dir, index_dir, mst_deg, num_clusters, cluster_size)
        else:
            raise Exception('Invalid data type ' + dtype)
    else:
        raise Exception('Invalid metric ' + metric)


def build_pynndescent_index(metric, dtype, data_dir, index_dir, max_deg, num_clusters, cluster_size, alpha, delta):
    if metric == 'Euclidian':
        if dtype == 'uint8':
            build_pynndescent_uint8_euclidian_index(metric, data_dir, index_dir, max_deg, num_clusters, cluster_size, alpha, delta)
        elif dtype == 'int8':
            build_pynndescent_int8_euclidian_index(metric, data_dir, index_dir, max_deg, num_clusters, cluster_size, alpha, delta)
        elif dtype == 'float':
            build_pynndescent_float_euclidian_index(metric, data_dir, index_dir, max_deg, num_clusters, cluster_size, alpha, delta)
        else:
            raise Exception('Invalid data type ' + dtype)
    elif metric == 'mips':
        if dtype == 'uint8':
            build_pynndescent_uint8_mips_index(metric, data_dir, index_dir, max_deg, num_clusters, cluster_size, alpha, delta)
        elif dtype == 'int8':
            build_pynndescent_int8_mips_index(metric, data_dir, index_dir, max_deg, num_clusters, cluster_size, alpha, delta)
        elif dtype == 'float':
            build_pynndescent_float_mips_index(metric, data_dir, index_dir, max_deg, num_clusters, cluster_size, alpha, delta)
        else:
            raise Exception('Invalid data type ' + dtype)
    else:
        raise Exception('Invalid metric ' + metric)


def build_hnsw_index(metric, dtype, data_dir, index_dir, R, L, alpha, two_pass):
    if metric == 'Euclidian':
        if dtype == 'uint8':
            build_hnsw_uint8_euclidian_index(metric, data_dir, index_dir, R, L, alpha, two_pass)
        elif dtype == 'int8':
            build_hnsw_int8_euclidian_index(metric, data_dir, index_dir, R, L, alpha, two_pass)
        elif dtype == 'float':
            build_hnsw_float_euclidian_index(metric, data_dir, index_dir, R, L, alpha, two_pass)
        else:
            raise Exception('Invalid data type ' + dtype)
    elif metric == 'mips':
        if dtype == 'uint8':
            build_hnsw_uint8_mips_index(metric, data_dir, index_dir, R, L, alpha, two_pass)
        elif dtype == 'int8':
            build_hnsw_int8_mips_index(metric, data_dir, index_dir, R, L, alpha, two_pass)
        elif dtype == 'float':
            build_hnsw_float_mips_index(metric, data_dir, index_dir, R, L, alpha, two_pass)
        else:
            raise Exception('Invalid data type ' + dtype)
    else:
        raise Exception('Invalid metric ' + metric)

        
def load_index(metric, dtype, data_dir, index_dir, n, d, hnsw=False):
    if metric == 'Euclidian':
        if dtype == 'uint8':
            return UInt8EuclidianIndex(data_dir, index_dir, n, d, hnsw)
        elif dtype == 'int8':
            return Int8EuclidianIndex(data_dir, index_dir, n, d, hnsw)
        elif dtype == 'float':
            return FloatEuclidianIndex(data_dir, index_dir, n, d, hnsw)
        else:
            raise Exception('Invalid data type')
    elif metric == 'mips':
        if dtype == 'uint8':
            return UInt8MipsIndex(data_dir, index_dir, n, d, hnsw)
        elif dtype == 'int8':
            return Int8MipsIndex(data_dir, index_dir, n, d, hnsw)
        elif dtype == 'float':
            return FloatMipsIndex(data_dir, index_dir, n, d, hnsw)
        else:
            raise Exception('Invalid data type')
    else:
        raise Exception('Invalid metric')
