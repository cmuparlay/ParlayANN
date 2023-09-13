from _ParlayANNpy import *

def build_vamana_index(metric, dtype, data_dir, index_dir, R, L, alpha):
    if metric == 'Euclidian':
        if dtype == 'uint8':
            build_vamana_uint8_euclidian_index(metric, data_dir, index_dir, R, L, alpha)
        elif dtype == 'int8':
            build_vamana_int8_euclidian_index(metric, data_dir, index_dir, R, L, alpha)
        elif dtype == 'float':
            build_vamana_float_euclidian_index(metric, data_dir, index_dir, R, L, alpha)
        else:
            raise Exception('Invalid data type')
    elif metric == 'mips':
        if dtype == 'uint8':
            build_vamana_uint8_mips_index(metric, data_dir, index_dir, R, L, alpha)
        elif dtype == 'int8':
            build_vamana_int8_mips_index(metric, data_dir, index_dir, R, L, alpha)
        elif dtype == 'float':
            build_vamana_float_mips_index(metric, data_dir, index_dir, R, L, alpha)
        else:
            raise Exception('Invalid data type')
    else:
        raise Exception('Invalid metric')
        
def load_vamana_index(metric, dtype, data_dir, index_dir, n, d):
    if metric == 'Euclidian':
        if dtype == 'uint8':
            return VamanaUInt8EuclidianIndex(data_dir, index_dir, n, d)
        elif dtype == 'int8':
            return VamanaInt8EuclidianIndex(data_dir, index_dir, n, d)
        elif dtype == 'float':
            return VamanaFloatEuclidianIndex(data_dir, index_dir, n, d)
        else:
            raise Exception('Invalid data type')
    elif metric == 'mips':
        if dtype == 'uint8':
            return VamanaUInt8MipsIndex(data_dir, index_dir, n, d)
        elif dtype == 'int8':
            return VamanaInt8MipsIndex(data_dir, index_dir, n, d)
        elif dtype == 'float':
            return VamanaFloatMipsIndex(data_dir, index_dir, n, d)
        else:
            raise Exception('Invalid data type')
    else:
        raise Exception('Invalid metric')