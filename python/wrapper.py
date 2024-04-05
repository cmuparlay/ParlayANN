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
            raise Exception('Invalid data type ' + dtype)
    elif metric == 'mips':
        # raise Exception('MIPS commented out to speed up build')
        if dtype == 'uint8':
            build_vamana_uint8_mips_index(metric, data_dir, index_dir, R, L, alpha)
        elif dtype == 'int8':
            build_vamana_int8_mips_index(metric, data_dir, index_dir, R, L, alpha)
        elif dtype == 'float':
            build_vamana_float_mips_index(metric, data_dir, index_dir, R, L, alpha)
        else:
            raise Exception('Invalid data type ' + dtype)
    else:
        raise Exception('Invalid metric ' + metric)
        
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
        # raise Exception('MIPS commented out to speed up build')
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
    
def init_ivf_index(metric, dtype):
    if metric == 'Euclidian':
        if dtype == 'uint8':
            return IVFUInt8EuclidianIndex()
        elif dtype == 'int8':
            return IVFInt8EuclidianIndex()
        elif dtype == 'float':
            return IVFFloatEuclidianIndex()
        else:
            raise Exception('Invalid data type ' + dtype)
    elif metric == 'mips':
        # raise Exception('MIPS commented out to speed up build')
        if dtype == 'uint8':
            return IVFUInt8MipsIndex()
        elif dtype == 'int8':
            return IVFInt8MipsIndex()
        elif dtype == 'float':
            return IVFFloatMipsIndex()
        else:
            raise Exception('Invalid data type ' + dtype)
    else:
        raise Exception('Invalid metric ' + metric)
    
def init_filtered_ivf_index(metric, dtype):
    if metric == 'Euclidian':
        if dtype == 'uint8':
            return FilteredIVFUInt8EuclidianIndex()
        elif dtype == 'int8':
            return FilteredIVFInt8EuclidianIndex()
        elif dtype == 'float':
            return FilteredIVFFloatEuclidianIndex()
        else:
            raise Exception('Invalid data type ' + dtype)
    elif metric == 'mips':
        # raise Exception('MIPS commented out to speed up build')
        if dtype == 'uint8':
            return FilteredIVFUInt8MipsIndex()
        elif dtype == 'int8':
            return FilteredIVFInt8MipsIndex()
        elif dtype == 'float':
            return FilteredIVFFloatMipsIndex()
        else:
            raise Exception('Invalid data type ' + dtype)
    else:
        raise Exception('Invalid metric ' + metric)
    
def init_2_stage_filtered_ivf_index(metric, dtype):
    if metric == 'Euclidian':
        if dtype == 'uint8':
            return Filtered2StageIVFUInt8EuclidianIndex()
        elif dtype == 'int8':
            return Filtered2StageIVFInt8EuclidianIndex()
        elif dtype == 'float':
            return Filtered2StageIVFFloatEuclidianIndex()
        else:
            raise Exception('Invalid data type ' + dtype)
    elif metric == 'mips':
        # raise Exception('MIPS commented out to speed up build')
        if dtype == 'uint8':
            return Filtered2StageIVFUInt8MipsIndex()
        elif dtype == 'int8':
            return Filtered2StageIVFInt8MipsIndex()
        elif dtype == 'float':
            return Filtered2StageIVFFloatMipsIndex()
        else:
            raise Exception('Invalid data type ' + dtype)
    else:
        raise Exception('Invalid metric ' + metric)

def init_squared_ivf_index(metric, dtype):
    if metric == 'Euclidian':
        if dtype == 'uint8':
            return SquaredIVFUInt8EuclidianIndex()
        elif dtype == 'int8':
            return SquaredIVFInt8EuclidianIndex()
        elif dtype == 'float':
            return SquaredIVFFloatEuclidianIndex()
        else:
            raise Exception('Invalid data type ' + dtype)
    elif metric == 'mips':
        # raise Exception('MIPS commented out to speed up build')
        if dtype == 'uint8':
            return SquaredIVFUInt8MipsIndex()
        elif dtype == 'int8':
            return SquaredIVFInt8MipsIndex()
        elif dtype == 'float':
            return SquaredIVFFloatMipsIndex()
        else:
            raise Exception('Invalid data type ' + dtype)
    else:
        raise Exception('Invalid metric ' + metric)
    
def init_stitched_vamana_index(metric, dtype):
    if metric == 'Euclidian':
        if dtype == 'uint8':
            return StitchedVamanaUInt8EuclidianIndex()
        elif dtype == 'int8':
            return StitchedVamanaInt8EuclidianIndex()
        elif dtype == 'float':
            return StitchedVamanaFloatEuclidianIndex()
        else:
            raise Exception('Invalid data type ' + dtype)
    elif metric == 'mips':
        # raise Exception('MIPS commented out to speed up build')
        if dtype == 'uint8':
            return StitchedVamanaUInt8MipsIndex()
        elif dtype == 'int8':
            return StitchedVamanaInt8MipsIndex()
        elif dtype == 'float':
            return StitchedVamanaFloatMipsIndex()
        else:
            raise Exception('Invalid data type ' + dtype)
    else:
        raise Exception('Invalid metric ' + metric)

def init_hybrid_stitched_vamana_index(metric, dtype):
    if metric == 'Euclidian':
        if dtype == 'uint8':
            return HybridStitchedVamanaUInt8EuclidianIndex()
        elif dtype == 'int8':
            return HybridStitchedVamanaInt8EuclidianIndex()
        elif dtype == 'float':
            return HybridStitchedVamanaFloatEuclidianIndex()
        else:
            raise Exception('Invalid data type ' + dtype)
    elif metric == 'mips':
        # raise Exception('MIPS commented out to speed up build')
        if dtype == 'uint8':
            return HybridStitchedVamanaUInt8MipsIndex()
        elif dtype == 'int8':
            return HybridStitchedVamanaInt8MipsIndex()
        elif dtype == 'float':
            return HybridStitchedVamanaFloatMipsIndex()
        else:
            raise Exception('Invalid data type ' + dtype)
    else:
        raise Exception('Invalid metric ' + metric)
    
def init_filtered_dataset(metric, dtype):
    if metric == 'Euclidian':
        if dtype == 'uint8':
            return FilteredUInt8EuclidianDataset
        elif dtype == 'int8':
            return FilteredInt8EuclidianDataset
        elif dtype == 'float':
            return FilteredFloatEuclidianDataset
        else:
            raise Exception('Invalid data type ' + dtype)
    elif metric == 'mips':
        # raise Exception('MIPS commented out to speed up build')
        if dtype == 'uint8':
            return FilteredUInt8MipsDataset
        elif dtype == 'int8':
            return FilteredInt8MipsDataset
        elif dtype == 'float':
            return FilteredFloatMipsDataset
        else:
            raise Exception('Invalid data type ' + dtype)
    else:
        raise Exception('Invalid metric ' + metric)