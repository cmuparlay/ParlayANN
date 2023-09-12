# // This code is part of the Problem Based Benchmark Suite (PBBS)
# // Copyright (c) 2011 Guy Blelloch and the PBBS team
# //
# // Permission is hereby granted, free of charge, to any person obtaining a
# // copy of this software and associated documentation files (the
# // "Software"), to deal in the Software without restriction, including
# // without limitation the rights (to use, copy, modify, merge, publish,
# // distribute, sublicense, and/or sell copies of the Software, and to
# // permit persons to whom the Software is furnished to do so, subject to
# // the following conditions:
# //
# // The above copyright notice and this permission notice shall be included
# // in all copies or substantial portions of the Software.
# //
# // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# // OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# // MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# // NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# // LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# // OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# // WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from . import VectorDType, VectorIdentifierBatch, VectorLikeBatch
from . import _ParlayANNpy as _parlayann
from ._common import (
    _assert,
    _assert_is_nonnegative_uint32,
    _assert_is_positive_uint32,
    _castable_dtype_or_raise,
    _valid_metric,
    _write_index_metadata,
    valid_dtype,
)
from ._ParlayANNpy import defaults
from ._files import tags_to_file, vectors_metadata_from_file, vectors_to_file


def _valid_path_and_dtype(
    data: Union[str, VectorLikeBatch],
    vector_dtype: VectorDType,
    index_path: str,
    index_prefix: str,
) -> Tuple[str, VectorDType]:
    if isinstance(data, str):
        vector_bin_path = data
        _assert(
            Path(data).exists() and Path(data).is_file(),
            "if data is of type `str`, it must both exist and be a file",
        )
        vector_dtype_actual = valid_dtype(vector_dtype)
    else:
        vector_bin_path = os.path.join(index_path, f"{index_prefix}_vectors.bin")
        if Path(vector_bin_path).exists():
            raise ValueError(
                f"The path {vector_bin_path} already exists. Remove it and try again."
            )
        vector_dtype_actual = valid_dtype(data.dtype)
        vectors_to_file(vector_file=vector_bin_path, vectors=data)

    return vector_bin_path, vector_dtype_actual



def build_memory_index(
    data: Union[str, VectorLikeBatch],
    distance_metric: str,
    index_directory: str,
    beam_width: int,
    graph_degree: int,
    alpha: float = defaults.ALPHA,
    vector_dtype: Optional[VectorDType] = None,
    index_prefix: str = "ann",
) -> None:
    _assert(
        (isinstance(data, str) and vector_dtype is not None)
        or isinstance(data, np.ndarray),
        "vector_dtype is required if data is a str representing a path to the vector bin file",
    )
    dap_metric = _valid_metric(distance_metric)
    _assert_is_positive_uint32(complexity, "complexity")
    _assert_is_positive_uint32(graph_degree, "graph_degree")
    _assert(
        alpha >= 1,
        "alpha must be >= 1, and realistically should be kept between [1.0, 2.0)",
    )
    _assert_is_nonnegative_uint32(num_threads, "num_threads")
    _assert_is_nonnegative_uint32(num_pq_bytes, "num_pq_bytes")
    _assert_is_nonnegative_uint32(filter_complexity, "filter_complexity")
    _assert(index_prefix != "", "index_prefix cannot be an empty string")

    index_path = Path(index_directory)
    _assert(
        index_path.exists() and index_path.is_dir(),
        "index_directory must both exist and be a directory",
    )

    vector_bin_path, vector_dtype_actual = _valid_path_and_dtype(
        data, vector_dtype, index_directory, index_prefix
    )

    num_points, dimensions = vectors_metadata_from_file(vector_bin_path)

    if vector_dtype_actual == np.uint8:
        _builder = _parlayann.build_memory_uint8_index
    elif vector_dtype_actual == np.int8:
        _builder = _parlayann.build_memory_int8_index
    else:
        _builder = _parlayann.build_memory_float_index

    index_prefix_path = os.path.join(index_directory, index_prefix)

    _builder(
        distance_metric=dap_metric,
        data_file_path=vector_bin_path,
        index_output_path=index_prefix_path,
        beam_width=beam_width,
        graph_degree=graph_degree,
        alpha=alpha,
    )

    _write_index_metadata(
        index_prefix_path, vector_dtype_actual, dap_metric, num_points, dimensions
    )