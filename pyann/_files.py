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

import warnings
from typing import BinaryIO, NamedTuple

import numpy as np
import numpy.typing as npt

from . import VectorDType, VectorIdentifierBatch, VectorLikeBatch
from ._common import _assert, _assert_2d, _assert_dtype, _assert_existing_file


class Metadata(NamedTuple):
    """DiskANN binary vector files contain a small stanza containing some metadata about them."""

    num_vectors: int
    """ The number of vectors in the file. """
    dimensions: int
    """ The dimensionality of the vectors in the file. """


def vectors_metadata_from_file(vector_file: str) -> Metadata:
    """
    Read the metadata from a DiskANN binary vector file.
    ### Parameters
    - **vector_file**: The path to the vector file to read the metadata from.

    ### Returns
    `diskannpy.Metadata`
    """
    _assert_existing_file(vector_file, "vector_file")
    points, dims = np.fromfile(file=vector_file, dtype=np.int32, count=2)
    return Metadata(points, dims)


def _write_bin(data: np.ndarray, file_handler: BinaryIO):
    if len(data.shape) == 1:
        _ = file_handler.write(np.array([data.shape[0], 1], dtype=np.int32).tobytes())
    else:
        _ = file_handler.write(np.array(data.shape, dtype=np.int32).tobytes())
    _ = file_handler.write(data.tobytes())


def vectors_to_file(vector_file: str, vectors: VectorLikeBatch) -> None:
    """
    Utility function that writes a DiskANN binary vector formatted file to the location of your choosing.

    ### Parameters
    - **vector_file**: The path to the vector file to write the vectors to.
    - **vectors**: A 2d array of dtype `numpy.float32`, `numpy.uint8`, or `numpy.int8`
    """
    _assert_dtype(vectors.dtype)
    _assert_2d(vectors, "vectors")
    with open(vector_file, "wb") as fh:
        _write_bin(vectors, fh)


def vectors_from_file(vector_file: str, dtype: VectorDType) -> npt.NDArray[VectorDType]:
    """
    Read vectors from a DiskANN binary vector file.

    ### Parameters
    - **vector_file**: The path to the vector file to read the vectors from.
    - **dtype**: The data type of the vectors in the file. Ensure you match the data types exactly

    ### Returns
    `numpy.typing.NDArray[dtype]`
    """
    points, dims = vectors_metadata_from_file(vector_file)
    return np.fromfile(file=vector_file, dtype=dtype, offset=8).reshape(points, dims)





