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

"""
# Documentation Overview
`ParlayANNpy` is mostly structured around 2 distinct processes: [Index Builder Functions](#index-builders) and [Search Classes](#search-classes)
This code is adapted from the DiskANN pybindings and heavily draws on their implementation.

It also includes a few nascent [utilities](#utilities).

And lastly, it makes substantial use of type hints, with various shorthand [type aliases](#parameter-and-response-type-aliases) documented. 
When reading the `ParlayANNpy` code we refer to the type aliases.

## Index Builders
- `build_vamana_index` - Builds an in-memory Vamana index

## Search Classes
- `VamanaIndex` - fully in memory and static

## Parameter Defaults
- `ParlayANNpy.defaults` - Default values exported from the C++ extension for Python users

## Parameter and Response Type Aliases
- `DistanceMetric` - What distance metrics does `ParlayANNpy` support?
- `VectorDType` - What vector datatypes does `ParlayANNpy` support?
- `QueryResponse` - What can I expect as a response to my search?
- `QueryResponseBatch` - What can I expect as a response to my batch search?
- `VectorIdentifier` - What types do `ParlayANNpy` support as vector identifiers?
- `VectorIdentifierBatch` - A batch of identifiers of the exact same type. The type can change, but they must **all** change.
- `VectorLike` - How does a vector look to `ParlayANNpy`, to be inserted or searched with.
- `VectorLikeBatch` - A batch of those vectors, to be inserted or searched with.
- `Metadata` - DiskANN vector binary file metadata (num_points, vector_dim)

## Utilities
- `vectors_to_file` - Turns a 2 dimensional `numpy.typing.NDArray[VectorDType]` with shape `(number_of_points, vector_dim)` into a ParlayANN vector bin file.
- `vectors_from_file` - Reads a ParlayANN vector bin file representing stored vectors into a numpy ndarray.
- `vectors_metadata_from_file` - Reads metadata stored in a ParlayANN vector bin file without reading the entire file
- `valid_dtype` - Checks if a given vector dtype is supported by `ParlayANNpy`
"""

from typing import Any, Literal, NamedTuple, Type, Union

import numpy as np
from numpy import typing as npt

DistanceMetric = Literal["Euclidian", "mips"]
""" Type alias for one of {"l2", "mips",} """
VectorDType = Union[Type[np.float32], Type[np.int8], Type[np.uint8]]
""" Type alias for one of {`numpy.float32`, `numpy.int8`, `numpy.uint8`} """
VectorLike = npt.NDArray[VectorDType]
""" Type alias for something that can be treated as a vector """
VectorLikeBatch = npt.NDArray[VectorDType]
""" Type alias for a batch of VectorLikes """
VectorIdentifier = np.uint32
""" 
Type alias for a vector identifier, whether it be an implicit array index identifier from StaticMemoryIndex or 
StaticDiskIndex, or an explicit tag identifier from DynamicMemoryIndex 
"""
VectorIdentifierBatch = npt.NDArray[np.uint32]
""" Type alias for a batch of VectorIdentifiers """


class QueryResponse(NamedTuple):
    """
    Tuple with two values, identifiers and distances. Both are 1d arrays, positionally correspond, and will contain the
    nearest neighbors from [0..k_neighbors)
    """

    identifiers: npt.NDArray[VectorIdentifier]
    """ A `numpy.typing.NDArray[VectorIdentifier]` array of vector identifiers, 1 dimensional """
    distances: npt.NDArray[np.float32]
    """
    A `numpy.typing.NDAarray[numpy.float32]` of distances as calculated by the distance metric function,  1 dimensional
    """


class QueryResponseBatch(NamedTuple):
    """
    Tuple with two values, identifiers and distances. Both are 2d arrays, with dimensionality determined by the
    rows corresponding to the number of queries made, and the columns corresponding to the k neighbors
    requested. The two 2d arrays have an implicit, position-based relationship
    """

    identifiers: npt.NDArray[VectorIdentifier]
    """ 
    A `numpy.typing.NDArray[VectorIdentifier]` array of vector identifiers, 2 dimensional. The row corresponds to index 
    of the query, and the column corresponds to the k neighbors requested 
    """
    distances: np.ndarray[np.float32]
    """  
    A `numpy.typing.NDAarray[numpy.float32]` of distances as calculated by the distance metric function, 2 dimensional. 
    The row corresponds to the index of the query, and the column corresponds to the distance of the query to the 
    *k-th* neighbor 
    """


from . import defaults
from ._builder import build_vamana_index
from ._common import valid_dtype
# TODO implement searching once index build works
# from ._dynamic_memory_index import DynamicMemoryIndex
from ._files import (
    Metadata,
    vectors_from_file,
    vectors_metadata_from_file,
    vectors_to_file,
)
# from ._static_disk_index import StaticDiskIndex
# from ._static_memory_index import StaticMemoryIndex

__all__ = [
    "build_vamana_index",
    # "StaticDiskIndex", //TODO add back search index
    "defaults",
    "DistanceMetric",
    "VectorDType",
    "QueryResponse",
    "QueryResponseBatch",
    "VectorIdentifier",
    "VectorIdentifierBatch",
    "VectorLike",
    "VectorLikeBatch",
    "Metadata",
    "vectors_metadata_from_file",
    "vectors_to_file",
    "vectors_from_file",
    "valid_dtype",
]