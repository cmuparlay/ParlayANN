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

from typing import BinaryIO, Optional, overload

import numpy as np

from . import VectorDType, VectorIdentifierBatch, VectorLikeBatch

def numpy_to_diskann_file(vectors: np.ndarray, file_handler: BinaryIO): ...
def build_vamana_index(
    data: str,
    distance_metric: str,
    index_directory: str,
    beam_width: int,
    graph_degree: int,
    alpha: float,
    vector_dtype: VectorDType,
    index_prefix: str,
) -> None: ...

