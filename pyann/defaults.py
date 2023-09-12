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
# Parameter Defaults
These parameter defaults are re-exported from the C++ extension module, and used to keep the pythonic wrapper in sync with the C++.
"""
from ._ParlayANNpy import defaults as _defaults

ALPHA = _defaults.ALPHA
""" 
Note that, as ALPHA is a `float32` (single precision float) in C++, when converted into Python it becomes a 
`float64` (double precision float). The actual value is 1.2f. The alpha parameter (>=1) is used to control the nature 
and number of points that are added to the graph. A higher alpha value (e.g., 1.4) will result in fewer hops (and IOs) 
to convergence, but probably more distance comparisons compared to a lower alpha value.
"""
GRAPH_DEGREE = _defaults.GRAPH_DEGREE
""" 
Graph degree (a.k.a. `R`) is the maximum degree allowed for a node in the index's graph structure. This degree will be 
pruned throughout the course of the index build, but it will never grow beyond this value. Higher R values require 
longer index build times, but may result in an index showing excellent recall and latency characteristics. 
"""
BEAMWIDTH = _defaults.BEAMWIDTH
""" 
Complexity (a.k.a `L`) references the size of the list we store candidate approximate neighbors in while doing build
or search tasks. It's used during index build as part of the index optimization processes. It's used in index search 
classes both to help mitigate poor latencies during cold start, as well as on subsequent queries to conduct the search. 
Large values will likely increase latency but also may improve recall, and tuning these values for your particular 
index is certainly a reasonable choice.
"""