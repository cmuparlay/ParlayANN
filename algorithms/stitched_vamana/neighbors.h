// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>

#include "../utils/NSGDist.h"
#include "../utils/beamSearch.h"
#include "../utils/check_nn_recall.h"
#include "../utils/parse_results.h"
#include "../utils/filters.h"
#include "../utils/stats.h"
#include "../utils/types.h"
#include "../utils/graph.h"
#include "stitched_vamana.h"
#include "../vamana/index.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"


template<typename T, typename Point, typename PointRange>
void ANN(Graph<indexType> &G, BuildParams &BP_large, BuildParams &BP_small,
         PointRange &Points, csr_filters& filters) {
  parlay::internal::timer t("ANN");
  using index = StitchedVamana<T, Point>;
  index I(BP_large, BP_small);
  double idx_time;


  I.build_index(Points, G, filters);
  idx_time = t.next_time();
  std::cout << "Built index in " << idx_time << " seconds" << std::endl;

  
  }

int main(){
  csr_filters filters("/ssd1/anndata/bigann/");
  PointRange<uint8_t, Euclidian_Point<uint8_t>> Points;

  
}



