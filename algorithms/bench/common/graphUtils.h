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

#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include "graph.h"
#include "../parlay/parallel.h"
#include "../parlay/primitives.h"
#include "../parlay/random.h"
#include "dataGen.h"

using namespace std;

template <class intV, class Weight = DefaultWeight>
wghEdgeArray<intV,Weight> addRandWeights(edgeArray<intV> const &G) {
  using WE = wghEdge<intV,Weight>;
  parlay::random r(257621);
  intV m = G.nonZeros;
  intV n = G.numRows;
  auto E = parlay::tabulate(m, [&] (size_t i) -> WE {
      return WE(G.E[i].u, G.E[i].v, (Weight) dataGen::hash<Weight>(i));});
  return wghEdgeArray<intV,Weight>(std::move(E), n);
}

template <class intV>
edgeArray<intV> randomShuffle(edgeArray<intV> const &A) {
  auto E =  parlay::random_shuffle(A.E);
  return edgeArray<intV>(std::move(E), A.numRows, A.numCols);
}

template <class intV>
edgeArray<intV> remDuplicates(edgeArray<intV> const &A) {
  auto lessE = [&] (edge<intV> a, edge<intV> b) {
    return (a.u < b.u) || ((a.u == b.u) && (a.v < b.v));};
  parlay::sequence<edge<intV>> E =
    parlay::remove_duplicates_ordered(A.E, lessE);
  return edgeArray<intV>(std::move(E), A.numRows, A.numCols);
}

template <class intV>
edgeArray<intV> makeSymmetric(edgeArray<intV> const &A) {
  parlay::sequence<edge<intV>> EF = parlay::filter(A.E, [&] (edge<intV> e) {
      return e.u != e.v;});
  auto FE = parlay::delayed_seq<edge<intV>>(EF.size(), [&] (size_t i) {
      return edge<intV>(EF[i].v, EF[i].u);});
  return remDuplicates(edgeArray<intV>(parlay::append(EF, FE),
				       A.numRows, A.numCols));
}

template <class intV, class intE = intV>
graph<intV,intE> graphFromEdges(edgeArray<intV> const &EA, bool makeSym) {
  edgeArray<intV> SA;
  if (makeSym) SA = makeSymmetric<intV>(EA);
  edgeArray<intV> const &A = (makeSym) ? SA : EA;

  size_t m = A.nonZeros;
  size_t n = std::max(A.numCols, A.numRows);

  parlay::sequence<size_t> counts;
  parlay::sequence<intE> offsets;
  parlay::sequence<edge<intV>> E;
  size_t nn;
  auto getu = [&] (edge<intV> e) {return e.u;};
  std::tie(E, counts) = parlay::internal::integer_sort_with_counts(parlay::make_slice(A.E), getu, n);
  std::tie(offsets,nn) = parlay::scan(parlay::delayed_seq<intE>(n+1, [&] (size_t i) {
	return (i == n) ? 0 : counts[i];}), parlay::addm<intE>());

  return graph<intV,intE>(std::move(offsets),
			  parlay::tabulate(m, [&] (size_t i) -> intV {return E[i].v;}),
			  n);
}

template <class intV, class Weight, class intE=intV>
wghGraph<intV,Weight,intE>
wghGraphFromEdges(wghEdgeArray<intV,Weight> const &A) {
  using WE = wghEdge<intV,Weight>;
  size_t n = A.n;
  size_t m = A.m;

  parlay::sequence<size_t> counts;
  parlay::sequence<intE> offsets;
  parlay::sequence<WE> E;
  size_t nn;
  auto getu = [&] (WE e) {return e.u;};
  std::tie(E, counts) = parlay::internal::integer_sort_with_counts(parlay::make_slice(A.E), getu, n);
  std::tie(offsets,nn) = parlay::scan(parlay::delayed_seq<intE>(n+1, [&] (size_t i) {
	return (i == n) ? 0 : counts[i];}), parlay::addm<intE>());

  return wghGraph<intV,Weight,intE>(std::move(offsets),
				    parlay::tabulate(m, [&] (size_t i)->intV {return E[i].v;}),
				    parlay::tabulate(m, [&] (size_t i) -> Weight {
					return E[i].weight;}),
				    n);
}

template <class intV, class intE>
edgeArray<intV> edgesFromGraph(graph<intV,intE> const &G) {
  size_t numRows = G.numVertices();
  size_t nonZeros = G.numEdges();

  // flatten
  parlay::sequence<edge<intV>> E(nonZeros);
  parlay::parallel_for(0, numRows, [&] (size_t j) {
      size_t off = G.get_offsets()[j];
      vertex<intV> v = G[j];
      for (size_t i = 0; i < v.degree; i++)
	E[off+i] = edge<intV>(j, v.Neighbors[i]);
    });
  return edgeArray<intV>(std::move(E), numRows, numRows);
}

// offset for start of each vertex if flattening the edge listd
template <class intV, class intE, class Vtx>
parlay::sequence<intE> getOffsets(parlay::sequence<Vtx> const &V) {
  size_t n = V.size();
  auto degrees = parlay::delayed_seq<intE>(n+1, [&] (size_t i) -> intE {
      return (i == n) ? 0 : V[i].degree;});
  auto x = parlay::scan(degrees, parlay::addm<intE>());
  return x.first;
}

// packs a graph so that there are no gaps in the edge array (i.e. into CSR)
template <class intV, class intE>
graph<intV,intE> packGraph(graph<intV,intE> const &G) {
  size_t n = G.numVertices();
  auto degrees = parlay::delayed_seq<intE>(n+1, [&] (size_t i) -> intE {
						  return (i == n) ? 0 : G[i].degree;});
  // calculate new offsets
  auto sr = parlay::scan(degrees, parlay::addm<intE>());
  // allocate new edge array
  parlay::sequence<intV> outEdges(sr.second);
  // copy edges so they are contiguous
  parlay::parallel_for (0, G.n, [&] (size_t i) {
      vertex<intV> v = G[i];
      size_t offset = sr.first[i];
      for (size_t j=0; j < v.degree; j++)
	outEdges[offset + j] = v.Neighbors[j];
    });
  return graph<intV,intE>(std::move(sr.first), std::move(outEdges), n);
}

// if I is NULL then it randomly reorders
template <class intV, class intE>
graph<intV,intE> graphReorder(graph<intV,intE> const &Gr,
			      parlay::sequence<intV> const &I = parlay::sequence<intV>(0)) {
  intV n = Gr.numVertices();
  intV m = Gr.numEdges();

  bool noI = (I.size()==0);
  parlay::sequence<intV> const &II = noI ? parlay::random_permutation<intV>(n) : I;

  // now write vertices to new locations
  // inverse permutation
  parlay::sequence<vertex<intV>> V(n);
  parlay::parallel_for (0, n, [&] (size_t i) {
      V[II[i]] = Gr[i];});
  parlay::sequence<intE> offsets = getOffsets<intV,intE>(V);
  parlay::sequence<intV> E(m);
  parlay::parallel_for (0, n, [&] (size_t i) {
      size_t o = offsets[i];
      for (size_t j=0; j < V[i].degree; j++) 
	E[o + j] = II[V[i].Neighbors[j]];
      std::sort(E.begin() + o, E.begin() + o + V[i].degree);
    }, 1000);
  return graph<intV>(std::move(offsets), std::move(E), n);
}

template <class intV, class intE>
int graphCheckConsistency(graph<intV,intE> const &Gr) {
  size_t n = Gr.numVertices();
  size_t m = Gr.numEdges();
  size_t edgecount = parlay::reduce(parlay::delayed_seq<size_t>(n, [&] (size_t i) {
	return Gr[i].degree;}), parlay::addm<size_t>());
  if (m != edgecount) {
    cout << "bad edge count in graphCheckConsistency: m = " 
	 << m << " sum of degrees = " << edgecount << endl;
    return 1;
  }
  size_t error_loc = parlay::reduce(parlay::delayed_seq<size_t>(n, [&] (size_t i) {
	for (size_t j=0; j < Gr[i].degree; j++) 
	  if (Gr[i].Neighbors[j] >= n) return i;
	return n;
      }), parlay::minm<size_t>());
  if (error_loc < n) {
    cout << "edge out of range in graphCheckConsistency: at i = " 
	 << error_loc << endl;
    return 1;
  }
}

// template <class intV>
// sparseRowMajor<double,intV> sparseFromCsrFile(const char* fname) {
//   FILE *f = fopen(fname,"r");
//   if (f == NULL) {
//     cout << "Trying to open nonexistant file: " << fname << endl;
//     abort();
//   }

//   intV numRows;  intV numCols;  intV nonZeros;
//   intV nc = fread(&numRows, sizeof(intV), 1, f);
//   nc = fread(&numCols, sizeof(intV), 1, f);
//   nc = fread(&nonZeros, sizeof(intV), 1, f); 

//   double *Values = (double *) malloc(sizeof(double)*nonZeros);
//   intV *ColIds = (intV *) malloc(sizeof(intV)*nonZeros);
//   intV *Starts = (intV *) malloc(sizeof(intV)*(1 + numRows));
//   Starts[numRows] = nonZeros;

//   size_t r;
//   r = fread(Values, sizeof(double), nonZeros, f);
//   r = fread(ColIds, sizeof(intV), nonZeros, f);
//   r = fread(Starts, sizeof(intV), numRows, f); 
//   fclose(f);
//   return sparseRowMajor<double,intV>(numRows,numCols,nonZeros,Starts,ColIds,Values);
// }

// template <class intV>
// edgeArray<intV> edgesFromMtxFile(const char* fname) {
//   ifstream file (fname, ios::in);
//   char* line = newA(char,1000);
//   intV i,j = 0;
//   while (file.peek() == '%') {
//     j++;
//     file.getline(line,1000);
//   }
//   intV numRows, numCols, nonZeros;
//   file >> numRows >> numCols >> nonZeros;
//   //cout << j << "," << numRows << "," << numCols << "," << nonZeros << endl;
//   edge<intV> *E = newA(edge<intV>,nonZeros);
//   double toss;
//   for (i=0, j=0; i < nonZeros; i++) {
//     file >> E[j].u >> E[j].v >> toss;
//     E[j].u--;
//     E[j].v--;
//     if (toss != 0.0) j++;
//   }
//   nonZeros = j;
//   //cout << "nonzeros = " << nonZeros << endl;
//   file.close();  
//   return edgeArray<intV>(E,numRows,numCols,nonZeros);
// }

