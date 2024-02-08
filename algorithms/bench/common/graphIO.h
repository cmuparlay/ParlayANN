// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2010 Guy Blelloch and the PBBS team
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
#include <stdint.h>
#include <cstring>
#include "../parlay/parallel.h"
#include "IO.h"
#include "graphUtils.h"

#include <sys/mman.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using namespace benchIO;

template <class intV>
int xToStringLen(edge<intV> a) {
  return xToStringLen(a.u) + xToStringLen(a.v) + 1;
}

template <class intV>
void xToString(char* s, edge<intV> a) {
  int l = xToStringLen(a.u);
  xToString(s, a.u);
  s[l] = ' ';
  xToString(s+l+1, a.v);
}

template <class intV, class Weight>
int xToStringLen(wghEdge<intV,Weight> a) {
  return xToStringLen(a.u) + xToStringLen(a.v) + xToStringLen(a.weight) + 2;
}

template <class intV, class Weight>
void xToString(char* s, wghEdge<intV, Weight> a) {
  int lu = xToStringLen(a.u);
  int lv = xToStringLen(a.v);
  xToString(s, a.u);
  s[lu] = ' ';
  xToString(s+lu+1, a.v);
  s[lu+lv+1] = ' ';
  xToString(s+lu+lv+2, a.weight);
}

namespace benchIO {
  using namespace std;

  string AdjGraphHeader = "AdjacencyGraph";
  string EdgeArrayHeader = "EdgeArray";
  string WghEdgeArrayHeader = "WeightedEdgeArray";
  string WghAdjGraphHeader = "WeightedAdjacencyGraph";

  template <class intV, class intE>
  int writeGraphToFile(graph<intV, intE> const &G, char* fname) {
    if (G.degrees.size() > 0) {
      graph<intV, intE> GP = packGraph(G);
      return writeGraphToFile(GP, fname);
    }
    size_t m = G.numEdges();
    size_t n = G.numVertices();
    size_t totalLen = 2 + n + m;
    parlay::sequence<size_t> Out(totalLen);
    Out[0] = n;
    Out[1] = m;

    // write offsets to Out[2,..,2+n)
    parlay::sequence<intE> const &offsets = G.get_offsets();
    parlay::parallel_for (0, n, [&] (size_t i) {
    	Out[i+2] = offsets[i];});

    // write out edges to Out[2+n,..,2+n+m)
    parlay::parallel_for(0, n, [&] (size_t i) {
    	size_t o = offsets[i] + 2 + n;
    	for (intV j = 0; j < G[i].degree; j++) 
    	  Out[o + j] = G[i].Neighbors[j];
      });

    int r = writeSeqToFile(AdjGraphHeader, Out, fname);
    return r;
  }

  template <class intV, class Weight, class intE>
  int writeWghGraphToFile(wghGraph<intV,Weight,intE> G, char* fname) {
    size_t m = G.m;
    size_t n = G.n;
    // weights have to separate since they could be floats
    parlay::sequence<size_t> Out1(2 + n + m);
    parlay::sequence<Weight> Out2(m);
    Out1[0] = n;
    Out2[1] = m;

    // write offsets to Out[2,..,2+n)
    auto offsets = G.get_offsets();
    parlay::parallel_for (0, n, [&] (size_t i) {
	Out1[i+2] = offsets[i];});

    // write out edges to Out1[2+n,..,2+n+m)
    // and weights to Out2[0,..,m)
    parlay::parallel_for(0, n, [&] (size_t i) {
	size_t o = offsets[i];
	wghVertex<intV,Weight> v = G[i];
	for (intV j = 0; j < v.degree; j++) {
	  Out1[2 + n + o + j] = v.Neighbors[j];
	  Out2[o + j] = v.nghWeights[j]; }
      });
    int r = write2SeqToFile(WghAdjGraphHeader, Out1, Out2, fname);
    return r;
  }

  template <class intV>
  int writeEdgeArrayToFile(edgeArray<intV> const &EA, char* fname) {
    return writeSeqToFile(EdgeArrayHeader, EA.E, fname);
  }

  template <class intV, class intE>
  int writeWghEdgeArrayToFile(wghEdgeArray<intV,intE>
			      const &EA, char* fname) {
    return writeSeqToFile(WghEdgeArrayHeader, EA.E, fname);
  }

  template <class intV>
  edgeArray<intV> readEdgeArrayFromFile(char* fname) {
    parlay::sequence<char> S = readStringFromFile(fname);
    parlay::sequence<char*> W = stringToWords(S);
    if (W[0] != EdgeArrayHeader) {
      cout << "Bad input file" << endl;
      abort();
    }
    long n = (W.size()-1)/2;
    auto E = parlay::tabulate(n, [&] (long i) -> edge<intV> {
	return edge<intV>(atol(W[2*i + 1]),
			  atol(W[2*i + 2]));});

    auto mon = parlay::make_monoid([&] (edge<intV> a, edge<intV> b) {
	return edge<intV>(std::max(a.u, b.u), std::max(a.v, b.v));},
      edge<intV>(0,0));
    auto r = parlay::reduce(E, mon);

    intV maxrc = std::max(r.u, r.v) + 1;
    return edgeArray<intV>(std::move(E), maxrc, maxrc);
  }

  template <class intV, class Weight>
  wghEdgeArray<intV,Weight> readWghEdgeArrayFromFile(char* fname) {
    using WE = wghEdge<intV,Weight>;
    parlay::sequence<char> S = readStringFromFile(fname);
    parlay::sequence<char*> W = stringToWords(S);
    if (W[0] != WghEdgeArrayHeader) {
      cout << "Bad input file" << endl;
      abort();
    }
    long n = (W.size()-1)/3;
    auto E = parlay::tabulate(n, [&] (size_t i) -> WE {
	return WE(atol(W[3*i + 1]),
		  atol(W[3*i + 2]),
		  (Weight) atof(W[3*i + 3]));});

    auto mon = parlay::make_monoid([&] (WE a, WE b) {
	return WE(std::max(a.u, b.u), std::max(a.v, b.v), 0);},
      WE(0,0,0));
    auto r = parlay::reduce(E, mon);

    return wghEdgeArray<intV,Weight>(std::move(E), max<intV>(r.u, r.v) + 1);
  }

  template <class intV, class intE=intV>
  graph<intV, intE> readGraphFromFile(char* fname) {
    auto W = get_tokens(fname);
    string header(W[0].begin(), W[0].end());
    if (header != AdjGraphHeader) {
      cout << "Bad input file: missing header: " << AdjGraphHeader << endl;
      abort();
    }

    // file consists of [type, num_vertices, num_edges, <vertex offsets>, <edges>]
    // in compressed sparse row format
    long n = parlay::chars_to_long(W[1]);
    long m = parlay::chars_to_long(W[2]);
    if (W.size() != n + m + 3) {
      cout << "Bad input file: length = "<< W.size() << " n+m+3 = " << n+m+3 << endl;
      abort(); }
    
    // tags on m at the end (so n+1 total offsets)
    auto offsets = parlay::tabulate(n+1, [&] (size_t i) -> intE {
	return (i == n) ? m : parlay::chars_to_long(W[i+3]);});
    auto edges = parlay::tabulate(m, [&] (size_t i) -> intV {
	return parlay::chars_to_long(W[n+i+3]);});

    return graph<intV, intE>(std::move(offsets), std::move(edges), n);
  }

  // parlay::sequence<char> mmapStringFromFile(const char *filename) {
  //   struct stat sb;
  //   int fd = open(filename, O_RDONLY);
  //   if (fd == -1) {
  //     perror("open");
  //     exit(-1);
  //   }
  //   if (fstat(fd, &sb) == -1) {
  //     perror("fstat");
  //     exit(-1);
  //   }
  //   if (!S_ISREG (sb.st_mode)) {
  //     perror("not a file\n");
  //     exit(-1);
  //   }
  //   char *p = static_cast<char*>(mmap(0, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
  //   if (p == MAP_FAILED) {
  //     perror("mmap");
  //     exit(-1);
  //   }
  //   if (close(fd) == -1) {
  //     perror("close");
  //     exit(-1);
  //   }
  //   size_t n = sb.st_size;
  //   return parlay::sequence<char>(p, n); // Yikes!
  // }

  // template <class intV, class intV>
  // graphC<intV, intV> readGraphCFromFile(char* fname, bool mmap=false) {

  //   parlay::sequence<char*> W;
  //   if (mmap) {
  //     cout << "mmapping file" << endl;
  //     parlay::sequence<char> S = mmapStringFromFile(fname);
  //     // copy to new sequence
  //     parlay::sequence<char> bytes = S;
  //     // and unmap
  //     if (munmap(S.begin(), S.size()) == -1) {
  //       perror("munmap");
  //       exit(-1);
  //     }
  //     W = stringToWords(S);
  //     cout << "mmap'd" << endl;
  //   } else {
  //     auto S = readStringFromFile(fname);
  //     W = stringToWords(S);
  //   }

  //   if (W[0] != AdjGraphHeader) {
  //     cout << "Bad input file: missing header: " << AdjGraphHeader << endl;
  //     abort();
  //   }

  //   // num vertices, num edges, edge offsets, edge pointers
  //   long len = W.size() -1;
  //   long n = atol(W[1]);
  //   long m = atol(W[2]);
  //   if (len != n + m + 2) {
  //     cout << "Bad input file: length = "<<len<< " n+m+2 = " << n+m+2 << endl;
  //     abort();
  //   }
  //   sequence<intV> offsets(n+1, [&] (size_t i) {
  // 	return (i == n) ? m : atol(W[i+3]);});
  //   sequence<intV> edges(m, [&] (size_t i) {
  // 	return atol(W[n+i+3]);});

  //   return graphC<intV,intV>(offsets,edges,n,m);
  // }

  template <class intV, class Weight, class intE>
  wghGraph<intV, Weight, intE> readWghGraphFromFile(char* fname) {
    parlay::sequence<char> S = readStringFromFile(fname);
    parlay::sequence<char*> W = stringToWords(S);
    if (W[0] != WghAdjGraphHeader) {
      cout << "Bad input file" << endl;
      abort();
    }

    long n = atol(W[1]);
    long m = atol(W[2]);
    if (W.size() != n + 2*m + 3) {
      cout << "Bad input file: length = "<< W.size()
	   << " n + 2*m + 3 = " << n+2*m+3 << endl;
      abort(); }
    
    // tags on m at the end (so n+1 total offsets)
    auto offsets = parlay::tabulate(n+1, [&] (size_t i) -> intE {
	return (i == n) ? m : atol(W[i+3]);});
    auto edges = parlay::tabulate(m, [&] (size_t i) -> intV {
	return atol(W[n+i+3]);});
    auto weights = parlay::tabulate(m, [&] (size_t i) -> Weight {
	return (Weight) atof(W[n+i+3+m]);});

    return wghGraph<intV,Weight,intE>(std::move(offsets),
				      std::move(edges),
				      std::move(weights), n);
  }

  // The following two are used by the graph generators to write out in either format
  // and either with reordering or not
  template <class intV, class intE>
  void writeGraphFromAdj(graph<intV,intE> const &G,
			 char* fname, bool adjArray, bool ordered) {
    if (adjArray)
      if (ordered) writeGraphToFile(G, fname);
      else writeGraphToFile(graphReorder(G), fname);
    else {
      if (ordered)
	writeEdgeArrayToFile(edgesFromGraph(G), fname);
      else {
	auto B = edgesFromGraph(graphReorder(G));
	B = randomShuffle(B);
	writeEdgeArrayToFile(B, fname);
      }
    }
  }

  template <class intV, class intE=intV>
  void writeGraphFromEdges(edgeArray<intV> &EA, char* fname, bool adjArray, bool ordered) {
    writeGraphFromAdj(graphFromEdges<intV,intE>(EA, adjArray),
		      fname, adjArray, ordered);
  }

  // void errorOut(const char* s) {
  //   cerr << s << endl;
  //   throw s;
  // }

  // void packInt64(int64_t x, uint8_t buf[8]) {
  //   uint64_t xu = x;
  //   for (int i = 0; i < 8; ++i)
  //     buf[i] = (xu >> (8 * i)) & 0xff;
  // }
  // int64_t unpackInt64(const uint8_t buf[8]) {
  //   uint64_t xu = 0;
  //   for (int i = 0; i < 8; ++i)
  //     xu |= ((uint64_t)buf[i]) << (i * 8);
  //   return (int64_t)xu;
  // }

  // void writeInt(ostream& out, char buf[8], int64_t x) {
  //   packInt64(x, (uint8_t*)buf);
  //   out.write(buf, 8);
  // }
  // int64_t readInt(istream& in, char buf[8]) {
  //   in.read(buf, 8);
  //   return unpackInt64((uint8_t*)buf);
  // }

  // template<typename intV>
  // void writeFlowGraph(ostream& out, FlowGraph<intV> g) {
  //   char buf[8];
  //   out.write("FLOWFLOW", 8);
  //   writeInt(out, buf, g.g.n);
  //   writeInt(out, buf, g.g.m);
  //   writeInt(out, buf, g.source);
  //   writeInt(out, buf, g.sink);
  //   intV o = 0;
  //   for (intV i = 0; i < g.g.n; ++i) {
  //     writeInt(out, buf, o);
  //     o += g.g.V[i].degree;
  //   }
  //   for (intV i = 0; i < g.g.n; ++i) {
  //     wghVertex<intV>& v = g.g.V[i];
  //     for (intV j = 0; j < v.degree; ++j) {
  //       writeInt(out, buf, v.Neighbors[j]);
  //       writeInt(out, buf, v.nghWeights[j]);
  //     }
  //   }
  // }
  // template<typename intV>
  // FlowGraph<intV> readFlowGraph(istream& in) {
  //   char buf[10];
  //   in.read(buf, 8);
  //   buf[8] = 0;
  //   if (strcmp(buf, "FLOWFLOW"))
  //     errorOut("Invalid flow graph input file");
  //   intV n = readInt(in, buf);
  //   intV m = readInt(in, buf);
  //   intV S = readInt(in, buf);
  //   intV T = readInt(in, buf);
  //   intV *offset = newA(intV, n);
  //   intV* adj = newA(intV, m);
  //   intV* weights = newA(intV, m);
  //   wghVertex<intV>* v = newA(wghVertex<intV>, n);
  //   for (intV i = 0; i < n; ++i) {
  //     offset[i] = readInt(in, buf);
  //     v[i].Neighbors = adj + offset[i];
  //     v[i].nghWeights = weights + offset[i];
  //     if (i > 0)
  //       v[i - 1].degree = offset[i] - offset[i - 1];
  //   }
  //   v[n - 1].degree = m - offset[n - 1];
  //   free(offset);
  //   for (intV i = 0; i < m; ++i) {
  //     adj[i] = readInt(in, buf);
  //     weights[i] = readInt(in, buf);
  //   }
  //   return FlowGraph<intV>(wghGraph<intV>(v, n, m, adj, weights), S, T);
  // }

  // const char nl = '\n';
  // template <typename intV>
  // FlowGraph<intV> writeFlowGraphDimacs(ostream& out, FlowGraph<intV> g) {
  //   out << "c DIMACS flow network description" << nl;
  //   out << "c (problem-id, nodes, arcs)" << nl;
  //   out << "p max " << g.g.n << " " << g.g.m << nl;

  //   out << "c source" << nl;
  //   out << "n " << g.source + 1 << " s" << nl;
  //   out << "c sink" << nl;
  //   out << "n " << g.sink + 1 << " t" << nl;

  //   out << "c arc description (from, to, capacity)" << nl;

  //   for (intV i = 0; i < g.g.n; ++i) {
  //     wghVertex<intV>& v = g.g.V[i];
  //     for (intV j = 0; j < v.degree; ++j) {
  //       out << "a " << i + 1 << " " << v.Neighbors[j] + 1 << " "
  //           << v.nghWeights[j] << nl;
  //     }
  //   }
  // }

  // template<typename intV>
  // struct intWghEdge {
  //   intV from, to, w;
  // };
  // int readDimacsLinePref(istream& in, const char* expected) {
  //   char type;
  //   while (in >> type) {
  //     if (type == 'c') {
  //       while (in.peek() != EOF && in.peek() != '\n')
  //         in.ignore();
  //       in >> ws;
  //       continue;
  //     } else if (!strchr(expected, type)) {
  //       errorOut((string("Unexpected DIMACS line (expected 'c' or one of '")
  // 		  + expected + "')").c_str());
  //     }
  //     return type;
  //   }
  //   return EOF;
  // }

  // template <typename intV>
  // FlowGraph<intV> readFlowGraphDimacs(istream& in) {
  //   string tmp;
  //   intV n, m;
  //   int type = readDimacsLinePref(in, "p");
  //   if (type == EOF)
  //     errorOut("Unexpected EOF while reading DIMACS file");
  //   in >> tmp >> n >> m;
  //   intWghEdge<intV>* edges = newA(intWghEdge<intV>, m);
  //   intV edgei = 0;
  //   intV* pos = newA(intV, n + 1);
  //   intV S = -1, T = -1;
  //   while (EOF != (type = readDimacsLinePref(in, "an"))) {
  //     if (type == 'n') {
  //       intV x;
  //       char st;
  //       in >> x >> st;
  //       x--;
  //       if (st == 's') S = x;
  //       else T = x;
  //     } else { // type == 'a'
  //       intV from, to, cap;
  //       in >> from >> to >> cap;
  //       from--; to--;
  //       edges[edgei] = (intWghEdge<intV>) { from, to, cap };
  //       edgei++;
  //       pos[from + 1]++;
  //     }
  //   }
  //   if (S < 0)
  //     errorOut("No source was specified in DIMACS input file");
  //   if (T < 0)
  //     errorOut("No sink was specified in DIMACS input file");
  //   if (m != edgei)
  //     errorOut("Inconsistent edge count in DIMACS input file");
  //   intV* adj = newA(intV, m);
  //   intV* weights = newA(intV, m);
  //   wghVertex<intV>* v = newA(wghVertex<intV>, n);
  //   for (intV i = 0; i < n; ++i) {
  //     pos[i + 1] += pos[i];
  //     v[i].Neighbors = adj + pos[i];
  //     v[i].nghWeights = weights + pos[i];
  //     v[i].degree = pos[i + 1] - pos[i];
  //   }
  //   for (intV i = 0; i < m; ++i) {
  //     intV& p = pos[edges[i].from];
  //     adj[p] = edges[i].to;
  //     weights[p] = edges[i].w;
  //     p++;
  //   }
  //   free(edges);
  //   free(pos);
  //   return FlowGraph<intV>(wghGraph<intV>(v, n, m, adj, weights), S, T);
  // }
};
