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

#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parse_command_line.h"
#include "time_loop.h"
#include "../utils/parse_files.h"
#include "../utils/NSGDist.h"



#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>




// *************************************************************
//  TIMING
// *************************************************************


template<typename T>
void timeNeighbors(parlay::sequence<Tvec_point<T>> &pts,
		   parlay::sequence<Tvec_point<T>> &qpoints,
		   int k, int R, int beamSize,
		   int beamSizeQ, double delta, double alpha, char* outFile,
		   parlay::sequence<ivec_point>& groundTruth, int maxDeg, char* res_file, bool graph_built, Distance* D, data_store<T> &Data)
{
  size_t n = pts.size();
  auto v = parlay::tabulate(n, [&] (size_t i) -> Tvec_point<T>* {
      return &pts[i];});

  size_t q = qpoints.size();
  auto qpts =  parlay::tabulate(q, [&] (size_t i) -> Tvec_point<T>* {
      return &qpoints[i];});

    time_loop(1, 0,
      [&] () {},
      [&] () {
        ANN<T>(v, k, R, beamSize, beamSizeQ, alpha, delta, qpts, groundTruth, res_file, graph_built, D, Data);
      },
      [&] () {});

    if(outFile != NULL) {
      write_graph(v, outFile, maxDeg); 
    }


}

int main(int argc, char* argv[]) {
    commandLine P(argc,argv,
    "[-a <alpha>] [-d <delta>] [-R <deg>]"
        "[-L <bm>] [-k <k> ] [-L_range <bmq>] [-gt_path <g>] [-query_path <qF>]"
        "[-graph_path <gF>] [-graph_outfile <oF>] [-res_path <rF>]"
        "[-memory_flag <algoOpt>] [-Q <q>]"
        "[-data_type <tp>] [-dist_func <df>][-base_path <b>] <inFile>");

  char* iFile = P.getOptionValue("-base_path");
  char* oFile = P.getOptionValue("-graph_outfile");
  char* gFile = P.getOptionValue("-graph_path");
  char* qFile = P.getOptionValue("-query_path");
  char* cFile = P.getOptionValue("-gt_path");
  char* rFile = P.getOptionValue("-res_path");
  char* vectype = P.getOptionValue("-data_type");
  int R = P.getOptionIntValue("-R", 5);
  int L = P.getOptionIntValue("-L", 10);
  int Q = P.getOptionIntValue("-Q", 10);
  int k = P.getOptionIntValue("-k", 1);
  if (k > 1000 || k < 1) P.badArgument();
  double alpha = P.getOptionDoubleValue("-a", 1.2);
  double delta = P.getOptionDoubleValue("-d", .01);
  int algoOpt = P.getOptionIntValue("-memory_flag", 0);
  char* dfc = P.getOptionValue("-dist_func");

  std::string df = std::string(dfc);
  Distance* D;
  if(df == "Euclidian") D = new Euclidian_Distance();
  else if(df == "mips") D = new Mips_Distance();
  else{
    std::cout << "Error: invalid distance type" << std::endl;
    abort();
  }

  std::string tp = std::string(vectype);


  if((tp != "uint8") && (tp != "int8") && (tp != "float")){
    std::cout << "Error: vector type not specified correctly, specify int8, uint8, or float" << std::endl;
    abort();
  }

  parlay::sequence<ivec_point> groundTruth;

  int maxDeg;
  if(algoOpt == 1) maxDeg = L*R;
  else if(algoOpt == 2) maxDeg = 2*R;
  else maxDeg = R;

  bool graph_built = (gFile != NULL);


  if(cFile != NULL) groundTruth = parse_ibin(cFile);
  if(tp == "float"){
    data_store<float> DS = store_fbin(iFile, D, alpha);
    auto [md, points] = parse_fbin(iFile, gFile, maxDeg);
    maxDeg = md;
    parlay::sequence<Tvec_point<float>> qpoints;
    if(qFile != NULL){qpoints = parse_fbin(qFile, NULL, 0).second;}
    timeNeighbors<float>(points, qpoints, k, R, L, Q,
        delta, alpha, oFile, groundTruth, maxDeg, rFile, graph_built, D, DS);
  } else if(tp == "uint8"){
    data_store<uint8_t> DS = store_uint8bin(iFile, D, alpha);
    auto [md, points] = parse_uint8bin(iFile, gFile, maxDeg);
    maxDeg = md;
    parlay::sequence<Tvec_point<uint8_t>> qpoints;
    if(qFile != NULL){qpoints = parse_uint8bin(qFile, NULL, 0).second;}
    timeNeighbors<uint8_t>(points, qpoints, k, R, L, Q,
        delta, alpha, oFile, groundTruth, maxDeg, rFile, graph_built, D, DS);
  } else if(tp == "int8"){
    data_store<int8_t> DS = store_int8bin(iFile, D, alpha);
    auto [md, points] = parse_int8bin(iFile, gFile, maxDeg);
    maxDeg = md;
    parlay::sequence<Tvec_point<int8_t>> qpoints;
    if(qFile != NULL){qpoints = parse_int8bin(qFile, NULL, 0).second;}
    timeNeighbors<int8_t>(points, qpoints, k, R, L, Q,
        delta, alpha, oFile, groundTruth, maxDeg, rFile, graph_built, D, DS);
  }
  
}