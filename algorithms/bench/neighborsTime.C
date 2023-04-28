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
// #include "common/geometry.h"
// #include "common/geometryIO.h"
#include "parse_command_line.h"
#include "time_loop.h"
#include "../utils/parse_files.h"
#include "../utils/NSGDist.h"



#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// using namespace benchIO;

bool report_stats = true;


// *************************************************************
//  TIMING
// *************************************************************

template<typename T>
void timeNeighbors(parlay::sequence<Tvec_point<T>> &pts,
  int rounds, int R, int beamSize, double delta, double alpha, char* outFile, int maxDeg, bool graph_built, Distance* D)
{
  size_t n = pts.size();
  auto v = parlay::tabulate(n, [&] (size_t i) -> Tvec_point<T>* {
      return &pts[i];});

  time_loop(rounds, 0,
  [&] () {},
  [&] () {
    ANN<T>(v, R, beamSize, alpha, delta, graph_built, D);
  },
  [&] () {});

  if(outFile != NULL) {
    std::cout << "Writing graph..."; 
    write_graph(v, outFile, maxDeg); 
    std::cout << " done" << std::endl;
  }


}

template<typename T>
void timeNeighbors(parlay::sequence<Tvec_point<T>> &pts,
		   parlay::sequence<Tvec_point<T>> &qpoints,
		   int k, int rounds, int R, int beamSize,
		   int beamSizeQ, double delta, double alpha, char* outFile,
		   parlay::sequence<ivec_point>& groundTruth, int maxDeg, char* res_file, bool graph_built, Distance* D)
{
  size_t n = pts.size();
  auto v = parlay::tabulate(n, [&] (size_t i) -> Tvec_point<T>* {
      return &pts[i];});

  size_t q = qpoints.size();
  auto qpts =  parlay::tabulate(q, [&] (size_t i) -> Tvec_point<T>* {
      return &qpoints[i];});

    time_loop(rounds, 0,
      [&] () {},
      [&] () {
        ANN<T>(v, k, R, beamSize, beamSizeQ, alpha, delta, qpts, groundTruth, res_file, graph_built, D);
      },
      [&] () {});

    if(outFile != NULL) {
      std::cout << "Writing graph..."; 
      write_graph(v, outFile, maxDeg); 
      std::cout << " done" << std::endl;
    }


}

// Infile is a file in .fvecs format
int main(int argc, char* argv[]) {
    commandLine P(argc,argv,
    "[-a <alpha>] [-d <delta>] [-R <deg>]"
        "[-L <bm>] [-k <k> ] [-Q <bmq>] [-q <qF>]"
        "[-g <gF>] [-o <oF>] [-res <rF>] [-r <rnds>] [-b <algoOpt>] [-f <ft>] [-t <tp>] [-D <df>] <inFile>");

  char* iFile = P.getArgument(0);
  char* oFile = P.getOptionValue("-o");
  char* gFile = P.getOptionValue("-g");
  char* qFile = P.getOptionValue("-q");
  char* cFile = P.getOptionValue("-c");
  char* rFile = P.getOptionValue("-res");
  char* filetype = P.getOptionValue("-f");
  char* vectype = P.getOptionValue("-t");
  int R = P.getOptionIntValue("-R", 5);
  if (R < 1) P.badArgument();
  int L = P.getOptionIntValue("-L", 10);
  if (L < 1) P.badArgument();
  int Q = P.getOptionIntValue("-Q", L);
  if (Q < 1) P.badArgument();
  int rounds = P.getOptionIntValue("-r", 1);
  int k = P.getOptionIntValue("-k", 1);
  if (k > 1000 || k < 1) P.badArgument();
  double alpha = P.getOptionDoubleValue("-a", 1.2);
  double delta = P.getOptionDoubleValue("-d", .01);
  int algoOpt = P.getOptionIntValue("-b", 0);
  char* dfc = P.getOptionValue("-D");

  std::string df = std::string(dfc);
  Distance* D;
  if(df == "Euclidian") D = new Euclidian_Distance();
  else if(df == "mips") D = new Mips_Distance();
  else{
    std::cout << "Error: invalid distance type" << std::endl;
    abort();
  }

  std::string ft = std::string(filetype);
  std::string tp = std::string(vectype);

  if((ft != "bin") && (ft != "vec")){
    std::cout << "Error: file type not specified correctly, specify bin or vec" << std::endl;
    abort();
  }

  if((tp != "uint8") && (tp != "int8") && (tp != "float")){
    std::cout << "Error: vector type not specified correctly, specify int8, uint8, or float" << std::endl;
    abort();
  }

  if((ft == "vec") && (tp == "int8")){
    std::cout << "Error: incompatible file and vector types" << std::endl;
    abort();
  }

  parlay::sequence<ivec_point> groundTruth;

  int maxDeg;
  if(algoOpt == 1) maxDeg = L*R;
  else if(algoOpt == 2) maxDeg = 2*R;
  else maxDeg = R;

  bool graph_built = (gFile != NULL);

  if(ft == "vec"){
    if(cFile != NULL) groundTruth = parse_ivecs(cFile);
    if(tp == "float"){
      auto [md, points] = parse_fvecs(iFile, gFile, maxDeg);
      maxDeg = md;
      if(qFile != NULL){
        auto [fd, qpoints] = parse_fvecs(qFile, NULL, 0);
        timeNeighbors<float>(points, qpoints, k, rounds, R, L, Q,
          delta, alpha, oFile, groundTruth, maxDeg, rFile, graph_built, D);
      }
      else timeNeighbors<float>(points, rounds, R, L, delta, alpha, oFile, maxDeg, graph_built, D);
    }
    else if(tp == "uint8"){
      auto [md, points] = parse_bvecs(iFile, gFile, maxDeg);
      maxDeg = md;
      if(qFile != NULL){
        auto [fd, qpoints] = parse_bvecs(qFile, NULL, 0);
        timeNeighbors<uint8_t>(points, qpoints, k, rounds, R, L, Q,
          delta, alpha, oFile, groundTruth, maxDeg, rFile, graph_built, D);
      }
      else timeNeighbors<uint8_t>(points, rounds, R, L, delta, alpha, oFile, maxDeg, graph_built, D);
    }
  }else if(ft == "bin"){
    if(cFile != NULL) groundTruth = parse_ibin(cFile);
    if(tp == "float"){
      auto [md, points] = parse_fbin(iFile, gFile, maxDeg);
      maxDeg = md;
      if(qFile != NULL){
        auto [fd, qpoints] = parse_fbin(qFile, NULL, 0);
        timeNeighbors<float>(points, qpoints, k, rounds, R, L, Q,
          delta, alpha, oFile, groundTruth, maxDeg, rFile, graph_built, D);
      }
      else timeNeighbors<float>(points, rounds, R, L, delta, alpha, oFile, maxDeg, graph_built, D);
    } else if(tp == "uint8"){
      auto [md, points] = parse_uint8bin(iFile, gFile, maxDeg);
      maxDeg = md;
      if(qFile != NULL){
        auto [fd, qpoints] = parse_uint8bin(qFile, NULL, 0);
        timeNeighbors<uint8_t>(points, qpoints, k, rounds, R, L, Q,
          delta, alpha, oFile, groundTruth, maxDeg, rFile, graph_built, D);
      }
      else timeNeighbors<uint8_t>(points, rounds, R, L, delta, alpha, oFile, maxDeg, graph_built, D);
    } else if(tp == "int8"){
      auto [md, points] = parse_int8bin(iFile, gFile, maxDeg);
      maxDeg = md;
      if(qFile != NULL){
        auto [fd, qpoints] = parse_int8bin(qFile, NULL, 0);
        timeNeighbors<int8_t>(points, qpoints, k, rounds, R, L, Q,
          delta, alpha, oFile, groundTruth, maxDeg, rFile, graph_built, D);
      }
      else timeNeighbors<int8_t>(points, rounds, R, L, delta, alpha, oFile, maxDeg, graph_built, D);
    }
  }
}



//REGULAR CORRECTNESS CHECK
// if (outFile != NULL) {
      // int m = q * (k+1);
      // parlay::sequence<int> Pout(m);
      // parlay::parallel_for (0, q, [&] (size_t i) {
      //   Pout[(k+1)*i] = qpts[i]->id;
      //   for (int j=0; j < k; j++)
      //     Pout[(k+1)*i + j+1] = (qpts[i]->ngh)[j];
      // });
      // writeIntSeqToFile(Pout, outFile);
    // }


//DIRECTED DEGREES
// if (outFile != NULL) {
      // parlay::sequence<int> degrees(n);
      // parlay::parallel_for(0, n, [&] (size_t i){
      //   degrees[i] = v[i]->out_nbh.size();
      // });
      // writeIntSeqToFile(degrees, outFile);
    // }

//COORDINATES OF FILE
// if (outFile != NULL) {
      // int d = v[0]->coordinates.size();
      // int m = n*d; //total number of ints in the file
      // parlay::sequence<int> vals(m);
      // parlay::parallel_for(0, n, [&] (size_t i){
          // for(int j=0; j<d; j++){
          //   vals[i*d+j] = (int) v[i]->coordinates[j];
          // }
      // });
      // writeIntSeqToFile(vals, outFile);
    // }

//UNDIRECTED DEGREES (REALLY SLOW AND BAD)
 // if (outFile != NULL) {
 //      parlay::sequence<int> udegrees(n);
 //      parlay::sequence<parlay::sequence<index_pair>> to_flatten = parlay::sequence<parlay::sequence<index_pair>>(n);
 //      parlay::parallel_for(0, n, [&] (size_t i){
 //        size_t m = v[i]->out_nbh.size();
 //        parlay::sequence<index_pair> edges = parlay::sequence<index_pair>(2*m);
 //        for(int j=0; j<m; j++){
 //          edges[2*j] = std::make_pair(i, v[i]->out_nbh[j]);
 //          edges[2*j+1] = std::make_pair(v[i]->out_nbh[j], i);
 //        }
 //        to_flatten[i] = edges;
 //      });
 //      std::cout << "here1" << std::endl;
 //    auto edges_unsorted = parlay::flatten(to_flatten);
 //    std::cout << "here2" << std::endl;
 //    auto grouped_edges = parlay::group_by_key(edges_unsorted);
 //    std::cout << "here3" << std::endl;
 //    parlay::parallel_for(0, n, [&] (size_t i){
 //      int count = 0;
 //      // parlay::sequence<int> edge_ids = grouped_edges[i].second;
 //      std::cout << grouped_edges[i].second.size() << std::endl;
 //      for(int j=0; j<grouped_edges[i].second.size(); j++){
 //        count+=1;

 //        if(j<grouped_edges[i].second.size()-1){
 //          int current = grouped_edges[i].second[j];
 //          int next = grouped_edges[i].second[j+1];
 //          if(current == next) j+=1;
 //        }

 //      }
 //      udegrees[i] = count;
 //    });
 //    auto sortedDegrees = parlay::sort(udegrees);
 //    writeIntSeqToFile(sortedDegrees, outFile);
 //    }

  //GRAPH FORMAT

   // if (outFile != NULL) {
   //    parlay::sequence<int> graph(n*(maxDeg+1));
   //    parlay::parallel_for(0, n, [&] (size_t i){
   //      graph[i*(maxDeg+1)] = (int) i;
   //      int degree = v[i]->out_nbh.size();
   //      for(int j=0; j<degree; j++) graph[i*(maxDeg+1)+1+j] = v[i]->out_nbh[j];
   //      int rem = maxDeg - degree;
   //      for(int j=0; j<rem; j++) graph[i*(maxDeg+1)+1+degree+j] = -1;
   //    });
   //    writeIntSeqToFile(graph, outFile);
   //  }




