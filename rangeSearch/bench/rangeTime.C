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
#include "../../algorithms/bench/parse_command_line.h"
#include "../../algorithms/bench/time_loop.h"
#include "../utils/NSGDist.h"
#include "../utils/euclidian_point.h"
#include "../utils/point_range.h"
#include "../utils/mips_point.h"
#include "../utils/graph.h"



#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


using namespace parlayANN;

// *************************************************************
//  TIMING
// *************************************************************

using uint = unsigned int;


template<typename Point, typename PointRange, typename indexType>
void timeRange(Graph<indexType> &G,
		   PointRange &Query_Points, double rad, double esr,
		   BuildParams &BP, char* outFile,
		   RangeGroundTruth<indexType> GT, char* res_file, bool graph_built, PointRange &Points, bool is_early_stop, bool is_double_beam)
{


    time_loop(1, 0,
      [&] () {},
      [&] () {
        RNG<Point, PointRange, indexType>(G, rad, esr, BP, Query_Points, GT, res_file, graph_built, Points, is_early_stop, is_double_beam);
      },
      [&] () {});

    if(outFile != NULL) {
      G.save(outFile);
    }


}

int main(int argc, char* argv[]) {
    commandLine P(argc,argv,
    "[-a <alpha>] [-d <delta>] [-R <deg>]"
        "[-L <bm>] [-r <rad> ]  [-gt_path <g>] [-query_path <qF>]"
        "[-graph_path <gF>] [-graph_outfile <oF>] [-res_path <rF>]"
        "[-memory_flag <algoOpt>] [-mst_deg <q>] [num_clusters <nc>] [cluster_size <cs>]"
        "[-data_type <tp>] [-dist_func <df>][-base_path <b>][-search_mode <sm>] <inFile>");

  char* iFile = P.getOptionValue("-base_path");
  char* oFile = P.getOptionValue("-graph_outfile");
  char* gFile = P.getOptionValue("-graph_path");
  char* qFile = P.getOptionValue("-query_path");
  char* cFile = P.getOptionValue("-gt_path");
  char* rFile = P.getOptionValue("-res_path");
  char* vectype = P.getOptionValue("-data_type");
  long R = P.getOptionIntValue("-R", 0);
  if(R<0) P.badArgument();
  long L = P.getOptionIntValue("-L", 0);
  if(L<0) P.badArgument();
  long MST_deg = P.getOptionIntValue("-mst_deg", 0);
  if(MST_deg < 0) P.badArgument();
  long num_clusters = P.getOptionIntValue("-num_clusters", 0);
  if(num_clusters<0) P.badArgument();
  long cluster_size = P.getOptionIntValue("-cluster_size", 0);
  if(cluster_size<0) P.badArgument();
  double r = P.getOptionDoubleValue("-r", 0);
  double alpha = P.getOptionDoubleValue("-alpha", 0);
  int two_pass = P.getOptionIntValue("-two_pass", 1);
  bool pass = (two_pass == 1);
  double delta = P.getOptionDoubleValue("-delta", 0);
  if(delta<0) P.badArgument();
  char* dfc = P.getOptionValue("-dist_func");
  char* sm = P.getOptionValue("-search_mode");
  double esr = P.getOptionDoubleValue("-early_stopping_radius", 0);

  std::string df = std::string(dfc);
  std::string tp = std::string(vectype);

  std::string searchType = std::string(sm);
  bool is_early_stop = false;
  bool is_double_beam = false;

  if(searchType == "earlyStopping"){
    is_early_stop = true;
  }
  if(searchType == "doublingSearch"){
    is_double_beam = true;
  }

  BuildParams BP = BuildParams(R, L, alpha, two_pass, num_clusters, cluster_size, MST_deg, delta);
  long maxDeg = BP.max_degree();

  if((tp != "uint8") && (tp != "int8") && (tp != "float")){
    std::cout << "Error: vector type not specified correctly, specify int8, uint8, or float" << std::endl;
    abort();
  }

  if(df != "Euclidian" && df != "mips"){
    std::cout << "Error: specify distance type Euclidian or mips" << std::endl;
    abort();
  }

  bool graph_built = (gFile != NULL);

  RangeGroundTruth<uint> GT = RangeGroundTruth<uint>(cFile);
  
  if(tp == "float"){
    if(df == "Euclidian"){
      PointRange<Euclidian_Point<float>> Points = PointRange<Euclidian_Point<float>>(iFile);
      PointRange<Euclidian_Point<float>> Query_Points = PointRange<Euclidian_Point<float>>(qFile);
      Graph<unsigned int> G; 
      if(gFile == NULL) G = Graph<unsigned int>(maxDeg, Points.size());
      else G = Graph<unsigned int>(gFile);
      timeRange<Euclidian_Point<float>, PointRange<Euclidian_Point<float>>, uint>(G, Query_Points, r, esr, BP, 
        oFile, GT, rFile, graph_built, Points, is_early_stop, is_double_beam);
    } else if(df == "mips"){
      PointRange<Mips_Point<float>> Points = PointRange<Mips_Point<float>>(iFile);
      PointRange<Mips_Point<float>> Query_Points = PointRange<Mips_Point<float>>(qFile);
      Graph<unsigned int> G; 
      if(gFile == NULL) G = Graph<unsigned int>(maxDeg, Points.size());
      else G = Graph<unsigned int>(gFile);
      timeRange<Mips_Point<float>, PointRange<Mips_Point<float>>, uint>(G, Query_Points, r, esr, BP, 
        oFile, GT, rFile, graph_built, Points, is_early_stop, is_double_beam);
    }
    
  } else if(tp == "uint8"){
    if(df == "Euclidian"){
      PointRange<Euclidian_Point<uint8_t>> Points = PointRange<Euclidian_Point<uint8_t>>(iFile);
      PointRange<Euclidian_Point<uint8_t>> Query_Points = PointRange<Euclidian_Point<uint8_t>>(qFile);
      Graph<unsigned int> G; 
      if(gFile == NULL) G = Graph<unsigned int>(maxDeg, Points.size());
      else G = Graph<unsigned int>(gFile);
      timeRange<Euclidian_Point<uint8_t>, PointRange<Euclidian_Point<uint8_t>>, uint>(G, Query_Points, r, esr, BP, 
        oFile, GT, rFile, graph_built, Points, is_early_stop, is_double_beam);
    } else if(df == "mips"){
      PointRange< Mips_Point<uint8_t>> Points = PointRange< Mips_Point<uint8_t>>(iFile);
      PointRange< Mips_Point<uint8_t>> Query_Points = PointRange< Mips_Point<uint8_t>>(qFile);
      Graph<unsigned int> G; 
      if(gFile == NULL) G = Graph<unsigned int>(maxDeg, Points.size());
      else G = Graph<unsigned int>(gFile);
      timeRange<Mips_Point<uint8_t>, PointRange< Mips_Point<uint8_t>>, uint>(G, Query_Points, r, esr, BP, 
        oFile, GT, rFile, graph_built, Points, is_early_stop, is_double_beam);
    }
  } else if(tp == "int8"){
    if(df == "Euclidian"){
      PointRange<Euclidian_Point<int8_t>> Points = PointRange<Euclidian_Point<int8_t>>(iFile);
      PointRange<Euclidian_Point<int8_t>> Query_Points = PointRange< Euclidian_Point<int8_t>>(qFile);
      Graph<unsigned int> G; 
      if(gFile == NULL) G = Graph<unsigned int>(maxDeg, Points.size());
      else G = Graph<unsigned int>(gFile);
      timeRange<Euclidian_Point<int8_t>, PointRange<Euclidian_Point<int8_t>>, uint>(G, Query_Points, r, esr, BP,
        oFile, GT, rFile, graph_built, Points, is_early_stop, is_double_beam);
    } else if(df == "mips"){
      PointRange<Mips_Point<int8_t>> Points = PointRange<Mips_Point<int8_t>>(iFile);
      PointRange<Mips_Point<int8_t>> Query_Points = PointRange<Mips_Point<int8_t>>(qFile);
      Graph<unsigned int> G; 
      if(gFile == NULL) G = Graph<unsigned int>(maxDeg, Points.size());
      else G = Graph<unsigned int>(gFile);
      timeRange<Mips_Point<int8_t>, PointRange< Mips_Point<int8_t>>, uint>(G, Query_Points, r, esr, BP,
        oFile, GT, rFile, graph_built, Points, is_early_stop, is_double_beam);
    }
  }
  
}