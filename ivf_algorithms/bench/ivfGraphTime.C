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
#include "utils/NSGDist.h"
#include "utils/euclidian_point.h"
#include "utils/point_range.h"
#include "utils/mips_point.h"



#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>




// *************************************************************
//  TIMING
// *************************************************************

using uint = unsigned int;


template<typename Point, typename PointRange, typename indexType>
void timeIVF(
		   PointRange &Query_Points, long k,
		   BuildParams &BP, char* outFile,
		   groundTruth<indexType> GT, char* res_file, bool graph_built, PointRange &Points, char* gFile, char* oFile)
{


    time_loop(1, 0,
      [&] () {},
      [&] () {
        IVFGraph<Point, PointRange, indexType>(k, BP, Query_Points, GT, res_file, graph_built, Points, gFile, oFile);
      },
      [&] () {});



}

int main(int argc, char* argv[]) {
    commandLine P(argc,argv,
    "[-a <alpha>] [-d <delta>] [-R <deg>]"
        "[-L <bm>] [-k <k> ]  [-gt_path <g>] [-query_path <qF>]"
        "[-cluster_path <gF>] [-cluster_outfile <oF>] [-res_path <rF>]"
        "[-memory_flag <algoOpt>] [-mst_deg <q>] [num_clusters <nc>] [cluster_size <cs>]"
        "[-data_type <tp>] [-dist_func <df>][-base_path <b>] <inFile>");

  char* iFile = P.getOptionValue("-base_path");
  char* oFile = P.getOptionValue("-cluster_outfile");
  char* gFile = P.getOptionValue("-cluster_path");
  char* qFile = P.getOptionValue("-query_path");
  char* cFile = P.getOptionValue("-gt_path");
  char* rFile = P.getOptionValue("-res_path");
  char* vectype = P.getOptionValue("-data_type");
  long num_clusters = P.getOptionIntValue("-num_clusters", 0);
  if(num_clusters<0) P.badArgument();
  long k = P.getOptionIntValue("-k", 0);
  if (k > 1000 || k < 0) P.badArgument();
  char* dfc = P.getOptionValue("-dist_func");

  std::string df = std::string(dfc);
  std::string tp = std::string(vectype);

  //TODO add constructor
  BuildParams BP = BuildParams(num_clusters);

  if((tp != "uint8") && (tp != "int8") && (tp != "float")){
    std::cout << "Error: vector type not specified correctly, specify int8, uint8, or float" << std::endl;
    abort();
  }

  if(df != "Euclidian" && df != "mips"){
    std::cout << "Error: specify distance type Euclidian or mips" << std::endl;
    abort();
  }


  groundTruth<uint> GT = groundTruth<uint>(cFile);
  bool cluster_built = false;
  
  if(tp == "float"){
    if(df == "Euclidian"){
      PointRange<float, Euclidian_Point<float>> Points = PointRange<float, Euclidian_Point<float>>(iFile);
      PointRange<float, Euclidian_Point<float>> Query_Points = PointRange<float, Euclidian_Point<float>>(qFile);
      timeIVF<Euclidian_Point<float>, PointRange<float, Euclidian_Point<float>>, uint>(Query_Points, k, BP, 
        oFile, GT, rFile, cluster_built, Points, gFile, oFile);
    } else if(df == "mips"){
      PointRange<float, Mips_Point<float>> Points = PointRange<float, Mips_Point<float>>(iFile);
      PointRange<float, Mips_Point<float>> Query_Points = PointRange<float, Mips_Point<float>>(qFile);
      timeIVF<Mips_Point<float>, PointRange<float, Mips_Point<float>>, uint>(Query_Points, k, BP, 
        oFile, GT, rFile, cluster_built, Points, gFile, oFile);
    }
    
  } else if(tp == "uint8"){
    if(df == "Euclidian"){
      PointRange<uint8_t, Euclidian_Point<uint8_t>> Points = PointRange<uint8_t, Euclidian_Point<uint8_t>>(iFile);
      PointRange<uint8_t, Euclidian_Point<uint8_t>> Query_Points = PointRange<uint8_t, Euclidian_Point<uint8_t>>(qFile);
      timeIVF<Euclidian_Point<uint8_t>, PointRange<uint8_t, Euclidian_Point<uint8_t>>, uint>(Query_Points, k, BP, 
        oFile, GT, rFile, cluster_built, Points, gFile, oFile);
    } else if(df == "mips"){
      PointRange<uint8_t, Mips_Point<uint8_t>> Points = PointRange<uint8_t, Mips_Point<uint8_t>>(iFile);
      PointRange<uint8_t, Mips_Point<uint8_t>> Query_Points = PointRange<uint8_t, Mips_Point<uint8_t>>(qFile);
      timeIVF<Mips_Point<uint8_t>, PointRange<uint8_t, Mips_Point<uint8_t>>, uint>(Query_Points, k, BP, 
        oFile, GT, rFile, cluster_built, Points, gFile, oFile);
    }
  } else if(tp == "int8"){
    if(df == "Euclidian"){
      PointRange<int8_t, Euclidian_Point<int8_t>> Points = PointRange<int8_t, Euclidian_Point<int8_t>>(iFile);
      PointRange<int8_t, Euclidian_Point<int8_t>> Query_Points = PointRange<int8_t, Euclidian_Point<int8_t>>(qFile);
      timeIVF<Euclidian_Point<int8_t>, PointRange<int8_t, Euclidian_Point<int8_t>>, uint>(Query_Points, k, BP,
        oFile, GT, rFile, cluster_built, Points, gFile, oFile);
    } else if(df == "mips"){
      PointRange<int8_t, Mips_Point<int8_t>> Points = PointRange<int8_t, Mips_Point<int8_t>>(iFile);
      PointRange<int8_t, Mips_Point<int8_t>> Query_Points = PointRange<int8_t, Mips_Point<int8_t>>(qFile);
      timeIVF<Mips_Point<int8_t>, PointRange<int8_t, Mips_Point<int8_t>>, uint>(Query_Points, k, BP,
        oFile, GT, rFile, cluster_built, Points, gFile, oFile);
    }
  }
  
  return 0;
}
