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




// *************************************************************
//  TIMING
// *************************************************************

using uint = unsigned int;


template<typename Point, typename PointRange, typename indexType, typename GraphType>
void timeNeighbors(GraphType &Graph,
		   PointRange &Query_Points, long k,
		   BuildParams &BP, char* outFile,
		   groundTruth<indexType> GT, char* res_file, bool graph_built, PointRange &Points)
{


    time_loop(1, 0,
      [&] () {},
      [&] () {
        ANN<Point, PointRange, indexType, GraphType>(Graph, k, BP, Query_Points, GT, res_file, graph_built, Points);
      },
      [&] () {});

    if(outFile != NULL) {
      Graph.save(outFile);
    }


}

int main(int argc, char* argv[]) {
    commandLine P(argc,argv,
    "[-a <alpha>] [-d <delta>] [-R <deg>]"
        "[-L <bm>] [-k <k> ]  [-gt_path <g>] [-query_path <qF>]"
        "[-graph_path <gF>] [-graph_outfile <oF>] [-res_path <rF>]"
        "[-memory_flag <algoOpt>] [-mst_deg <q>] [num_clusters <nc>] [cluster_size <cs>]"
        "[-data_type <tp>] [-dist_func <df>][-base_path <b>] <inFile>");

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
  long k = P.getOptionIntValue("-k", 0);
  if (k > 1000 || k < 0) P.badArgument();
  double alpha = P.getOptionDoubleValue("-alpha", 1.0);
  int two_pass = P.getOptionIntValue("-two_pass", 0);
  if(two_pass > 1 | two_pass < 0) P.badArgument();
  bool pass = (two_pass == 1);
  double delta = P.getOptionDoubleValue("-delta", 0);
  if(delta<0) P.badArgument();
  char* dfc = P.getOptionValue("-dist_func");
  char* gt = P.getOptionValue("-graph_type");

  std::string df = std::string(dfc);
  std::string tp = std::string(vectype);

  BuildParams BP = BuildParams(R, L, alpha, pass, num_clusters, cluster_size, MST_deg, delta);
  long maxDeg = BP.max_degree();

  if((tp != "uint8") && (tp != "int8") && (tp != "float")){
    std::cout << "Error: vector type not specified correctly, specify int8, uint8, or float" << std::endl;
    abort();
  }

  if(df != "Euclidian" && df != "mips"){
    std::cout << "Error: specify distance type Euclidian or mips" << std::endl;
    abort();
  }

  std::string graph_type;
  if(gt == nullptr) graph_type = "flat";
  else graph_type = std::string(gt);
  if(graph_type != "flat" && graph_type != "aspen"){
    std::cout << "Error: specify graph type flat or aspen" << std::endl;
    abort();
  }



  //read the number of points in order to prepare graph 
  std::ifstream reader(iFile);
  assert(reader.is_open());
  unsigned int num_points;
  reader.read((char*)(&num_points), sizeof(unsigned int));
  reader.close();

  groundTruth<uint> GT = groundTruth<uint>(cFile);

  if(graph_type == "flat"){
    bool graph_built = (gFile != NULL);
    Flat_Graph<uint> Graph;
    if(gFile == NULL) Graph = Flat_Graph<uint>(maxDeg, num_points);
    else Graph = Flat_Graph<uint>(gFile);

    if(tp == "float"){
      if(df == "Euclidian"){
        PointRange<float, Euclidian_Point<float>> Points = PointRange<float, Euclidian_Point<float>>(iFile);
        PointRange<float, Euclidian_Point<float>> Query_Points = PointRange<float, Euclidian_Point<float>>(qFile);
        timeNeighbors<Euclidian_Point<float>, PointRange<float, Euclidian_Point<float>>, uint>(Graph, Query_Points, k, BP, 
          oFile, GT, rFile, graph_built, Points);
      } else if(df == "mips"){
        PointRange<float, Mips_Point<float>> Points = PointRange<float, Mips_Point<float>>(iFile);
        PointRange<float, Mips_Point<float>> Query_Points = PointRange<float, Mips_Point<float>>(qFile);
        timeNeighbors<Mips_Point<float>, PointRange<float, Mips_Point<float>>, uint>(Graph, Query_Points, k, BP, 
          oFile, GT, rFile, graph_built, Points);
      }
    } else if(tp == "uint8"){
      if(df == "Euclidian"){
        PointRange<uint8_t, Euclidian_Point<uint8_t>> Points = PointRange<uint8_t, Euclidian_Point<uint8_t>>(iFile);
        PointRange<uint8_t, Euclidian_Point<uint8_t>> Query_Points = PointRange<uint8_t, Euclidian_Point<uint8_t>>(qFile);
        timeNeighbors<Euclidian_Point<uint8_t>, PointRange<uint8_t, Euclidian_Point<uint8_t>>, uint>(Graph, Query_Points, k, BP, 
          oFile, GT, rFile, graph_built, Points);
      } else if(df == "mips"){
        PointRange<uint8_t, Mips_Point<uint8_t>> Points = PointRange<uint8_t, Mips_Point<uint8_t>>(iFile);
        PointRange<uint8_t, Mips_Point<uint8_t>> Query_Points = PointRange<uint8_t, Mips_Point<uint8_t>>(qFile);
        timeNeighbors<Mips_Point<uint8_t>, PointRange<uint8_t, Mips_Point<uint8_t>>, uint>(Graph, Query_Points, k, BP, 
          oFile, GT, rFile, graph_built, Points);
      }
    } else if(tp == "int8"){
      if(df == "Euclidian"){
        PointRange<int8_t, Euclidian_Point<int8_t>> Points = PointRange<int8_t, Euclidian_Point<int8_t>>(iFile);
        PointRange<int8_t, Euclidian_Point<int8_t>> Query_Points = PointRange<int8_t, Euclidian_Point<int8_t>>(qFile);
        timeNeighbors<Euclidian_Point<int8_t>, PointRange<int8_t, Euclidian_Point<int8_t>>, uint>(Graph, Query_Points, k, BP,
          oFile, GT, rFile, graph_built, Points);
      } else if(df == "mips"){
        PointRange<int8_t, Mips_Point<int8_t>> Points = PointRange<int8_t, Mips_Point<int8_t>>(iFile);
        PointRange<int8_t, Mips_Point<int8_t>> Query_Points = PointRange<int8_t, Mips_Point<int8_t>>(qFile);
        timeNeighbors<Mips_Point<int8_t>, PointRange<int8_t, Mips_Point<int8_t>>, uint>(Graph, Query_Points, k, BP,
          oFile, GT, rFile, graph_built, Points);
      }
    }
  }else if(graph_type == "aspen"){
    bool graph_built = (gFile != NULL);
    Aspen_Graph<uint> Graph;
    if(gFile == NULL) Graph = Aspen_Graph<uint>(maxDeg, num_points);
    else Graph = Aspen_Graph<uint>(gFile);

    if(tp == "float"){
      if(df == "Euclidian"){
        PointRange<float, Euclidian_Point<float>> Points = PointRange<float, Euclidian_Point<float>>(iFile);
        PointRange<float, Euclidian_Point<float>> Query_Points = PointRange<float, Euclidian_Point<float>>(qFile);
        timeNeighbors<Euclidian_Point<float>, PointRange<float, Euclidian_Point<float>>, uint>(Graph, Query_Points, k, BP, 
          oFile, GT, rFile, graph_built, Points);
      } else if(df == "mips"){
        PointRange<float, Mips_Point<float>> Points = PointRange<float, Mips_Point<float>>(iFile);
        PointRange<float, Mips_Point<float>> Query_Points = PointRange<float, Mips_Point<float>>(qFile);
        timeNeighbors<Mips_Point<float>, PointRange<float, Mips_Point<float>>, uint>(Graph, Query_Points, k, BP, 
          oFile, GT, rFile, graph_built, Points);
      }
    } else if(tp == "uint8"){
      if(df == "Euclidian"){
        PointRange<uint8_t, Euclidian_Point<uint8_t>> Points = PointRange<uint8_t, Euclidian_Point<uint8_t>>(iFile);
        PointRange<uint8_t, Euclidian_Point<uint8_t>> Query_Points = PointRange<uint8_t, Euclidian_Point<uint8_t>>(qFile);
        timeNeighbors<Euclidian_Point<uint8_t>, PointRange<uint8_t, Euclidian_Point<uint8_t>>, uint>(Graph, Query_Points, k, BP, 
          oFile, GT, rFile, graph_built, Points);
      } else if(df == "mips"){
        PointRange<uint8_t, Mips_Point<uint8_t>> Points = PointRange<uint8_t, Mips_Point<uint8_t>>(iFile);
        PointRange<uint8_t, Mips_Point<uint8_t>> Query_Points = PointRange<uint8_t, Mips_Point<uint8_t>>(qFile);
        timeNeighbors<Mips_Point<uint8_t>, PointRange<uint8_t, Mips_Point<uint8_t>>, uint>(Graph, Query_Points, k, BP, 
          oFile, GT, rFile, graph_built, Points);
      }
    } else if(tp == "int8"){
      if(df == "Euclidian"){
        PointRange<int8_t, Euclidian_Point<int8_t>> Points = PointRange<int8_t, Euclidian_Point<int8_t>>(iFile);
        PointRange<int8_t, Euclidian_Point<int8_t>> Query_Points = PointRange<int8_t, Euclidian_Point<int8_t>>(qFile);
        timeNeighbors<Euclidian_Point<int8_t>, PointRange<int8_t, Euclidian_Point<int8_t>>, uint>(Graph, Query_Points, k, BP,
          oFile, GT, rFile, graph_built, Points);
      } else if(df == "mips"){
        PointRange<int8_t, Mips_Point<int8_t>> Points = PointRange<int8_t, Mips_Point<int8_t>>(iFile);
        PointRange<int8_t, Mips_Point<int8_t>> Query_Points = PointRange<int8_t, Mips_Point<int8_t>>(qFile);
        timeNeighbors<Mips_Point<int8_t>, PointRange<int8_t, Mips_Point<int8_t>>, uint>(Graph, Query_Points, k, BP,
          oFile, GT, rFile, graph_built, Points);
      }
    }
  }
  
  
}
