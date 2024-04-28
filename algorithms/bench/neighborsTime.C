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
#include "../utils/aspen_graph.h"
#include "../utils/aspen_flat_graph.h"



#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>




// *************************************************************
//  TIMING
// *************************************************************

using uint = unsigned int;

//file order: {base, query, graph, groundtruth, graph_outfile, res_file}
template<typename PointRange, typename GraphType>
void timeNeighbors(parlay::sequence<char*> files, long k, BuildParams &BP, size_t n){
  using indexType = typename GraphType::iT;
  using Point = typename PointRange::pT;

  //load GT
  groundTruth<indexType> GT = groundTruth<indexType>(files[3]);

  //load graph
  char* gFile = files[2];
  long maxDeg = BP.max_degree();
  bool graph_built = (gFile != NULL);
  GraphType Graph;
  if(gFile == NULL) Graph = GraphType(maxDeg, n);
  else Graph = GraphType(gFile);

  //load point ranges
  PointRange Points = PointRange(files[0]);
  PointRange Query_Points = PointRange(files[1]);

  time_loop(1, 0,
    [&] () {},
    [&] () {
      ANN<Point, PointRange, indexType, GraphType>(Graph, k, BP, Query_Points, GT, files[4], graph_built, Points);
    },
    [&] () {});

  if(files[5] != NULL) {
    Graph.save(files[5]);
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
  if(graph_type != "flat" && graph_type != "aspen" && graph_type != "aspen_flat"){
    std::cout << "Error: specify graph type flat or aspen or aspen_flat" << std::endl;
    abort();
  }


  //read the number of points in order to prepare graph 
  std::ifstream reader(iFile);
  assert(reader.is_open());
  unsigned int num_points;
  reader.read((char*)(&num_points), sizeof(unsigned int));
  reader.close();

  parlay::sequence<char*> files = {iFile, qFile, gFile, cFile, oFile, rFile};

  if(graph_type == "flat"){
    if(tp == "float"){
      if(df == "Euclidian"){
        timeNeighbors<PointRange<float, Euclidian_Point<float>>, Flat_Graph<uint>>(files, k, BP, num_points);
      } else if(df == "mips"){
        timeNeighbors<PointRange<float, Mips_Point<float>>, Flat_Graph<uint>>(files, k, BP, num_points);
      }
    } else if(tp == "uint8"){
      if(df == "Euclidian"){
        timeNeighbors<PointRange<uint8_t, Euclidian_Point<uint8_t>>, Flat_Graph<uint>>(files, k, BP, num_points);
      } else if(df == "mips"){
        timeNeighbors<PointRange<uint8_t, Mips_Point<uint8_t>>, Flat_Graph<uint>>(files, k, BP, num_points);
      }
    } else if(tp == "int8"){
      if(df == "Euclidian"){
        timeNeighbors<PointRange<int8_t, Euclidian_Point<int8_t>>, Flat_Graph<uint>>(files, k, BP, num_points);
      } else if(df == "mips"){
        timeNeighbors<PointRange<int8_t, Mips_Point<int8_t>>, Flat_Graph<uint>>(files, k, BP, num_points);
      }
    }
  } else if(graph_type == "aspen"){
    if(tp == "float"){
      if(df == "Euclidian"){
        timeNeighbors<PointRange<float, Euclidian_Point<float>>, Aspen_Graph<uint>>(files, k, BP, num_points);
      } else if(df == "mips"){
        timeNeighbors<PointRange<float, Mips_Point<float>>, Aspen_Graph<uint>>(files, k, BP, num_points);
      }
    } else if(tp == "uint8"){
      if(df == "Euclidian"){
        timeNeighbors<PointRange<uint8_t, Euclidian_Point<uint8_t>>, Aspen_Graph<uint>>(files, k, BP, num_points);
      } else if(df == "mips"){
        timeNeighbors<PointRange<uint8_t, Mips_Point<uint8_t>>, Aspen_Graph<uint>>(files, k, BP, num_points);
      }
    } else if(tp == "int8"){
      if(df == "Euclidian"){
        timeNeighbors<PointRange<int8_t, Euclidian_Point<int8_t>>, Aspen_Graph<uint>>(files, k, BP, num_points);
      } else if(df == "mips"){
        timeNeighbors<PointRange<int8_t, Mips_Point<int8_t>>, Aspen_Graph<uint>>(files, k, BP, num_points);
      }
    }
  } else if(graph_type == "aspen_flat"){
    if(tp == "float"){
      if(df == "Euclidian"){
        timeNeighbors<PointRange<float, Euclidian_Point<float>>, Aspen_Flat_Graph<uint>>(files, k, BP, num_points);
      } else if(df == "mips"){
        timeNeighbors<PointRange<float, Mips_Point<float>>, Aspen_Flat_Graph<uint>>(files, k, BP, num_points);
      }
    } else if(tp == "uint8"){
      if(df == "Euclidian"){
        timeNeighbors<PointRange<uint8_t, Euclidian_Point<uint8_t>>, Aspen_Flat_Graph<uint>>(files, k, BP, num_points);
      } else if(df == "mips"){
        timeNeighbors<PointRange<uint8_t, Mips_Point<uint8_t>>, Aspen_Flat_Graph<uint>>(files, k, BP, num_points);
      }
    } else if(tp == "int8"){
      if(df == "Euclidian"){
        timeNeighbors<PointRange<int8_t, Euclidian_Point<int8_t>>, Aspen_Flat_Graph<uint>>(files, k, BP, num_points);
      } else if(df == "mips"){
        timeNeighbors<PointRange<int8_t, Mips_Point<int8_t>>, Aspen_Flat_Graph<uint>>(files, k, BP, num_points);
      }
    }
  }
  
  
}
