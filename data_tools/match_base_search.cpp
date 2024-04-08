#include <iostream>
#include <algorithm>
#include <set>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include "parlay/sequence.h"
#include "utils/graph.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"
#include "utils/types.h"
#include "utils/check_nn_recall.h"
#include "utils/parse_results.h"
#include "utils/stats.h"
#include "utils/index.h"





using coord = long int;

template <typename Graph, typename Point, typename PointRange, typename indexType>
void match_base_search(groundTruth<unsigned int> &GT, PointRange &Query_Points, PointRange &Points, 
Graph& G,BuildParams &BP, char* res_file, long k){
  size_t gtSize = GT.size();
  std::set<coord> result;
  for(int i=0; i<gtSize; i++){
        // for(int j=0; j<GT[i].size(); j++){
        // result.insert(GT[i][j]);
        // }
      if(GT[i].size()>0){
          result.insert(i);
      }
  }

  

  //convert result into vector

  parlay::sequence<long int> res_vec(result.begin(),result.end());
  
  using findex = knn_index<Point, PointRange, indexType>;
  findex I(BP);
  double idx_time;
  stats<unsigned int> BuildStats(G.size());
  I.build_index(G, Points, BuildStats);
  std::string name = "Vamana";
  std::string params =
      "R = " + std::to_string(BP.R) + ", L = " + std::to_string(BP.L);
  auto [avg_deg, max_deg] = graph_stats_(G);
  auto vv = BuildStats.visited_stats();
  std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1]
            << std::endl;
  Graph_ G_(name, params, G.size(), avg_deg, max_deg, idx_time);
  G_.print();

  if(Query_Points.size() != 0){
    search_and_parse_multi<Point, PointRange, indexType>(G_, G, Points, Query_Points, GT, res_file, k, false, res_vec);
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

  bool graph_built = (gFile != NULL);

  groundTruth<unsigned int> GT = groundTruth<unsigned int>(cFile);
  Graph<unsigned int> G = Graph<unsigned int>(gFile);
  
  if(tp == "float"){
    if(df == "Euclidian"){
      PointRange<float, Euclidian_Point<float>> Points = PointRange<float, Euclidian_Point<float>>(iFile);
      PointRange<float, Euclidian_Point<float>> Query_Points = PointRange<float, Euclidian_Point<float>>(qFile);
      match_base_search<Graph<unsigned int>, Euclidian_Point<float>, PointRange<float, Euclidian_Point<float>>, uint>(GT, Query_Points, Points, G, BP,rFile,k);
      // Graph<unsigned int> G; 
      // if(gFile == NULL) G = Graph<unsigned int>(maxDeg, Points.size());
      // else G = Graph<unsigned int>(gFile);
      // timeNeighbors<Euclidian_Point<float>, PointRange<float, Euclidian_Point<float>>, uint>(G, Query_Points, k, BP, 
      //   oFile, GT, rFile, graph_built, Points);
    } else if(df == "mips"){
      PointRange<float, Mips_Point<float>> Points = PointRange<float, Mips_Point<float>>(iFile);
      PointRange<float, Mips_Point<float>> Query_Points = PointRange<float, Mips_Point<float>>(qFile);
      match_base_search<Graph<unsigned int>, Mips_Point<float>, PointRange<float, Mips_Point<float>>, long int>(GT, Query_Points, Points, G, BP,rFile,k);
      // Graph<unsigned int> G; 
      // if(gFile == NULL) G = Graph<unsigned int>(maxDeg, Points.size());
      // else G = Graph<unsigned int>(gFile);
      // timeNeighbors<Mips_Point<float>, PointRange<float, Mips_Point<float>>, uint>(G, Query_Points, k, BP, 
      //   oFile, GT, rFile, graph_built, Points);
    }
    
  } else if(tp == "uint8"){
    if(df == "Euclidian"){
      PointRange<uint8_t, Euclidian_Point<uint8_t>> Points = PointRange<uint8_t, Euclidian_Point<uint8_t>>(iFile);
      PointRange<uint8_t, Euclidian_Point<uint8_t>> Query_Points = PointRange<uint8_t, Euclidian_Point<uint8_t>>(qFile);
      match_base_search<Graph<unsigned int>, Euclidian_Point<uint8_t>, PointRange<uint8_t, Euclidian_Point<uint8_t>>, long int>(GT, Query_Points, Points, G, BP, rFile,k);
      // if(gFile == NULL) G = Graph<unsigned int>(maxDeg, Points.size());
      // else G = Graph<unsigned int>(gFile);
      // timeNeighbors<Euclidian_Point<uint8_t>, PointRange<uint8_t, Euclidian_Point<uint8_t>>, uint>(G, Query_Points, k, BP, 
      //   oFile, GT, rFile, graph_built, Points);
    } else if(df == "mips"){
      PointRange<uint8_t, Mips_Point<uint8_t>> Points = PointRange<uint8_t, Mips_Point<uint8_t>>(iFile);
      PointRange<uint8_t, Mips_Point<uint8_t>> Query_Points = PointRange<uint8_t, Mips_Point<uint8_t>>(qFile);
      match_base_search<Graph<unsigned int>, Mips_Point<uint8_t>, PointRange<uint8_t, Mips_Point<uint8_t>>, long int>(GT, Query_Points, Points, G, BP, rFile,k);
      // Graph<unsigned int> G; 
      // if(gFile == NULL) G = Graph<unsigned int>(maxDeg, Points.size());
      // else G = Graph<unsigned int>(gFile);
      // timeNeighbors<Mips_Point<uint8_t>, PointRange<uint8_t, Mips_Point<uint8_t>>, uint>(G, Query_Points, k, BP, 
      //   oFile, GT, rFile, graph_built, Points);
    }
  } else if(tp == "int8"){
    if(df == "Euclidian"){
      PointRange<int8_t, Euclidian_Point<int8_t>> Points = PointRange<int8_t, Euclidian_Point<int8_t>>(iFile);
      PointRange<int8_t, Euclidian_Point<int8_t>> Query_Points = PointRange<int8_t, Euclidian_Point<int8_t>>(qFile);
      match_base_search<Graph<unsigned int>, Euclidian_Point<int8_t>, PointRange<int8_t, Euclidian_Point<int8_t>>, long int>(GT, Query_Points, Points, G, BP,rFile, k);
      // Graph<unsigned int> G; 
      // if(gFile == NULL) G = Graph<unsigned int>(maxDeg, Points.size());
      // else G = Graph<unsigned int>(gFile);
      // timeNeighbors<Euclidian_Point<int8_t>, PointRange<int8_t, Euclidian_Point<int8_t>>, uint>(G, Query_Points, k, BP,
      //   oFile, GT, rFile, graph_built, Points);
    } else if(df == "mips"){
      PointRange<int8_t, Mips_Point<int8_t>> Points = PointRange<int8_t, Mips_Point<int8_t>>(iFile);
      PointRange<int8_t, Mips_Point<int8_t>> Query_Points = PointRange<int8_t, Mips_Point<int8_t>>(qFile);
      match_base_search<Graph<unsigned int>, Mips_Point<int8_t>, PointRange<int8_t, Mips_Point<int8_t>>, long int>(GT, Query_Points, Points, G, BP,rFile,k);
      // Graph<unsigned int> G; 
      // if(gFile == NULL) G = Graph<unsigned int>(maxDeg, Points.size());
      // else G = Graph<unsigned int>(gFile);
      // timeNeighbors<Mips_Point<int8_t>, PointRange<int8_t, Mips_Point<int8_t>>, uint>(G, Query_Points, k, BP,
      //   oFile, GT, rFile, graph_built, Points);
    }
  }
  

}