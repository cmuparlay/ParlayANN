#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>

#include "../algorithms/bench/parse_command_line.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/euclidian_point.h"
#include "utils/beamSearch.h"


// ./vbase_graph -base_path ../data/sift/sift_learn.fbin -graph_path ../data/sift/sift_learn_32_64 -data_type float -dist_func Euclidian

std::vector<int> numHopsFromOrigin(parlayANN::Graph<unsigned int> &G) {
  std::set<unsigned int> visited;
  std::vector<int> distances;
  for (int i = 0; i < G.size(); i++) distances.push_back(0);
  std::queue<unsigned int> q;


  q.push(0);
  while (!q.empty()) {
    unsigned int currentNode = q.front();
    q.pop();
    visited.insert(currentNode);
    parlayANN::edgeRange<unsigned int> neighbors = G[currentNode];
    for (size_t i = 0; i < neighbors.size(); i++) {
      unsigned int neighbor = neighbors[i];
      if (visited.find(neighbor) == visited.end()) {
	q.push(neighbor);
	distances[neighbor] = distances[currentNode] + 1;      
      }
    }
  }

  return distances;

}




void abort_with_message(std::string message) {
  std::cout << message << std::endl;
  std::abort();
}

// write output of vertex index and distance to query to file in current directory
void write_output_to_file(int argc, char* argv[], long random_query_index, parlay::sequence<size_t> distance_visted_rank, std::string output_filename = "data.txt") {
  // std::string command;
  // for (int i = 0; i < argc; i++) {
  //   command += std::string(argv[i]);
  //   command += " ";
  // }
  // command += "\n";

  std::ofstream output_file;
  output_file.open(output_filename, std::ios_base::app);
  // output_file << command;
  // output_file << random_query_index << "\n";
  
  for (int i = 0; i < distance_visted_rank.size(); i++) {
    output_file << i << "," << distance_visted_rank[i] << "\n";
  }
  output_file.close();
}


// saves the index and distance of a visited node on the path to a random query
int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
		"[-base_path <b>] [-graph_path <gF>] [-data_type <tp>] [-dist_func <df>] [-normalize] [-seed] [-k] [-beam_size] [-cut] [-limit] [-degree_limit] [-rerank_factor] [-cut]");
  char* iFile = P.getOptionValue("-base_path"); // path to points
  char* gFile = P.getOptionValue("-graph_path"); // path to already generated graph
  std::string vectype = P.getOptionValue("-data_type");
  std::string dist_func = P.getOptionValue("-dist_func");
  bool normalize = P.getOption("-normalize");
  int quantize = P.getOptionIntValue("-quantize_bits", 0);
  int seed = P.getOptionIntValue("-seed", -1);

  long limit = P.getOptionLongValue("-limit", -1);
  int rerank_factor = P.getOptionIntValue("-rerank_factor", 100);
  long degree_limit = P.getOptionLongValue("-dergee_limit", -1);
  int k = P.getOptionIntValue("-k", 10);
  int beam_size = P.getOptionIntValue("-beam_size", 10);
  double cut = P.getOptionDoubleValue("-cut", 1.35);
  
  std::random_device rd;
  std::mt19937 gen(rd()); // seed
  std::mt19937 gen_seed(seed);
  
  if (vectype == "float") {
    if (dist_func == "Euclidian"){
      parlayANN::PointRange<parlayANN::Euclidian_Point<float>> Points(iFile);
      if (normalize) abort_with_message("will impl normalization later");

      parlayANN::Graph<unsigned int> G(gFile);

      if (quantize == 8 || quantize == 16) abort_with_message( "Will impl support for quantize later");

      using Point = parlayANN::Euclidian_Point<float>;
      using PR = parlayANN::PointRange<Point>;
      
      std::uniform_int_distribution<> distr(0, Points.size() - 1);

      for (int oo = 0; oo < 100; oo++) {
      
	long random_index;

	if (seed == -1) random_index = distr(gen);
	else random_index = distr(gen_seed);

	const Point random_query_point = Points[random_index];


	unsigned int start_index = 0;
	parlay::sequence<unsigned int> starting_points = {start_index};
	parlayANN::QueryParams QP;

	QP.limit = limit != -1 ? limit : (long) G.size();
	QP.rerank_factor = rerank_factor;
	QP.degree_limit = degree_limit != -1 ? degree_limit : (long) G.max_degree();
	QP.k = k;
	QP.cut = cut ;
	QP.beamSize = beam_size;

	parlay::sequence<float> distances_from_query_to_all = parlay::tabulate(Points.size(), [&](long i) {
	  return Points[random_index].distance(Points[i]);
	});

	parlay::sequence<size_t> distances_from_query_to_all_rank = parlay::rank(distances_from_query_to_all);
	
	auto [frontier_and_visited,_] =  parlayANN::beam_search(random_query_point, G, Points, starting_points,  QP);
	
	auto visited = frontier_and_visited.second;

	parlay::sequence<size_t> distance_visted_rank = parlay::tabulate(visited.size(), [&] (long i) {return distances_from_query_to_all_rank[visited[i].first];});

	write_output_to_file(argc, argv, random_index, distance_visted_rank);

      }

      
    } else abort_with_message("Other distance functions are not supported at this moment");
  } else abort_with_message("Other vector types are not supported at this moment");

}
