#include <iostream>
#include <algorithm>
#include <random>

#include "../algorithms/bench/parse_command_line.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/euclidian_point.h"
#include "utils/beamSearch.h"

// ./vbase_graph -base_path ../data/sift/sift_learn.fbin -graph_path ../data/sift/sift_learn_32_64 -data_type float -dist_func Euclidian

void abort_with_message(std::string message) {
  std::cout << message << std::endl;
  std::abort();
}

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
		"[-base_path <b>] [-graph_path <gF>] [-data_type <tp>] [-dist_func <df>] [-normalize] [-seed]");
  char* iFile = P.getOptionValue("-base_path");
  char* gFile = P.getOptionValue("-graph_path");
  std::string vectype = P.getOptionValue("-data_type");
  std::string dist_func = P.getOptionValue("-dist_func");
  bool normalize = P.getOption("-normalize");
  int quantize = P.getOptionIntValue("-quantize_bits", 0);
  int seed = P.getOptionIntValue("-seed", -1);

  std::random_device rd;
  std::mt19937 gen(rd()); // seed
  std::mt19937 gen_seed(seed);
  
  if (vectype == "float") {
    if (dist_func == "Euclidian"){
      parlayANN::PointRange<parlayANN::Euclidian_Point<float>> Points(iFile);
      if (normalize) abort_with_message("will impl normalization later");

      parlayANN::Graph<unsigned int> G(gFile);

      if (quantize == 8 || quantize == 16) abort_with_message( "Will impl support for quantize later");
      else {
	using Point = parlayANN::Euclidian_Point<float>;
	using PR = parlayANN::PointRange<Point>;
      // 	std::uniform_int_distribution<> distr(0, 100);
      // 	long random_index;

      // 	if (seed == -1) random_index = distr(gen);
      // 	else random_index = distr(gen_seed);

	const Point random_query_point = Points[10];
	parlay::sequence<unsigned int> starting_points = {0};
	parlayANN::QueryParams QP;
	QP.limit = (long) G.size();
	QP.rerank_factor = 100;
	QP.degree_limit = (long) G.max_degree();
	QP.k = 10;
	QP.cut = 1.35;
	QP.beamSize = 10;
	auto [frontier_and_visited,_] =  parlayANN::beam_search(random_query_point, G, Points, starting_points,  QP);
	
	auto [frontier, visited] = frontier_and_visited;
	std::cout << visited.size() << std::endl;

      // auto [frontier_and_visisted, _] = parlayANN::beam_search<Point, PR, unsigned int>(random_query_point, G, Points, starting_points, QP);
      }

    } else abort_with_message("Other distance functions are not supported at this moment");
  } else abort_with_message("Other vector types are not supported at this moment");

}
