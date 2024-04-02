#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include "parlay/random.h"
#include "utils/aspen_graph.h"
#include "utils/aspen_flat_graph.h"
#include "utils/graph.h"
#include "rand_r_32.h"

#include <random>

template<typename GraphType>
void time_accesses(GraphType &Graph, int p, double trial_time){
    

    parlay::sequence<size_t> totals(p);
    parlay::internal::timer t("ANN");
    auto G = Graph.Get_Graph_Read_Only();
    size_t n = G.size();
    // run benchmark
    t.start();
    auto start = std::chrono::system_clock::now();
    volatile int test;

    parlay::parallel_for(0, p, [&] (size_t i) {
      int cnt = 0;
      size_t total = 0;
      my_rand::init(i);
      while (true) {
        // every once in a while check if time is over
        if (cnt == 100) {
          cnt = 0;
          auto current = std::chrono::system_clock::now();
          double duration = std::chrono::duration_cast<std::chrono::seconds>(current - start).count();
          if (duration > trial_time) {
            totals[i] = total;
            return;
          }
        }
        int idx = my_rand::get_rand()%n;
        auto nbh = G[idx].neighbors();
        test = nbh[nbh.size()-1];
        cnt++;
        total++;
      }
			       }, 1, true);
    double duration = t.stop();


    size_t num_ops = parlay::reduce(totals);
    std::cout << "throughput (Mop/s): "
            << num_ops / (duration * 1e6) << std::endl << std::endl;

    Graph.Release_Graph(std::move(G));

}


int main(int argc, char* argv[]) {
  commandLine P(argc,argv,
      "[-graph_path <outfile>]");

  char* gFile = P.getOptionValue("-graph_path");

  int p = 144;
  float trial_time = 10.0;
  
  Flat_Graph<unsigned int> flat_graph = Flat_Graph<unsigned int>(gFile);
  Aspen_Graph<unsigned int> aspen_graph = Aspen_Graph<unsigned int>(gFile);
  Aspen_Flat_Graph<unsigned int> aspen_flat_graph = Aspen_Flat_Graph<unsigned int>(gFile);

  std::cout << "Flat Graph" << std::endl;
  time_accesses(flat_graph, p, trial_time);
  std::cout << std::endl;

  std::cout << "Aspen Graph" << std::endl;
  time_accesses(aspen_graph, p, trial_time);
  std::cout << std::endl;

  std::cout << "Aspen Flat Graph" << std::endl;
  time_accesses(aspen_flat_graph, p, trial_time);
  std::cout << std::endl;



  return 0;
}

