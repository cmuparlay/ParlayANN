#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include "utils/graph.h"

template<typename Graph>
void degree_histogram(Graph &G){
    //convert degrees to a sequence of integers
    parlay::sequence<int> degrees = parlay::tabulate(G.size(), [&] (size_t i){return static_cast<int>(G[i].size());});
    int maxDeg = G.max_degree();
    auto histogram = parlay::histogram_by_index(degrees, maxDeg);
    std::cout << parlay::to_chars(histogram) << std::endl;
}

int main(int argc, char* argv[]) {
    commandLine P(argc,argv,
    "[[-graph_path <g>] ]");
    char* gFile = P.getOptionValue("-graph_path");

    Graph<unsigned int> G =  Graph<unsigned int>(gFile);
    degree_histogram(G);
}