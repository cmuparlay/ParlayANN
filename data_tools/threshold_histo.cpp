#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include "utils/graph.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"

#define RATIO 1



template<typename Graph, typename PointRange>
void ball_histogram(Graph &G, PointRange &B){   
    //parlay::sequence<int> degrees = parlay::tabulate(G.size(), [&] (size_t i){return static_cast<int>(G[i].size());});
    int maxDeg = G.max_degree();

    std::cout<< "started ball histogram"<< std::endl;

    parlay::sequence<parlay::sequence<unsigned int>> index = 
        parlay::tabulate(G.size(), [&](size_t i){
            // parlay::sequence<unsigned int> intG = parlay::map(G[i],[&](auto j){return static_cast<unsigned int>(j);});
            // return intG;
            return parlay::tabulate(G[i].size(), [&] (size_t j) {return G[i][j];});
        });

    std::cout<< index[0].size() << std::endl;

    parlay::sequence<parlay::sequence<float>> distance = parlay::tabulate(index.size(),
     [&](size_t i){return parlay::tabulate(index[i].size(),[&](size_t j){return B[i].distance(B[index[i][j]]);});}); 
     
    parlay::sequence<float> distanceForReduction = parlay::flatten(distance);

    parlay::sort_inplace(distanceForReduction);

    size_t totalSize = distanceForReduction.size();
    int zeropointonePct = (int)(totalSize * 0.001);
    int smallest = (int)(totalSize * 0.01);
    int fstPct = (int)(totalSize * 0.25);
    int sndPct = (int)(totalSize * 0.5);
    int thdPct = (int)(totalSize * 0.75);
    int furthest = (int)(totalSize * 0.99);

    std::cout << "zeropointonePct" << distanceForReduction[zeropointonePct] << std::endl;
    std::cout << "smallest:" << distanceForReduction[smallest] << std::endl;
    std::cout << "fstPct" << distanceForReduction[fstPct] << std::endl;
    std::cout << "sndPct" << distanceForReduction[sndPct] << std::endl;
    std::cout << "thdPct" << distanceForReduction[thdPct] << std::endl;
    std::cout << "furthest" << distanceForReduction[furthest] << std::endl;

    //float minDistance = parlay::reduce(distanceForReduction,parlay::minimum<float>());
    //printf("minDistance: %f\n",minDistance);
    std::cout<< "dist0:" << distance[0].size() << std::endl;
    std::cout<< "dist00:" << distance[0][0] << std::endl;

    float sumDistance = parlay::reduce(distanceForReduction,parlay::plus<float>());
    float avgDistance = sumDistance / distanceForReduction.size();
    //printf("distSize: %f\n",distanceForReduction.size());
    std::cout << "distsize:" << distanceForReduction.size() << std::endl;
    std::cout << "avgDistance:" << avgDistance << std::endl;
    //printf("avgDistance: %f\n",avgDistance);

    //decide radius somehow-> I would start by taking the minimum of overall, maybe multiply by some constant and then
    //see how many edges are left inside each point
    float radius = 96327*2;

    parlay::sequence<parlay::sequence<float>> filteredDistance = parlay::tabulate(distance.size(), [&](size_t i)
    {
        parlay::sequence<float> result = parlay::filter(distance[i], [&](float j){return (j <= radius);});
        return result;
    });
    // parlay::parallel_for(0, distance.size(),[&](long i){
    //     parlay::sequence<float> result = parlay::filter(distance[i], [&](float j){return j <= radius;});
    //     append(filteredDistance, result);
    // });

    parlay::sequence<int> shortDegrees = parlay::tabulate(filteredDistance.size(), [&](size_t i){
        return static_cast<int>(filteredDistance[i].size());
    });

    auto histogram = parlay::histogram_by_index(shortDegrees, maxDeg);
    int majority=0;
    for(int i=maxDeg/2;i<maxDeg;i++){
        majority += (histogram[i]);
    }
    std::cout << parlay::to_chars(histogram) << std::endl;
    std::cout << "majority:" << majority << std::endl;
}

//Based on bucket by difference between minimum and maximum, and then divide it by power of two. 
template<typename Graph,typename PointRange>
void minmax_histogram(Graph &G,PointRange &B){
    int maxDeg = G.max_degree();
    parlay::sequence<parlay::sequence<unsigned int>> index = 
        parlay::tabulate(G.size(), [&](size_t i){
            parlay::sequence<unsigned int> intG = parlay::map(G[i],[&](size_t j){return static_cast<unsigned int>(G[i][j]);});
            return intG;});
    parlay::sequence<parlay::sequence<float>> distance = parlay::tabulate(index.size(),
     [&](size_t i){return parlay::tabulate(index[i].size(),[&](size_t j){return B[i].distance(B[index[i][j]]);});}); 
    parlay::sequence<float> distanceForReduction = parlay::flatten(distance);
    float minDistance = parlay::reduce(distanceForReduction,parlay::minimum<float>());
    float maxDistance = parlay::reduce(distanceForReduction,parlay::maximum<float>());
    //get the minimum distance, and maximum distance inside the graph edge
    //Map all the value into distance for each point, then do reduction by minimum/maximum for each point, and then
    //reduce once again among all vertex

    float diffDistance = maxDistance - minDistance;


    //Then, calculate the difference between minimum and maximum, and then bucket this by 10, 

    //auto histogram = parlay::histogram_by_index(degrees, maxDeg);
    //std::cout << parlay::to_chars(histogram) << std::endl;
}

template <typename PointRange>
void distanceMap(PointRange &B){
    size_t totalSize = B.size();
    size_t blockSize = (int)(totalSize/100);
    auto permutation = parlay::random_permutation(totalSize);
    auto block = parlay::tabulate(blockSize, [&](size_t i){return permutation[i];});
    auto distanceMap = parlay::tabulate(block.size(),[&](size_t i){
        return parlay::tabulate(totalSize,[&](size_t j){return B[i].distance(B[j]);});
    });
    auto distanceMapFlatten = parlay::flatten(distanceMap);
    size_t totalSize2 = distanceMapFlatten.size();
    parlay::sort_inplace(distanceMapFlatten);
    auto nonZero = parlay::filter(distanceMapFlatten,[&](size_t i){return i>0;});
    if(distanceMapFlatten[0]==0){
      std::cout << "Minimum: " << nonZero[0] << std::endl;
    }else{
      std::cout << "Minimum: " << distanceMapFlatten[0] << std::endl;
    }
    
    
    int64_t zeropointonePct = (int64_t)(totalSize2 * 0.001);
    int64_t onePct = (int64_t)(totalSize2 * 0.01);
    int64_t fstQrt = (int64_t)(totalSize2 * 0.25);
    int64_t sndQrt = (int64_t)(totalSize2 * 0.5);
    int64_t thdQrt = (int64_t)(totalSize2 * 0.75);
    int64_t furthest = (int64_t)(totalSize2 * 0.99);

    std::cout << "zeropointonePct: " << distanceMapFlatten[zeropointonePct] << std::endl;
    std::cout << "onePct: " << distanceMapFlatten[onePct] << std::endl;
    std::cout << "fstQrt: " << distanceMapFlatten[fstQrt] << std::endl;
    std::cout << "sndQrt: " << distanceMapFlatten[sndQrt] << std::endl;
    std::cout << "thdQrt: " << distanceMapFlatten[thdQrt] << std::endl;
    std::cout << "furthest: " << distanceMapFlatten[furthest] << std::endl;

}




int main(int argc, char* argv[]) {
    commandLine P(argc,argv,
    "[[-base_path <b>],[-graph_path <g>],[-data_type <d>], [-dist_func <d>]]");
    char* gFile = P.getOptionValue("-graph_path");
    char* bFile = P.getOptionValue("-base_path");
    char* vectype = P.getOptionValue("-data_type");
    char* dfc = P.getOptionValue("-dist_func");

    std::string tp = std::string(vectype);
    std::string base = std::string(bFile);
    std::string df = std::string(dfc);


    if(tp == "float"){
    std::cout << "Detected float coordinates" << std::endl;
    if(df == "Euclidian"){
      PointRange<float, Euclidian_Point<float>> B = PointRange<float, Euclidian_Point<float>>(bFile);
      //PointRange<float, Euclidian_Point<float>> Q = PointRange<float, Euclidian_Point<float>>(qFile);
    } else if(df == "mips"){
      PointRange<float, Mips_Point<float>> B = PointRange<float, Mips_Point<float>>(bFile);
      //PointRange<float, Mips_Point<float>> Q = PointRange<float, Mips_Point<float>>(qFile);
    }
  }else if(tp == "uint8"){
    std::cout << "Detected uint8 coordinates" << std::endl;
    if(df == "Euclidian"){
      PointRange<uint8_t, Euclidian_Point<uint8_t>> B = PointRange<uint8_t, Euclidian_Point<uint8_t>>(bFile);
      //PointRange<uint8_t, Euclidian_Point<uint8_t>> Q = PointRange<uint8_t, Euclidian_Point<uint8_t>>(qFile);
    } else if(df == "mips"){
      PointRange<uint8_t, Mips_Point<uint8_t>> B = PointRange<uint8_t, Mips_Point<uint8_t>>(bFile);
      //PointRange<uint8_t, Mips_Point<uint8_t>> Q = PointRange<uint8_t, Mips_Point<uint8_t>>(qFile);
    }
  }else if(tp == "int8"){
    std::cout << "Detected int8 coordinates" << std::endl;
    if(df == "Euclidian"){
      PointRange<int8_t, Euclidian_Point<int8_t>> B = PointRange<int8_t, Euclidian_Point<int8_t>>(bFile);
      //PointRange<int8_t, Euclidian_Point<int8_t>> Q = PointRange<int8_t, Euclidian_Point<int8_t>>(qFile);
    } else if(df == "mips"){
      PointRange<int8_t, Mips_Point<int8_t>> B = PointRange<int8_t, Mips_Point<int8_t>>(bFile);
      //PointRange<int8_t, Mips_Point<int8_t>> Q = PointRange<int8_t, Mips_Point<int8_t>>(qFile);
    }
  }

    //PointRange<uint8_t, Euclidian_Point<uint8_t>> B = PointRange<uint8_t, Euclidian_Point<uint8_t>>(bFile);
    //PointRange<float, Mips_Point<float>> B = PointRange<float, Mips_Point<float>>(bFile);
    //PointRange<uint8_t, Euclidian_Point<uint8_t>> Q = PointRange<uint8_t, Euclidian_Point<uint8_t>>(qFile);

    //Graph<unsigned int> G =  Graph<unsigned int>(gFile);
    

    ball_histogram(G,B);
    std::cout << "----------------------------------" << std::endl;
    //minmax_histogram(G,B);
    distanceMap(B);
}