#include <iostream>
#include <algorithm>
#include <set>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include "utils/graph.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"
#include "utils/types.h"


using coord = unsigned int;

// Get slice for each of the point in query point Q, and then take union in the range
// Take the size for each of the 
void unionData(RangeGroundTruth<unsigned int>& GT){
    size_t gtSize = GT.size();
    //answers = compute_groundtruth<PointRange<float, Euclidian_Point<float>>>(B, Q, k);
    int printPeriod = (int)gtSize/100;
    std::set<coord> result;
    size_t cnt=0;
    for(int i=0; i<gtSize; i++){
        if(GT[i].size()>0){
            cnt += 1;
        }
        for(int j=0; j<GT[i].size(); j++){
        result.insert(GT[i][j]);

        }
        if (i% printPeriod ==0){
            std::cout << "current union size:" << result.size() << std::endl;
        }
    }    
    std::cout << "Union matches: " << cnt << std::endl;
} 


int main(int argc, char* argv[]) {
    commandLine P(argc,argv,
    "[[-gt_path <g>]]");
    char* gFile = P.getOptionValue("-gt_path");
    //char* bFile = P.getOptionValue("-base_path");
    //char* qFile = P.getOptionValue("-query_path");
    //int k = P.getOptionValue("-k",100);
    //char* vectype = P.getOptionValue("-dist_func");

    //std::string tp = std::string(vectype);
    //std::string base = std::string(bFile);
    //std::string query = std::string(qFile);

    //PointRange<uint8_t, Euclidian_Point<uint8_t>> B = PointRange<uint8_t, Euclidian_Point<uint8_t>>(bFile);
    //PointRange<uint8_t, Euclidian_Point<uint8_t>> Q = PointRange<uint8_t, Euclidian_Point<uint8_t>>(qFile);

    RangeGroundTruth<unsigned int> GT = RangeGroundTruth<unsigned int>(gFile);


    unionData(GT);
}