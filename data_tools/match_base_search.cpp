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
void match_base_search(RangeGroundTruth<unsigned int>& GT){
    size_t gtSize = GT.size();
    std::set<coord> result;
    for(int i=0; i<gtSize; i++){
        for(int j=0; j<GT[i].size(); j++){
        result.insert(GT[i][j]);
        }
        // if (i% printPeriod ==0){
        //     std::cout << "current union size:" << result.size() << std::endl;
        // }
    }
    
} 


int main(int argc, char* argv[]) {
    commandLine P(argc,argv,
    "[[-gt_path <g>]]");
    char* gFile = P.getOptionValue("-gt_path");

    RangeGroundTruth<unsigned int> GT = RangeGroundTruth<unsigned int>(gFile);


    unionData(GT);
}