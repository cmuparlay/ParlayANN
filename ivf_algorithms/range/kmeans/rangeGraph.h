#include "../../ivf/ivf.h"
#include "../../ivf/clustering.h"
#include "../../ivf/posting_list.h"
#include "../../ivf/check_ivf_range_recall.h"
#include "../../ivf/ivfGraph.h"

#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"
#include "utils/types.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>

template<typename Point, typename PointRange, typename indexType>
void RNG(double rad, BuildParams &BP,
         PointRange &Query_Points,
         RangeGroundTruth<indexType> GT, char *res_file,
         bool graph_built, PointRange &Points, char* index_path, char* index_savepath) {

    using T = typename PointRange::TT;

    std::cout << "Building with " << BP.num_clusters << " clusters" << std::endl;
    parlay::internal::timer t("IVF");
    double idx_time;
    KMeansClusterer<T, Point, indexType> K(BP.num_clusters);
    GraphPostingListIndex<T, Point, indexType> PostingList(Points, K, 0, index_path);
    idx_time = t.next_time();
    std::cout << "Index built in " << idx_time << " s" << std::endl;
    std::string params = "Num_clusters = " + std::to_string(BP.num_clusters);
    IVF_ I("Kmeans", params, Points.size(), idx_time);
    
    search_and_parse_range<Point, PointRange, indexType, GraphPostingListIndex<T, Point, indexType>>(PostingList, Points, Query_Points, GT, res_file, rad, I);

    if(index_savepath != nullptr) {
        std::cout << "Saving index..." << std::endl;
        PostingList.save(index_savepath); 
    }
    
}