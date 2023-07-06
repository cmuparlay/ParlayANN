#include "ClusterIndex.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/random.h"
#include "utils/indexTools.h"

#include "../HCNNG/hcnng_index.h"


/* 
    A trivial IVF clustering algorithm which does no clustering (constructs a single shard) and then does HCNNG on that shard.

    Should in theory be equivalent to just using HCNNG.
*/

Routing trivial_route = [](parlay::sequence<shard>* shards, 
                           parlay::sequence<size_t>* to_search, 
                           Tvec_point<T>* query, 
                           size_t k) {
    to_search->push_back(0);
};

Clustering trivial_cluster = [](parlay::sequence<Tvec_point<T>>* data, 
                                parlay::sequence<parlay::sequence<size_t>>* shards, 
                                parlay::sequence<size_t>* centroid_indices,
                                size_t k) {
    shards->push_back(parlay::sequence<size_t>(data->size()));
    for (size_t i = 0; i < data->size(); i++) {
        (*shards)[0][i] = i;
    }
    centroid_indices->push_back(0);
};


