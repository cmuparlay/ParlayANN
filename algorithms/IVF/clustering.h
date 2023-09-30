/* structs which should take a PointRange argument and return a sequence of sequences representing clusters */
#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"

#include "../utils/point_range.h"
#include "../HCNNG/clusterEdge.h"
#include "../utils/graph.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <utility>  

template<typename Point, typename PointRange, typename indexType>
using cluster_struct = cluster<Point, PointRange, indexType>;

// using index_type = int32_t;

template<class Point, class PointRange, typename index_type>
struct HCNNGClusterer {
    size_t cluster_size = 1000;
    Graph<index_type> G; // not actually used but needed as a function argument for clusterEdge

    HCNNGClusterer() {}

    HCNNGClusterer(size_t cluster_size) : cluster_size(cluster_size) {}


    parlay::sequence<parlay::sequence<index_type>> cluster(PointRange points) {
        size_t num_points = points.size();
        // we make a very sparse sequence of sequences to store the clusters
        parlay::sequence<parlay::sequence<index_type>> clusters(num_points);
        auto assign = [&] (Graph<index_type> G, PointRange points, parlay::sequence<index_type> active_indices, long MSTDeg) {
            index_type cluster_index = active_indices[0];
            clusters[cluster_index] = parlay::tabulate(cluster_size, [&] (size_t i) {return static_cast<index_type>(active_indices[i]);});
        };
        
        // should populate the clusters sequence
        cluster_struct<Point, PointRange, index_type>().random_clustering_wrapper(G, points, cluster_size, assign, 0);

        // remove empty clusters
        clusters = parlay::filter(clusters, [&] (parlay::sequence<index_type> cluster) {return cluster.size() > 0;});

        return clusters;
    }
};

#endif // CLUSTERING_H