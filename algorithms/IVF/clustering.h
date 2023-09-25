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

template<class Point, class PointRange>
struct HCNNGClusterer {
    size_t cluster_size = 1000;
    Graph<size_t> G; // not actually used but needed as a function argument for clusterEdge

    HCNNGClusterer() {}

    HCNNGClusterer(size_t cluster_size) : cluster_size(cluster_size) {}


    parlay::sequence<parlay::sequence<size_t>> cluster(PointRange points) {
        size_t num_points = points.size();
        // we make a very sparse sequence of sequences to store the clusters
        parlay::sequence<parlay::sequence<size_t>> clusters(num_points);
        auto assign = [&] (Graph<size_t> G, PointRange points, parlay::sequence<size_t> active_indices, long MSTDeg) {
            size_t cluster_index = active_indices[0];
            clusters[cluster_index] = active_indices;
        };
        
        // should populate the clusters sequence
        cluster_struct<Point, PointRange, size_t>().random_clustering_wrapper(G, points, cluster_size, assign, 0);

        // remove empty clusters
        clusters = parlay::filter(clusters, [&] (parlay::sequence<size_t> cluster) {return cluster.size() > 0;});

        return clusters;
    }
};

#endif // CLUSTERING_H