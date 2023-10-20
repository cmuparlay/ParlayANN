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


    // parlay::sequence<parlay::sequence<index_type>> cluster(PointRange points) {
    //     size_t num_points = points.size();
    //     // we make a very sparse sequence to store the clusters
    //     parlay::sequence<std::pair<index_type*, index_type>> clusters(num_points);
    //     // lambda to assign a cluster
    //     auto assign = [&] (Graph<index_type> G, PointRange points, parlay::sequence<index_type>& active_indices, long MSTDeg) {
    //         if (active_indices.size() == 0) { // this should never happen
    //             // std::cout << "HCNNGClusterer::cluster: active_indices.size() == 0" << std::endl;
    //             return;
    //         }
    //         if (std::max_element(active_indices.begin(), active_indices.end())[0] > num_points) {
    //             std::cout << "lambda receiving oversized index" << std::endl;
    //             return;
    //         }
    //         index_type cluster_index = active_indices[0];
    //         // clusters[cluster_index] = parlay::tabulate(cluster_size, [&] (size_t i) {return static_cast<index_type>(active_indices[i]);});
    //         index_type* cluster = new index_type[active_indices.size()];

    //         for (size_t i = 0; i < active_indices.size(); i++) {
    //             cluster[i] = active_indices[i];
    //             if (active_indices[i] > num_points) {
    //                 std::cout << "assign: active_indices[i] > num_points" << std::endl;
    //                 break;
    //             }
    //         }
    //         clusters[cluster_index] = std::make_pair(cluster, active_indices.size());
    //         return;
    //     };

    //     std::cout << "HCNNGClusterer::cluster: calling random_clustering_wrapper" << std::endl;
    //     // should populate the clusters sequence
    //     cluster_struct<Point, PointRange, index_type>().random_clustering_wrapper(G, points, cluster_size, assign, 0);

    //     std::cout << "HCNNGClusterer::cluster: filtering empty clusters" << std::endl;

    //     // remove empty clusters
    //     // clusters = parlay::filter(clusters, [&] (parlay::sequence<index_type> cluster) {return cluster.size() > 0;});
    //     clusters = parlay::filter(clusters, [&] (std::pair<index_type*, index_type> cluster) {return cluster.second > 0;});
    //     parlay::sequence<parlay::sequence<index_type>> result(clusters.size());
    //     for (size_t i = 0; i < clusters.size(); i++) {
    //         result[i] = parlay::sequence<index_type>(clusters[i].first, clusters[i].first + clusters[i].second);
    //     }

    //     // free the memory allocated for the clusters
    //     for (size_t i = 0; i < clusters.size(); i++) {
    //         delete[] clusters[i].first;
    //     }

    //     return result;
    // }

    parlay::sequence<parlay::sequence<index_type>> cluster(PointRange points, parlay::sequence<index_type> indices) {
        size_t num_points = indices.size();
        // we make a very sparse sequence to store the clusters
        parlay::sequence<std::pair<index_type*, index_type>> clusters(num_points);
        // lambda to assign a cluster
        auto assign = [&] (Graph<index_type> G, PointRange points, parlay::sequence<index_type>& active_indices, long MSTDeg) {
            if (active_indices.size() == 0) { // this should never happen
                // std::cout << "HCNNGClusterer::cluster: active_indices.size() == 0" << std::endl;
                return;
            }
            if (std::max_element(active_indices.begin(), active_indices.end())[0] > num_points) {
                std::cout << "lambda receiving oversized index" << std::endl;
                return;
            }
            index_type cluster_index = active_indices[0];
            // clusters[cluster_index] = parlay::tabulate(cluster_size, [&] (size_t i) {return static_cast<index_type>(active_indices[i]);});
            index_type* cluster = new index_type[active_indices.size()];

            for (size_t i = 0; i < active_indices.size(); i++) {
                cluster[i] = active_indices[i];
                if (active_indices[i] > num_points) {
                    std::cout << "assign: active_indices[i] > num_points" << std::endl;
                    break;
                }
            }
            clusters[cluster_index] = std::make_pair(cluster, active_indices.size());
            return;
        };

        std::cout << "HCNNGClusterer::cluster: calling random_clustering_wrapper" << std::endl;
        // should populate the clusters sequence
        cluster_struct<Point, PointRange, index_type>().active_indices_rcw(G, points, indices, cluster_size, assign, 0);

        std::cout << "HCNNGClusterer::cluster: filtering empty clusters" << std::endl;

        // remove empty clusters
        // clusters = parlay::filter(clusters, [&] (parlay::sequence<index_type> cluster) {return cluster.size() > 0;});
        clusters = parlay::filter(clusters, [&] (std::pair<index_type*, index_type> cluster) {return cluster.second > 0;});
        parlay::sequence<parlay::sequence<index_type>> result(clusters.size());
        for (size_t i = 0; i < clusters.size(); i++) {
            result[i] = parlay::sequence<index_type>(clusters[i].first, clusters[i].first + clusters[i].second);
        }

        // free the memory allocated for the clusters
        for (size_t i = 0; i < clusters.size(); i++) {
            delete[] clusters[i].first;
        }

        return result;
    }

    parlay::sequence<parlay::sequence<index_type>> cluster(PointRange points) {
        auto active_indices = parlay::tabulate(points.size(), [&] (index_type i) {return i;});
        return this->cluster(points, active_indices);
    }

};

#endif // CLUSTERING_H