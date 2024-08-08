/* An IVF index */
#pragma once

#include "parlay/internal/get_time.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/delayed_sequence.h"
#include "parlay/slice.h"
#include "parlay/internal/group_by.h"

#include "../../algorithms/HCNNG/clusterEdge.h"
#include "utils/beamSearch.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/types.h"
#include "../../algorithms/vamana/index.h"
#include "clustering.h"
#include "posting_list.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <filesystem>
#include <fstream>
#include <utility>
#include <variant>
#include <vector>

/* An extremely minimal IVF index which only returns full posting lists-worth of
   points

    For convenience, also does clustering on the points during initialization
   with a provided clusterer
*/
template <typename T, typename Point, typename indexType>
struct PostingListIndex {
  PointRange<T, Point>& Points;
  parlay::sequence<parlay::sequence<indexType>> clusters;
  parlay::sequence<Point> centroids;
  std::unique_ptr<T[]>
     centroid_data;   // those point objects store const pointers to their data,
                      // so we need to keep it around
  size_t dim;
  size_t aligned_dim;
  size_t n;                                   // n points in the index
  indexType id;                              // id of this index
  std::pair<size_t, size_t> cluster_params;   // build params for the clustering


  PostingListIndex(PostingListIndex<T, Point, indexType>&& other) {
    std::cout << "Move constructor called" << std::endl;
  };

  void save(char* filename){
    save_posting_list(std::string(filename));
  }

  PostingListIndex(char* index_path){
    load_posting_list(std::string(index_path));
  }

  template <typename Clusterer>
  PostingListIndex(PointRange<T, Point>& Points, Clusterer clusterer, indexType id,
                   char* index_path)
      : id(id),
        Points(Points)
     {
    auto indices = parlay::tabulate(Points.size(), [&] (size_t i){return static_cast<indexType>(i);});

 
    dim = Points.dimension();
    aligned_dim = Points.aligned_dimension();

    n = Points.size();
    cluster_params = clusterer.get_build_params();

    if (index_path != nullptr &&
        std::filesystem::exists(pl_filename(std::string(index_path)))) {
        load_posting_list(std::string(index_path));
    } else {

      std::cout << "Calculating clusters" << std::endl;
      clusters = clusterer.two_round_cluster(Points, indices);

      centroid_data =
         std::make_unique<T[]>(clusters.size() * aligned_dim);

      if (clusters.size() == 0) {
        throw std::runtime_error("PostingListIndex: no clusters generated");
      }

      std::cout << "Calculating centroids" << std::endl;
      for (size_t i = 0; i < clusters.size(); i++) {
        size_t offset = i * aligned_dim;
        parlay::sequence<double> tmp_centroid(dim);
        for (size_t p = 0; p < clusters[i].size();
             p++) {   // for each point in this cluster
          for (size_t d = 0; d < dim; d++) {   // for each dimension
            tmp_centroid[d] += Points[clusters[i][p]][d];
          }
        }
        // divide by the number of points in the cluster and assign to centroid
        // data
        for (size_t d = 0; d < dim; d++) {   // for each dimension
          centroid_data[offset + d] = static_cast<T>(
             std::round(tmp_centroid[d] / clusters[i].size()));
        }

        centroids.push_back(Point(centroid_data.get() + offset,
                                        dim, aligned_dim, i));
      }

      std::cout << "Found " << centroids.size() << " centroids" << std::endl;
    }


  }

  

  parlay::sequence<indexType> sorted_near(Point query, int target_points) const {
    parlay::sequence<std::pair<indexType, float>> nearest_centroids(clusters.size());
    
    for (size_t i = 0; i < clusters.size(); i++) {
      float dist = query.distance(centroids[i]);
      nearest_centroids[i] = std::make_pair(i, dist);
    }

    std::sort(
       nearest_centroids.begin(), nearest_centroids.end(),
       [&](std::pair<indexType, float> a, std::pair<indexType, float> b) {
         return a.second < b.second;
       });

    auto result = parlay::sequence<indexType>();
    size_t i = 0;
    do {
      result.append(clusters[nearest_centroids[i].first]);
      i++;
    } while (result.size() < target_points && i < clusters.size());

    std::sort(result.begin(), result.end());

    return result;
  }


  /* computes the knn with the ivf index */
  std::pair<parlay::sequence<std::pair<indexType, float>>, size_t> ivf_knn(
     Point query, int k, int n_probes) {
      // we do linear traversal over the centroids to get the nearest ones
      size_t pl_frontier_size = n_probes < centroids.size() ? n_probes : centroids.size();
      parlay::sequence<std::pair<indexType, float>> pl_frontier(pl_frontier_size, std::make_pair(0, std::numeric_limits<float>::max()));

      size_t dist_cmps = centroids.size();

      for (indexType i = 0; i < centroids.size(); i++) {
        float dist = query.distance(centroids[i]);
        if (dist < pl_frontier[pl_frontier_size - 1].second) {
          pl_frontier.pop_back();
          pl_frontier.push_back(std::make_pair(i, dist));
          std::sort(pl_frontier.begin(), pl_frontier.end(),
                    [&](std::pair<indexType, float> a,
                        std::pair<indexType, float> b) {
                      return a.second < b.second;
                    });
        }
      }

      // now we do search on the points in the posting lists
      parlay::sequence<std::pair<indexType, float>> frontier(
         k, std::make_pair(0, std::numeric_limits<float>::max()));

      for (indexType i = 0; i < pl_frontier.size(); i++) {
        dist_cmps += clusters[pl_frontier[i].first].size();
        for (indexType j = 0; j < clusters[pl_frontier[i].first].size(); j++) {
          float dist = query.distance(Points[clusters[pl_frontier[i].first][j]]);
          if (dist < frontier[k - 1].second) {
            frontier.pop_back();
            frontier.push_back(std::make_pair(clusters[pl_frontier[i].first][j], dist));
            std::sort(frontier.begin(), frontier.end(),
                      [&](std::pair<indexType, float> a,
                          std::pair<indexType, float> b) {
                        return a.second < b.second;
                      });
          }
        }
    }

    // std::cout <<"printing out frontier" <<std::endl;

    // for(int i=0; i< frontier.size(); i++){
    //   std::cout << " dist: " << frontier[i].second;
    // }

    // std::cout << std::endl;

    //std::cout <<" cmps:" << dist_cmps<< std::endl;

    return std::make_pair(frontier, dist_cmps);
  }

  /* computes range results with the ivf index */
  std::pair<parlay::sequence<std::pair<indexType, float>>, size_t> ivf_range(
     Point query, double rad, int n_probes) {
      // we do linear traversal over the centroids to get the nearest ones
      // TODO should we exclude centroids that are too far away from the radius?
      size_t pl_frontier_size = n_probes < centroids.size() ? n_probes : centroids.size();
      parlay::sequence<std::pair<indexType, float>> pl_frontier(pl_frontier_size, std::make_pair(0, std::numeric_limits<float>::max()));

      size_t dist_cmps = centroids.size();

      for (indexType i = 0; i < centroids.size(); i++) {
        float dist = query.distance(centroids[i]);
        if (dist < pl_frontier[pl_frontier_size - 1].second) {
          pl_frontier.pop_back();
          pl_frontier.push_back(std::make_pair(i, dist));
          std::sort(pl_frontier.begin(), pl_frontier.end(),
                    [&](std::pair<indexType, float> a,
                        std::pair<indexType, float> b) {
                      return a.second < b.second;
                    });
        }
      }

      // now we do search on the points in the posting lists
      parlay::sequence<std::pair<indexType, float>> frontier;

      for (indexType i = 0; i < pl_frontier.size(); i++) {
        dist_cmps += clusters[pl_frontier[i].first].size();
        for (indexType j = 0; j < clusters[pl_frontier[i].first].size(); j++) {
          float dist = query.distance(Points[clusters[pl_frontier[i].first][j]]);
          if (dist < rad) {
            frontier.push_back(std::make_pair(clusters[pl_frontier[i].first][j], dist));
          }
        }
      }

      return std::make_pair(frontier, dist_cmps);
  } 
  /* computes range results with the ivf index, and returns two comparsion number.
  First distance argument as  */
  std::pair<parlay::sequence<std::pair<indexType, float>>, std::pair<size_t,size_t>> ivf_range_dist_cmp(
     Point query, double rad, int n_probes) {
      // we do linear traversal over the centroids to get the nearest ones
      // TODO should we exclude centroids that are too far away from the radius?
      size_t pl_frontier_size = n_probes < centroids.size() ? n_probes : centroids.size();
      parlay::sequence<std::pair<indexType, float>> pl_frontier(pl_frontier_size, std::make_pair(0, std::numeric_limits<float>::max()));

      size_t dist_cmps_1 = centroids.size();
      size_t dist_cmps_2 = 0;

      for (indexType i = 0; i < centroids.size(); i++) {
        float dist = query.distance(centroids[i]);
        if (dist < pl_frontier[pl_frontier_size - 1].second) {
          pl_frontier.pop_back();
          pl_frontier.push_back(std::make_pair(i, dist));
          std::sort(pl_frontier.begin(), pl_frontier.end(),
                    [&](std::pair<indexType, float> a,
                        std::pair<indexType, float> b) {
                      return a.second < b.second;
                    });
        }
      }

      // now we do search on the points in the posting lists
      parlay::sequence<std::pair<indexType, float>> frontier;

      for (indexType i = 0; i < pl_frontier.size(); i++) {
        dist_cmps_2 += clusters[pl_frontier[i].first].size();
        for (indexType j = 0; j < clusters[pl_frontier[i].first].size(); j++) {
          float dist = query.distance(Points[clusters[pl_frontier[i].first][j]]);
          if (dist < rad) {
            frontier.push_back(std::make_pair(clusters[pl_frontier[i].first][j], dist));
          }
        }
      }

      return std::make_pair(frontier, std::make_pair(dist_cmps_1,dist_cmps_2));
  } 



  std::string pl_filename(std::string filename_prefix) {
    return filename_prefix + std::to_string(id) + "_postingList_" +
           std::to_string(cluster_params.first) + "_" +
           std::to_string(cluster_params.second) + ".bin";
  }

  /* Saves the posting list to a file named based on the cluster params */
  void save_posting_list(const std::string& filename_prefix) {
    std::string filename = pl_filename(filename_prefix);
    std::ofstream output(filename, std::ios::binary);

    if (!output.is_open()) {
      throw std::runtime_error("PostingListIndex: could not open file " +
                               filename);
    }

    // Write number of centroids/posting lists
    size_t num_centroids = clusters.size();
    output.write(reinterpret_cast<const char*>(&num_centroids), sizeof(size_t));

    // Write centroids
    size_t centroid_bytes = sizeof(T) * num_centroids * aligned_dim;
    output.write(reinterpret_cast<const char*>(centroid_data.get()),
                 centroid_bytes);

    // For CSR-like format
    parlay::sequence<indexType> indices =
       parlay::sequence<indexType>::uninitialized(n);
    parlay::sequence<indexType> indptr(1, 0);   // Start with an initial zero
    indptr.reserve(num_centroids + 1);

    for (const auto& list : clusters) {
      memcpy(indices.data() + indptr.back(), list.data(),
             sizeof(indexType) * list.size());
      indptr.push_back(indptr.back() + list.size());
    }

    // Write indptr array
    output.write(reinterpret_cast<const char*>(indptr.data()),
                 sizeof(indexType) * indptr.size());

    // Write indices array
    output.write(reinterpret_cast<const char*>(indices.data()),
                 sizeof(indexType) * indices.size());

    output.close();
  }

  /* Loads the posting list from a file named based on the cluster params */
  void load_posting_list(std::string filename_prefix) {
    std::string filename = pl_filename(filename_prefix);
    std::ifstream input(filename, std::ios::binary);

    if (!input.is_open()) {
      throw std::runtime_error("PostingListIndex: could not open file " +
                               filename);
    }

    // Read number of centroids/posting lists
    size_t num_centroids;
    input.read(reinterpret_cast<char*>(&num_centroids), sizeof(size_t));

    // Read centroids
    size_t centroid_bytes = sizeof(T) * num_centroids * aligned_dim;
    centroid_data =
       std::make_unique<T[]>(num_centroids * aligned_dim);
    input.read(reinterpret_cast<char*>(centroid_data.get()),
               centroid_bytes);

    for (size_t i = 0; i < num_centroids; i++) {
      centroids.push_back(
         Point(centroid_data.get() + i * aligned_dim, dim,
               aligned_dim, i));
    }
    // Read indptr array
    parlay::sequence<indexType> indptr(num_centroids + 1);
    input.read(reinterpret_cast<char*>(indptr.data()),
               sizeof(indexType) * indptr.size());

    // Read indices array
    parlay::sequence<indexType> indices(indptr.back());
    input.read(reinterpret_cast<char*>(indices.data()),
               sizeof(indexType) * indices.size());

    input.close();

    // Convert indptr and indices to CSR-like format
    clusters = parlay::sequence<parlay::sequence<indexType>>(
       num_centroids);   // should in theory be ininitialized but that causes a
                         // segfault
    for (size_t i = 0; i < num_centroids; i++) {
      clusters[i].reserve(indptr[i + 1] - indptr[i]);
      // this is heinous hopefully this is not needed
      for (size_t j = indptr[i]; j < indptr[i + 1]; j++) {
        clusters[i].push_back(indices[j]);
      }
    }
  }



  size_t footprint() const {
    return sizeof(indexType) * n // posting lists
            + sizeof(T) * aligned_dim * clusters.size(); // centroids
  }

}; //end PostingListIndex






