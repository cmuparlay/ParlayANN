/* An IVF index */
#ifndef IVF_H
#define IVF_H

#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"

#include "../utils/point_range.h"
#include "../utils/types.h"
#include "../HCNNG/clusterEdge.h"
#include "posting_list.h"
#include "clustering.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <utility>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using NeighborsAndDistances = std::pair<py::array_t<unsigned int>, py::array_t<float>>;

/* A reasonably extensible ivf index */
template<typename T, typename Point, typename PostingList>
struct IVFIndex {
    // PointRange<T, Point> points;
    parlay::sequence<PostingList> posting_lists = parlay::sequence<PostingList>();
    parlay::sequence<Point> centroids = parlay::sequence<Point>();
    size_t dim;
    size_t aligned_dim;

    IVFIndex() {}

    // IVFIndex(PointRange<T, Point> points) : points(points) {}

    virtual void fit(PointRange<T, Point> points, size_t cluster_size=1000){
        // cluster the points
        auto clusterer = HCNNGClusterer<Point, PointRange<T, Point>, index_type>(cluster_size);
        parlay::sequence<parlay::sequence<index_type>> clusters = clusterer.cluster(points);

        // generate the posting lists
        posting_lists = parlay::tabulate(clusters.size(), [&] (size_t i) {
            return PostingList(points, clusters[i]);
        });
        centroids = parlay::map(posting_lists, [&] (PostingList pl) {return pl.centroid();});
        
        dim = points.dimension();
        aligned_dim = points.aligned_dimension();
    }

    void fit_from_filename(std::string filename, size_t cluster_size=1000){
        PointRange<T, Point> points(filename.c_str());
        this->fit(points, cluster_size);
    }

    /* A utility function to do nearest k centroids in linear time */
    parlay::sequence<unsigned int> nearest_centroids(Point query, unsigned int n){
        parlay::sequence<std::pair<unsigned int, float>> nearest_centroids = parlay::tabulate(n, [&] (unsigned int i) {
            return std::make_pair(std::numeric_limits<unsigned int>().max(), std::numeric_limits<float>().max());
        });
        for (unsigned int i=0; i<n; i++){
            float dist = query.distance(centroids[i]);
            if(dist < nearest_centroids[n-1].second){
                nearest_centroids[n-1] = std::make_pair(i, dist);
                std::sort(nearest_centroids.begin(), nearest_centroids.end(), [&] (std::pair<unsigned int, float> a, std::pair<unsigned int, float> b) {
                    return a.second < b.second;
                });
            }
        }
        return parlay::map(nearest_centroids, [&] (std::pair<unsigned int, float> p) {return p.first;});
    }

    NeighborsAndDistances batch_search(py::array_t<T, py::array::c_style | py::array::forcecast> &queries, uint64_t num_queries, uint64_t knn, uint64_t n_lists){
        py::array_t<unsigned int> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});

        parlay::parallel_for(0, num_queries, [&] (unsigned int i){
            Point q = Point(queries.data(i), dim, dim, i);
            parlay::sequence<unsigned int> nearest_centroid_ids = nearest_centroids(q, n_lists);

            parlay::sequence<std::pair<unsigned int, float>> frontier = parlay::tabulate(knn, [&] (unsigned int i) {
                return std::make_pair(std::numeric_limits<unsigned int>().max(), std::numeric_limits<float>().max());
            });

            for (unsigned int j=0; j<nearest_centroid_ids.size(); j++){
                posting_lists[nearest_centroid_ids[j]].query(q, knn, frontier);
            }

            // this sort should be redundant
            // std::sort(frontier.begin(), frontier.end(), [&] (std::pair<unsigned int, float> a, std::pair<unsigned int, float> b) {
            //     return a.second < b.second;
            // });
            for (unsigned int j=0; j<knn; j++){
                ids.mutable_data(i)[j] = frontier[j].first;
                dists.mutable_data(i)[j] = frontier[j].second;
            }
        });

        // std::cout << "parfor done" << std::endl;

        return std::make_pair(std::move(ids), std::move(dists));
    }

    void print_stats(){
        size_t total = 0;
        size_t max = 0;
        size_t min = std::numeric_limits<size_t>().max();
        for (size_t i=0; i<posting_lists.size(); i++){
            size_t s = posting_lists[i].indices.size();
            total += s;
            if (s > max) max = s;
            if (s < min) min = s;
        }
        std::cout << "Total number of points: " << total << std::endl;
        std::cout << "Number of posting lists: " << posting_lists.size() << std::endl;
        std::cout << "Average number of points per list: " << total / posting_lists.size() << std::endl;
        std::cout << "Max number of points in a list: " << max << std::endl;
        std::cout << "Min number of points in a list: " << min << std::endl;
    }
};

template<typename T, typename Point, typename PostingList>
struct FilteredIVFIndex : IVFIndex<T, Point, PostingList> {
    csr_filters filters;

    FilteredIVFIndex() {}

    void fit(PointRange<T, Point> points, csr_filters& filters, size_t cluster_size=1000){
        this->filters = filters.transpose();
        // cluster the points
        auto clusterer = HCNNGClusterer<Point, PointRange<T, Point>, index_type>(cluster_size);
        parlay::sequence<parlay::sequence<index_type>> clusters = clusterer.cluster(points);

        // generate the posting lists
        this->posting_lists = parlay::tabulate(clusters.size(), [&] (size_t i) {
            return PostingList(points, clusters[i], this->filters);
        });
        this->centroids = parlay::map(this->posting_lists, [&] (PostingList pl) {return pl.centroid();});
        
        this->dim = points.dimension();
        this->aligned_dim = points.aligned_dimension();
    }

    void fit_from_filename(std::string filename, std::string filter_filename, size_t cluster_size=1000){
        PointRange<T, Point> points(filename.c_str());
        csr_filters filters(filter_filename.c_str());
        this->fit(points, filters, cluster_size);
    }

    /* The use of vector here is because that supposedly allows us to take python lists as input, although I'll believe it when I see it.
    
    This is incredibly easy, but might be slower than parsing the sparse array of filters in C++.  */
    NeighborsAndDistances batch_filter_search(py::array_t<T, py::array::c_style | py::array::forcecast> &queries, const std::vector<QueryFilter>& filters, uint64_t num_queries, uint64_t knn, uint64_t n_lists) {
        py::array_t<unsigned int> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});

        parlay::parallel_for(0, num_queries, [&] (unsigned int i){
            Point q = Point(queries.data(i), this->dim, this->dim, i);
            const QueryFilter& filter = filters[i];
            parlay::sequence<unsigned int> nearest_centroid_ids = this->nearest_centroids(q, n_lists);

            // maybe should be sequential? a memcopy?
            parlay::sequence<std::pair<unsigned int, float>> frontier = parlay::tabulate(knn, [&] (unsigned int i) {
                return std::make_pair(std::numeric_limits<unsigned int>().max(), std::numeric_limits<float>().max());
            });

            for (unsigned int j=0; j<nearest_centroid_ids.size(); j++){
                this->posting_lists[nearest_centroid_ids[j]].filtered_query(q, filter, knn, frontier);
            }

            for (unsigned int j=0; j<knn; j++){
                ids.mutable_data(i)[j] = frontier[j].first;
                dists.mutable_data(i)[j] = frontier[j].second;
            }
        });

        // std::cout << "parfor done" << std::endl;

        return std::make_pair(std::move(ids), std::move(dists));
    }
};

#endif // IVF_H