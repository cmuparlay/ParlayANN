/* An IVF index */
#ifndef IVF_H
#define IVF_H

#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"
#include "parlay/internal/get_time.h"

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
        auto timer = parlay::internal::timer();
        timer.start();

        // cluster the points
        auto clusterer = HCNNGClusterer<Point, PointRange<T, Point>, index_type>(cluster_size);

        std::cout << "clusterer initialized (" << timer.next_time() <<"s)" << std::endl;

        parlay::sequence<parlay::sequence<index_type>> clusters = clusterer.cluster(points);

        std::cout << "clusters generated" << std::endl;

        // check if there are indices in the clusters that are too large
        // for (size_t i=0; i<clusters.size(); i++){
        //     for (size_t j=0; j<clusters[i].size(); j++){
        //         if (clusters[i][j] >= points.size()){
        //             std::cout << "IVFIndex::fit: clusters[" << i << "][" << j << "] = " << clusters[i][j] << std::endl;
        //         }
        //     }
        // }

        // generate the posting lists
        posting_lists = parlay::tabulate(clusters.size(), [&] (size_t i) {
            return PostingList(points, clusters[i]);
        });

        std::cout << "posting lists generated" << std::endl;

        // check if there are indices in the posting lists that are too large
        // for (size_t i=0; i<posting_lists.size(); i++){
        //     for (size_t j=0; j<posting_lists[i].indices.size(); j++){
        //         if (posting_lists[i].indices[j] >= points.size()){
        //             std::cout << "IVFIndex::fit: posting_lists[" << i << "].indices[" << j << "] = " << posting_lists[i].indices[j] << std::endl;
        //         }
        //     }
        // }

        centroids = parlay::map(posting_lists, [&] (PostingList pl) {return pl.centroid();});
        // serially for debug
        // centroids = parlay::sequence<Point>();
        // for (size_t i=0; i<posting_lists.size(); i++){
        //     centroids.push_back(posting_lists[i].centroid());
        // }
        
        std::cout << "centroids generated" << std::endl;

        dim = points.dimension();
        aligned_dim = points.aligned_dimension();

        std::cout << "fit completed" << std::endl;
    }

    void fit_from_filename(std::string filename, size_t cluster_size=1000){
        PointRange<T, Point> points(filename.c_str());
        std::cout << "points loaded" << std::endl;
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
        auto timer = parlay::internal::timer();
        timer.start();

        this->filters = filters;

        this->filters.print_stats();

        std::cout << this->filters.first_label(42) << std::endl;

        // transpose the filters
        // this->filters.transpose_inplace();

        this->filters.print_stats();

        // std::cout << this->filters.first_label(6) << std::endl;

        // std::cout << "FilteredIVF: filters transposed (" << timer.next_time() << "s)" << std::endl;

        // cluster the points
        auto clusterer = HCNNGClusterer<Point, PointRange<T, Point>, index_type>(cluster_size);

        std::cout << "FilteredIVF: clusterer initialized (" << timer.next_time() << "s)" << std::endl;

        parlay::sequence<parlay::sequence<index_type>> clusters = clusterer.cluster(points);

        std::cout << "FilteredIVF: clusters generated (" << timer.next_time() << "s)" << std::endl;

        // we sort the clusters to facilitate the generation of subset filters
        parlay::parallel_for(0, clusters.size(), [&] (size_t i) {
            std::sort(clusters[i].begin(), clusters[i].end());
        });
        
        std::cout << "FilteredIVF: clusters sorted (" << timer.next_time() << "s)" << std::endl;

        // generate the posting lists
        this->posting_lists = parlay::tabulate(clusters.size(), [&] (size_t i) {
            return PostingList(points, clusters[i], this->filters);
        });

        std::cout << "FilteredIVF: posting lists generated (" << timer.next_time() << "s)" << std::endl;

        this->centroids = parlay::map(this->posting_lists, [&] (PostingList pl) {return pl.centroid();});
        
        std::cout << "FilteredIVF: centroids generated (" << timer.next_time() << "s)" << std::endl;

        this->dim = points.dimension();
        this->aligned_dim = points.aligned_dimension();
    }

    void fit_from_filename(std::string filename, std::string filter_filename, size_t cluster_size=1000){
        PointRange<T, Point> points(filename.c_str());
        std::cout << "points loaded" << std::endl;
        csr_filters filters(filter_filename.c_str());
        std::cout << "filters loaded" << std::endl;
        this->fit(points, filters, cluster_size);
        std::cout << "fit completed" << std::endl;
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