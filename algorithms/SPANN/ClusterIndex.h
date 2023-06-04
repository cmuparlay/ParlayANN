#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "utils/indexTools.h"
#include "parlay/random.h"
#include "parlay/sequence.h"

#include <utility>
#include <iostream>

/* 
    decomposing general case of clustering-based IVF-esqe indices. 

    independent components:
    - shard: a struct representing/storing a subset of the data, with a representative vector (centroid) and an index supporting search within the shard.
    - clustering: a function that takes a sequence of vectors and fills a sequence with sequences of indices representing shards.
    - router: a function which takes a sequence of shards and a query vector and fills a sequence with indices of shards that should be searched.
*/


/* 
    The struct representing a shard of the data. Supports indexing and searching within the shard.

    The constructor does not have arguments for the index's `build_index` for the sake of generality, so I recommend a wrapper struct, default values, or overloading `build_index` to provide the parameters.

    `index_type` here would be like `hcnng_index<T>`. Would be nice if this included a search method but they don't seem to by default.
 */
template <typename T, typename index_type>
struct shard {
    Tvec_point<T>* centroid; // representative vector for the shard (may not literally be a centroid)
    index_type index; // ann index used to search within the shard
    parlay::sequence<Tvec_point<T>>* data; // pointer to the data sequence
    parlay::sequence<size_t> indices; // indices of vectors in the shard

    shard(parlay::sequence<Tvec_point<T>>* data, parlay::sequence<size_t> indices, size_t centroid_index, *Distance D, int dim, int max_deg) {
        this->data = data;
        this->indices = indices;
        this->centroid = &((*data)[centroid_index]);
        this->index = index_type(max_deg, D, dim);

        auto data_slice = parlay::map(indices, [&] (size_t i) {
            return &((*data)[i]);
        }

        this->index.build_index(data_slice, index_alpha, index_beta);
    }

    // TODO: implement a method to search within the shard, because at least hcnng_index does not have a search method. Said method is currently expected to return a sequence of tuples of the form (distance to query, pointer to vector).
};


/* 
    A function which takes a sequence of vectors and fills a sequence with sequences of indices representing shards.

    @param data: sequence of vectors to be clustered
    @param shards: pointer to sequence of sequences to be filled. Should be initialized to empty sequences.
    @param centroid_indices: pointer to sequence of indices to be filled. Should be initialized to empty sequence.
    @param k: number of shards to be created
    
    This function signature does not include any parameters for the clustering algorithm for the sake of generality. I recommend a lambda function or struct with function call operator to handle said parameters.
 */
template <typename T>
void cluster(parlay::sequence<Tvec_point<T>>* data, 
                    parlay::sequence<parlay::sequence<size_t>>* shards, 
                    parlay::sequence<size_t>* centroid_indices,
                    size_t k);

/* 
    A function which takes a sequence of shards and a query vector and fills sequence with indices of shards that should be searched.

    @param shards: pointer to sequence of shards to be searched
    @param to_search: pointer to sequence of indices to be filled. Should be initialized to empty sequence.
    @param query: query vector
    @param k: max number of shards to be returned
    
    This function signature does not include any parameters for the routing algorithm for the sake of generality. I recommend a lambda function or struct with function call operator to handle said parameters.
 */
template <typename T>
void route(parlay::sequence<shard>* shards, 
           parlay::sequence<size_t>* to_search, 
           Tvec_point<T>* query, 
           size_t k);


// function pointers for each component
using Clustering = void (*)(parlay::sequence<Tvec_point<T>>* data, 
                            parlay::sequence<parlay::sequence<size_t>>* shards, 
                            parlay::sequence<size_t>* centroid_indices,
                            size_t k);

using Routing = void (*)(parlay::sequence<shard>* shards, 
                         parlay::sequence<size_t>* to_search, 
                         Tvec_point<T>* query, 
                         size_t k);


template <typename T, typename index_type>
struct ClusterIndex {
    Clustering cluster_func; // function pointer to clustering function
    Routing route_func; // function pointer to routing function
    parlay::sequence<Tvec_point<T>>* data; // pointer to data sequence
    parlay::sequence<shard<T, index_type>> shards; // sequence of shards

    size_t k; // number of shards
    Distance* D; // distance function
    size_t dim; // dimension of vectors

    ClusterIndex(Clustering cluster_func, Routing route_func, parlay::sequence<Tvec_point<T>>* data, size_t k, Distance* D, size_t dim) {
        this->cluster_func = cluster_func;
        this->route_func = route_func;
        this->data = data;
        this->k = k;
        this->D = D;
        this->dim = dim;
    }

    /* 
        Does clustering and initializes shards.
     */
    void build_index() {
        parlay::sequence<size_t> centroid_indices;
        parlay::sequence<parlay::sequence<size_t>> shard_indices;

        this->cluster_func(this->data, &shard_indices, &centroid_indices, this->k);

        this->shards = parlay::map(centroid_indices, [&] (size_t i) {
            return shard<T, index_type>(this->data, this->shard_indices[i], i, this->D, this->dim);
        });
    }

    /* 
        Searches the index for the k nearest neighbors of the query vector.

        @param query: query vector
        @param k: max number of neighbors to be returned

        @return: sequence of pointers to vectors in the data sequence

        This doesn't currently return the distance to each neighbor, but that could easily be added.
     */
    parlay::sequence<Tvec_point<T>*> search(Tvec_point<T>* query, size_t k) {
        parlay::sequence<size_t> to_search;
        this->route_func(this->shards, &to_search, query, k);

        // should be parlay::sequence<parlay::sequence<pair<double, Tvec_point<T>*>>> for a sequence of (distance, pointer to vector) pairs
        auto search_results = parlay::map(to_search, [&] (size_t i) {
            return this->shards[i].index.search(query, k);
        });

        parlay::sort_inplace(search_results, [&] (auto a, auto b) {
            return a.first < b.first;
        });

        return parlay::unique(parlay::map(search_results, [&] (auto a) {
            return a.second;
        }));

    }

};