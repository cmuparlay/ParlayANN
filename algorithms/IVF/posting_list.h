/* Posting list for the bottom level of an IVF index. */

#ifndef POSTING_LIST
#define POSTING_LIST

#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"

#include "../utils/point_range.h"
#include "../utils/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <utility>


/* 
A basic, minimal implementation of a posting list.

The operations here are serial for simplcity, since posting lists should be appropriately small that this is not a bottleneck.
 */
template<typename T, class Point>
struct NaivePostingList {
    PointRange<T, Point> points;
    parlay::sequence<size_t> indices;

    NaivePostingList() {}

    NaivePostingList(PointRange<T, Point> points, parlay::sequence<size_t> indices) : points(points), indices(indices) {}

    // see no reason to store its own centroid, but makes sense to be able to generate it
    /* 
    This generates (serially) the centroid of the points in the posting list.
     */
    Point centroid() {
        parlay::sequence<double> result(points.dimension(), 0);
        for (size_t i = 0; i < indices.size(); i++) {
            T* values = points[indices[i]].get();
            for (size_t j = 0; j < points.dimension(); j++) {
                result[j] += values[j];
            }
        }
        for (size_t j = 0; j < points.dimension(); j++) {
            result[j] /= indices.size();
        }

        parlay::sequence<T> casted_result = parlay::map(result, [](double x) {return static_cast<T>(x);});
        return Point(static_cast<T*>(casted_result.data()), points.dimension(), points.dimension(), 0); // this may cause issues with alignment
        // return parlay::map(result, [](double x) {return static_cast<T>(std::round(x));});
    }

    /* 
    Takes a query point, k, and a sequence of pairs of indices and distances, and if there are points in the posting list closer than the farthest point in the sequence, adds them to the sequence.
     */
    void query(Point query, unsigned int k, parlay::sequence<std::pair<unsigned int, float>>& result) {
        float farthest = result[result.size() - 1].second;
        for (unsigned int i = 0; i < indices.size(); i++) {
            float dist = points[indices[i]].distance(query);
            if (dist < farthest) {
                result.push_back(std::make_pair(indices[i], dist));
                std::sort(result.begin(), result.end(), [](std::pair<unsigned int, float> a, std::pair<unsigned int, float> b) {return a.second < b.second;});
                result.pop_back();
                farthest = result[result.size() - 1].second;
            }
        }
    }
};

#endif // POSTING_LIST