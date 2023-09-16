/* Posting list for the bottom level of an IVF index. */

#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"

#include "point_range.h"

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
    parlay::sequence<T> centroid() {
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
        return parlay::map(result, [](double x) {return static_cast<T>(std::round(x));});
    }

    /* 
    Takes a query point, k, and a sequence of pairs of indices and distances, and if there are points in the posting list closer than the farthest point in the sequence, adds them to the sequence.
     */
    void query(Point query, size_t k, parlay::sequence<std::pair<size_t, float>>& result) {
        float farthest = result[result.size() - 1].second;
        for (size_t i = 0; i < indices.size(); i++) {
            float dist = points[indices[i]].distance(query);
            if (dist < farthest) {
                result.push_back(std::make_pair(indices[i], dist));
                std::sort(result.begin(), result.end(), [](std::pair<size_t, float> a, std::pair<size_t, float> b) {return a.second < b.second;});
                result.pop_back();
                farthest = result[result.size() - 1].second;
            }
        }
    }
};