#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/random.h"
#include "utils/indexTools.h"

#include <algorithm>
#include <set>

// We adapt the kmeans++ implementation from parlaylib/examples/kmeans_pp.h
// to work with our data structures and distance functions.

const size_t MAX_ITER = 1000;

using Point = Tvec_point<T>;
using Points = parlay::sequence<Point>;

size_t closest_point(Point& p, Points& pts, Distance& D, int dim){
    auto dist = parlay::map(pts, [&] (const Point& q) {
        return D(p, q, dim);
    });
    return parlay::min_element(dist) - dist.begin();
}

auto kmeans(Points& pts, int k, Distance& D, double epsilon, int dim){
    size_t n = pts.size();
    parlay::random_generator rand;
    std::uniform_real_distribution<> dis(0.0,1.0);

    // add k initial points by the kmeans++ rule
    // random initial center
    Points kpts = parlay::sequence<Point>({pts[rand()%pts.size()]});
    for (int i=1; i<k; i++){
        // find the closest center for each point
        auto dist = parlay::map(pts, [&] (const Point& p) {
            return D(p, kpts[closest_point(p, kpts, D, dim)], dim);
        });

        // choose a new center with probability proportional to distance squared
        auto [sums, total] = parlay::scan(dist);
        auto pos = dis(rand) * total;
        
    }
}
