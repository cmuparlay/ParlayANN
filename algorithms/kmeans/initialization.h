/* 
    Initialization methods for k-means clustering.

    methods should be expressed as structs with an operator()

    It's expected that each implementation will:
        - update the centers array with the initial centers
        - update the asg array with the initial assignments

    Implementations which do not update asg (for algorithms which will have to recompute it regardless in the first iteration) should be named with a trailing '_noasg' (e.g. Forgy_noasg)
*/
#ifndef INITERS
#define INITERS

#include "parlay/random.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "distance.h"
//#include "utils/threadlocal.h"
#include "kmeans_bench.h"

#include <string>
#include <stdlib.h>


/* 
    Randomly assigns each point to a cluster, then computes the centers of each 
    cluster.
//  */
// template<typename T>
// struct Forgy {
//     /* 
//     args:
//         v: pointer to flat array of points
//         n: number of points
//         k: number of clusters
//         d: dimension of points
//         centers: pointer to array of centers
//         asg: pointer to array of assignments
//      */
//     void operator()(T* v, size_t n, size_t d, size_t k, float* centers, 
// size_t* asg, Distance& D) {
//         // actually uses the lighter weight i % k version
//         parlay::parallel_for(0, n, [&](size_t i) {
//             asg[i] = i % k;
//         });

//         //compute centers
//         threadlocal::accumulator<float>* acc = new threadlocal::accumulator<float>[k*d];
//         parlay::parallel_for(0, n, [&](size_t i) {
//             size_t c = i % k;
//             for (size_t j = 0; j < d; j++) {
//                 acc[c*d + j].add(v[i*d + j]);
//             }
//         });

//         parlay::parallel_for(0, k, [&](size_t i) {
//             for (size_t j = 0; j < d; j++) {
//                 size_t count = n / k + (i < n % k);
//                 centers[i*d + j] = acc[i*d + j].total() / count;
//             }
//         });
        
//         delete[] acc;
//     }

//     std::string name() {
//         return "Forgy";
//     }
// };

//version of Forgy that actually picks the points randomly
// template<typename T>
// struct ForgyRandom {
//     /* 
//     args:
//         v: pointer to flat array of points
//         n: number of points
//         k: number of clusters
//         d: dimension of points
//         centers: pointer to array of centers
//         asg: pointer to array of assignments
//      */
//     void operator()(T* v, size_t n, size_t d, size_t k, float* centers, 
// size_t* asg, Distance& D) {

//         parlay::random_generator gen(time(0)); //start time a customary "random" #
//         std::uniform_int_distribution<size_t> dis(0, k-1);
        
//         parlay::parallel_for(0, n, [&](size_t i) {
//             auto r = gen[i]; //why is it segmented into two lines like this? idk TODO understand why?
//             asg[i] = dis(r);
//         });
        
//         //making sure that numbers outputted randomly are different per run
//         // for (size_t i = 0; i < 25; i++) {
//         //     std::cout << asg[i] << " ";
//         //     std::cout << std::endl;
//         // }

//         //compute centers
//         threadlocal::accumulator<float>* acc = new threadlocal::accumulator<float>[k*d];
//         parlay::parallel_for(0, n, [&](size_t i) {
//             size_t c = i % k;
//             for (size_t j = 0; j < d; j++) {
//                 acc[c*d + j].add(v[i*d + j]);
//             }
//         });

//         parlay::parallel_for(0, k, [&](size_t i) {
//             for (size_t j = 0; j < d; j++) {
//                 size_t count = n / k + (i < n % k);
//                 centers[i*d + j] = acc[i*d + j].total() / count;
//             }
//         });
        
//         delete[] acc;
//     }

//     std::string name() {
//         return "ForgyRandom";
//     }
// };


/* 
Takes the first k points as centers, then assigns each point to the closest center.
(technically MacQueen (a) in the latex doc)
 */
template<typename T>
struct MacQueen {
    void operator()(T* v, size_t n, size_t d, size_t k, float* centers,
    size_t* asg, Distance& D) {
        // copy first k points into centers
        // this should hopefully compile to a memcpy if T is a float
        for (size_t i = 0; i < k*d; i++) {
            centers[i] = v[i];
        }

        // assign each point to the closest center
        std::fill(asg, asg + n, 0);
        parlay::parallel_for(0, n, [&](size_t i) {
            float min_dist = D.distance(v + i*d, centers, d);
            for (size_t j = 1; j < k; j++) {
                float dist = D.distance(v + i*d, centers + j*d, d);
                if (dist < min_dist) {
                    min_dist = dist;
                    asg[i] = j;
                }
            }
        });
        return;
    };

    std::string name() {
        return "MacQueen";
    };
};


//Lazy start makes the first k points the first k centers
//Then assigns cyclically
template<typename T>
struct LazyStart {
    void operator()(T* v, size_t n, size_t d, size_t k, float* c, 
    size_t* asg, Distance& D) {
        for (size_t i = 0; i < k; i++) {
            for (size_t j = 0; j < d; j++) {
                c[i*d + j] = v[i*d + j];

            }

        }
        for (size_t i = 0; i < n; i++) {
            asg[i] = i % k;
        }
    }

    std::string name() {
        return "Lazy";
    }
};

//Guy's kmeans++ code:
//  // add k initial points by the kmeans++ rule
//   // random initial center
//   Points kpts({pts[rand()%k]});
//   for (int i=1; i < k; i++) {
//     // find the closest center for all points
//     auto dist = parlay::map(pts, [&] (const Point& p) {
//       return distance(p, kpts[closest_point(p, kpts, distance)]);});

//     // pick with probability proportional do distance (squared)
//     auto [sums, total] = scan(dist);
//     auto pos = dis(rand) * total;
//     auto j = std::upper_bound(sums.begin(), sums.end(), pos) - sums.begin()-1;

//     // add to the k points
//     kpts.push_back(pts[j]);
//   }
template<typename T>
struct KmeansPlusPlus {

    //given some number of centers, find the distance between the pt id and its the closest center 
    float closest_dist(T* v, size_t n, size_t d, size_t k, float* c, Distance& D, size_t pt_id, size_t num_centers, parlay::sequence<size_t>& center_rang) {
        //need to copy point to float buffer first!
        float buf[2048];
        for (size_t i = 0; i < d; i++) {
            buf[i] = static_cast<float>(v[pt_id*d+i]);
        }

        auto distances = parlay::map(center_rang, [&] (size_t i) {
            return D.distance(buf,c+i*d,d);
        });
        return *parlay::min_element(distances);

    }

    void operator()(T* v, size_t n, size_t d, size_t k, float* c, size_t* asg, Distance& D) {
        parlay::random_generator rand(time(0)); // time 0 a random seed
        std::uniform_real_distribution<> dis(0.0,1.0);

        //TODO do better this is inefficient! (need for the map)
        parlay::sequence<size_t> rang = parlay::tabulate(n,[&] (size_t i) {return i;});

        size_t first_point_choice = rand() % n;
        for (size_t coord = 0; coord < d; coord++) {
            c[coord] = v[first_point_choice*d+coord];
        }

        //std::cout << "made it hey 2" << std::endl;

        for (size_t i = 1 ; i < k; i++) {

            parlay::sequence<size_t> center_rang = parlay::tabulate(i,[&] (size_t j) {
                return j;
            });

           // std::cout << "made it hey 3" << std::endl;


            auto dist = parlay::map(rang, [&] (size_t j) {
                return closest_dist(v,n,d,k,c,D,j,i, center_rang);
            });

          //  std::cout << "made it hey 4" << std::endl;


            auto [sums, total] = parlay::scan(dist);

            // std::cout << "dummy distance " << i << " " << D.distance(v,c,d) << std::endl;

            // std::cout << "printing scan " << i << std::endl;
            // std::cout << "total: " << total << std::endl;
            // std::cout << "sums: ";
            // for (size_t j = 0; j < sums.size(); j++) {
            //     std::cout << sums[j] << " ";
            // }
            //std::cout << std::endl;
            auto pos = dis(rand) * total;
            auto chosen_point_index = std::upper_bound(sums.begin(),sums.end(),pos) - sums.begin() - 1;
            // std::cout << "chosen index is " << chosen_point_index << std::endl;

            for (size_t coord = 0; coord < d; coord++) {
                c[i*d+coord] = v[chosen_point_index*d+coord];
            }


        }

        parlay::sequence<size_t> final_center_rang = parlay::tabulate(k,[&] (size_t j) {
            return j;
        });

        //std::cout << "made it hey 5" << std::endl;

        //set initial assignments in the naive way
        parlay::parallel_for(0,n,[&] (size_t i) {
            float buf[2048];
            for (size_t j = 0; j < d; j++) {
                buf[j] = static_cast<float>(v[i*d+j]);
            }
            auto distances = parlay::map(final_center_rang,[&] (size_t j) {
                return D.distance(buf,c+j*d,d);
            });
            asg[i] = parlay::min_element(distances)-distances.begin();
        });

        //std::cout << "made it hey 6" << std::endl;

    }

     std::string name() {
        return "Kmeans++";
    }

};

#endif //INITERS

