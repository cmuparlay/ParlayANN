README for kmeans code

Andrew Brady, acbrady2020@gmail.com
December 2023

Common notation;
ith = i^{th}
seq = sequence
TODO = area that could be improved

Common variable names/notation:
v = vertices = a flat array containing all of the points we want to cluster. Of size n*ad
n = number of points
d = number of dimensions in a point
ad = aligned dimension = number of dimensions of a point, including padding
k = number of centers
c = centers = a flat array containing all of the centers, which we are using k-means iterations to incrementally calculate. Of size k*ad.
Example: if you had an array of points v with d=10, ad=128, you would access the jth coordinate ith point at v[i*ad+j] 

asg = assignments = a flat array containing all of the assignments of points. 
Example: if asg[i]=m, then the ith point in our dataset has been assigned to the center with id m

D = Distance calculation object, defined in distance.h. Given two pointers and d, calling D.distance calculates the distance between them. When both pointers are floats, then D will do a vectorized SIMD distance calculation, which is much faster.

max_iter = max # of iterations allowed before run is cut off
epsilon = variable for early cutoff of k-means run. After each iteration, we check if the max movement of any center was less than epsilon, and if it is then we stop early
logger = logging object for kmeans run, see kmeans_bench object
suppress_logging = boolean flag used to determine whether we log statistics on a kmeans function. We might set suppress_logging to true (prevents logging) if we wanted to reduce the print output of a kmeans run, or if we are running kmeans as a subroutine for another kmeans algorithm
noverk = n / k = n over k
uint8_t = unsigned int of 8 bits type (int that goes 0-255)
msse = mean sum of square error = error = measure of how good/bad a chosen set of centers and assignments are
yy = yinyang (kmeans)
t = for yinyang, the # of center groups

id <- When referring to points or centers, the id of that point/center is the position of that point/center in the array. For example, if we have n points with dimension 100, the 3rd point in that array has id 3. Thus the ids of points go 0, 1, ..., n-1. 

Common template types:
T = point data type e.g. float, uint8_t, int8_t
CT = Center (data) type e.g. float, double
index_type = type used for the indexing/assignments of points e.g., size_t
Point = ParlayANN point object used (e.g. Euclidian_Point<uint8_t>)
CenterPoint = ParlayANN center point object used (e.g. Euclidian_Point<float>)



LSH initialization:
#include "initialization.h"
then do
LSH<T,float> init2;
init2(v, n, d, ad, k, c3, asg3,D);

Running kmeans code

To run an example of kmeans, use the kmeans.cpp file code. For example, see akmeans.sh.
Run by navigating to algorithms/kmeans folder then running "sh akmeans.sh"

In the command: 
$PARANN_DIR/bazel-bin/algorithms/kmeans/kmeans_run -k 200 -i /ssd1/anndata/bigann/base.1B.u8bin.crop_nb_1000000 -f bin -t uint8 -D fast -m 5 -bench_version two

-k 200: set k to 200
-i /ssd1/anndata/bigann/base.1B.u8bin.crop_nb_1000000 : take input from this file. This file is a bin file of 1'000'000 points each containing 128 coordinates that are 1 byte ints, a slice of the bigann dataset. 
-f bin: filetype is bin
-t uint8: type of point is uint8_t
-D fast: use our standard vectorized Euclidean distance function (recommended)
-m 5: set max_iter to 5
-bench_version two: run the code in the bench_two function. bench_two is the function to change if you want a specific algorithm, comparison to run.

WARNING: MIPS distance untested, probably does not work. Euclidean distance does work though.

Benching kmeans code

In addition to the functionality provided above, the bench.cpp file allows for benching multiple values of n,d,k,max_iter at once. 
Use sh abench.sh to run this style of benching (see abench.sh for explanation).

Testing kmeans code

To run a variety of unit tests, do sh test.sh
-----------
WARNING: with LSH initialization k-means likely will take all of the points from a center. In which case the tests will fail, because currently the written tests require all centers to be nonempty. With a Lazy initialization, having testing that requires all centers to be nonempty is nice and catches bugs. But the testing would need to be adapted in the case of an LSH initialization. Perhaps add a check to make sure at least 1/2 the centers are nonempty, and in the rest of the test allow an empty center.

LSH initialization can easily leave a center without any points because LSH initialized centers are averages of points and are NOT points in the dataset. On the other hand, Lazy initialized centers ARE points in the dataset. So with a Lazy initialization, it is very hard for a center to lose the point it was initially assigned to. 
------------

The provided unit tests run k-means to convergence in a variety of situations and make sure that the end result has a local minimum of the SSE (points assigned to closest center and center the average of all points assigned to it). 

Additionally, there is a basic unit test for the distance functions. There is also a test that runs yy and naive together, and makes sure that the resulting msses are similar (theoretically, they would be exactly the same, but we allow for some differences due to floating point rounding).

Note: the testing code is somewhat repetitive in form, could be written more concisely (TODO). 