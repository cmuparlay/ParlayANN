//naive testing


//TODO purge the include list to include only what's actually needed
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"
#include "parlay/random.h"
#include "parlay/internal/get_time.h"

#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>
#include <set>
#include <atomic>
#include <utility>
#include <type_traits>
#include <cmath>

#include "../bench/parse_command_line.h"
#include "parse_files.h"
#include "distance.h"
#include "kmeans_bench.h"
#include "initialization.h"
#include "naive.h"
#include "kmeans.h"

#include "../utils/point_range.h"
#include "../utils/euclidian_point.h"

#include <gtest/gtest.h>

//confirm that k=1 run is average of all points
//run to completion, check closest point and center average

//given a convergent kmeans run, confirm that the centers are the average of the points assigned to them
//T - point type
//CT - center coordinate type
//index_type - type of asg[0]
//TODO make this check for efficient (ie ||ize)
template<typename T, typename CT, typename index_type>
void assertCenteredCentroids(T* v, size_t n, size_t d, size_t ad, size_t k, CT* c, index_type* asg) {

  double* centroids = new double[ad*k];
  parlay::sequence<size_t> num_members = parlay::sequence<size_t>(k,0);
  for (size_t i = 0; i < k*ad; i++) {
    centroids[i]=0;
  }

  parlay::parallel_for(0,d,[&] (size_t j) {
    for (size_t i = 0; i < n; i++) {
      size_t center = asg[i];
      
      centroids[center*ad+j] += v[i*ad+j];
    }

  });

  for (size_t i = 0; i < n; i++) {
    num_members[asg[i]]++;

  }

  for (size_t i = 0; i < k; i++) {
    std::cout << "num_mem[" << i << "]: " << num_members[i] << std::endl; 
  }
  std::cout << "total mems " << parlay::reduce(num_members) << std::endl;

  EXPECT_EQ(parlay::reduce(num_members),n);


  //confirm that assignments are to a possible center
  for (size_t i = 0; i < n; i++) {
    EXPECT_LE(asg[i],k-1);
    EXPECT_GE(asg[i],0);
  }

  //confirm that num_members elts are between 0 andn n
  for (size_t i = 0; i < k; i++) {
    EXPECT_GE(num_members[i],0); //change to 0
    EXPECT_LE(num_members[i],n);
  }

  std::cout << "HERE4" << std::endl;

  for (size_t i = 0; i < 10; i++) {
    std::cout << centroids[i] << " " << c[i] << std::endl;
  }

  //TODO how tightly should we set this?
  //putting centroids in double to get a bit better precision
  for(size_t i=0; i < k*d; i++){
    EXPECT_LE(std::abs(centroids[i]/num_members[i/d]-c[i]),.1) //+10 is wrong
    << "This one failed\n";
  }
  
  delete[] centroids;


}



//Makes sure that each point is assigned to its closest center
template<typename T, typename CT, typename index_type>
void assertClosestPoints(T* v, size_t n, size_t d, size_t ad, size_t k, CT* c, index_type* asg, Distance& D) {
   //rang is range from 0 to k incl excl
    parlay::sequence<size_t> rang = parlay::tabulate(k,[&] (size_t i) {
      return i;
    });

    index_type* bests = new index_type[n];

    parlay::parallel_for(0,n,[&] (size_t i) {
      auto distances = parlay::delayed::map(rang, [&](size_t j) {
            return D.distance(v+i*d, c+j*d,d); });

     bests[i] = min_element(distances) - distances.begin();
    });

    for (size_t i = 0; i < n; i++) {
      EXPECT_EQ(bests[i],asg[i]) << "asg fail " << bests[i] << " " << asg[i];
    }

    delete[] bests; //memory cleanup
}

//Test fixture for NaiveKmeans Testing
//Open the datafile (so that we don't open it each test)
class NaiveData1 : public ::testing::Test {
  protected:

  static void SetUpTestSuite()  {
    auto [v2,n_int,d_int] = parse_fbin("/ssd1/anndata/bigann/text2image1B/base.1B.fbin.crop_nb_1000000");
    v=v2;
    base_n = (size_t) n_int;
    base_d = (size_t) d_int;
    ad = base_d;
    
    D = new EuclideanDistanceFast();
  }

  static void TearDownTestSuite()  {

    //TODO warning: deleting object of polymorphic class type 'Distance' which has non-virtual destructor might cause undefined behavior [-Wdelete-non-virtual-dtor]
    // delete D;

  }

  static float* v;
  static size_t base_n;
  static size_t base_d;
  static size_t ad;
  static Distance* D;

  //run all sorts of tests in here
  void naiveTestDimension() {
    EXPECT_EQ(base_n,1000000);
    EXPECT_EQ(base_d,200);
    EXPECT_EQ(3+5,8);
    std::cout << "n : " << base_n << std::endl;
  }

  void naiveTestOnce() {
    
    size_t n = 1000;
    size_t k = 10;
    float* c = new float[k*ad]; // centers
    size_t* asg = new size_t[n];
    size_t max_iter=1000;
    double epsilon=0;
    

    //initialization
    Lazy<float,size_t> init;
    //note that here, d=ad
    init(v,n,base_d,ad,k,c,asg);

   
    NaiveKmeans<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>> nie2;
    kmeans_bench logger_nie2 = kmeans_bench(n,base_d,k,max_iter,
    epsilon,"Lazy","Naive");
    logger_nie2.start_time();
    //note that d=ad here
    nie2.cluster_middle(v,n,base_d,ad,k,c,asg,*D,logger_nie2,max_iter,epsilon);
    logger_nie2.end_time();
     SCOPED_TRACE("float data, CENTROIDS"); //about to test centers
    assertCenteredCentroids<float,float,size_t>(v,n,base_d,ad,k,c,asg);
     SCOPED_TRACE("float data, CLOSEST_POINT"); //trace about to test closest points
    assertClosestPoints<float,float,size_t>(v,n,base_d,ad,k,c,asg,*D);

  }
 

};
//initialize static vals
float* NaiveData1::v = nullptr;
size_t NaiveData1::base_n = 0;
size_t NaiveData1::base_d = 0;
size_t NaiveData1::ad = 0;
Distance* NaiveData1::D = nullptr;



TEST_F(NaiveData1,Test1) {
  naiveTestDimension();
  naiveTestOnce();

}
