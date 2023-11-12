//naive testing
//note that the validity of this testing relies on the centers starting out at points in the data -- as this effectively guarantees that the converged solution will have all nonempty centers, because they will at least own their initial point.
//so our tests will fail if a center permanently loses all its points, when this may in fact be legitimate behavior. (TODO adjust this)

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
//confirm that k=n run has one point per center and msse 0
//run to completion, check closest point and center average
//make sure that at least half of the centers own points 

//given a convergent kmeans run, confirm that the centers are the average of the points assigned to them
//T - point type
//CT - center coordinate type
//index_type - type of asg[0]
//TODO make this check for efficient (ie ||ize)
//TODO what happens in this test if a center has no points (oh this test requires for every center to own points, okay that's fine)
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

  // for (size_t i = 0; i < k; i++) {
  //   std::cout << "num_mem[" << i << "]: " << num_members[i] << std::endl; 
  // }
  // std::cout << "total mems " << parlay::reduce(num_members) << std::endl;

  EXPECT_EQ(parlay::reduce(num_members),n);


  //confirm that assignments are to a possible center
  for (size_t i = 0; i < n; i++) {
    EXPECT_LE(asg[i],k-1) << "Asg >= k\n";
    EXPECT_GE(asg[i],0) << "Asg < 0\n";
  }

  //confirm that num_members elts are between 0 andn n
  for (size_t i = 0; i < k; i++) {
    EXPECT_GE(num_members[i],0) << "Center overassigned\n"; //change to 0
    EXPECT_LE(num_members[i],n) << "Center underassigned\n";
  }

  // std::cout << "HERE4" << std::endl;

  // for (size_t i = 0; i < 10; i++) {
  //   std::cout << centroids[i] << " " << c[i] << std::endl;
  // }

  //TODO how tightly should we set this?
  //putting centroids in double to get a bit better precision
  for (size_t i = 0; i < k; i++) {
    for (size_t j = 0; j < d; j++) {
      double diff = std::abs(centroids[i*ad+j]-c[i*ad+j]*num_members[i]);
      if (diff < .1) {
        continue;
      }
      else {
        EXPECT_LE(diff,.1) << "Center " << i << "differs from average";
        break; //prevent overprinting
      }
     
      
    }
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
        CT buf[2048];
        T* it = v+i*ad;
        for (size_t el = 0; el < d; el++) buf[el]=*(it++);
        return D.distance(buf, c+j*ad,d); 
      });

     bests[i] = min_element(distances) - distances.begin();
    });

    for (size_t i = 0; i < n; i++) {
      EXPECT_EQ(bests[i],asg[i]) << "asg fail " << bests[i] << " " << asg[i];
    }

    delete[] bests; //memory cleanup
}


//given a kmeans method, n, d, k, run to convergence, and make sure that the result is a valid kmeans result
//by setting k=1, we get the k is 1 test implicitly (don't need a separate function) (by k is 1 test, I mean that the result of clustering with k=1 is that the single center is the average of all the points)
template<typename T, typename CT, typename index_type, typename Kmeans>
double kmeansConvergenceTest(T* v, size_t n, size_t d, size_t ad, size_t k, Distance& D, bool suppress_logging=true) {
    CT* c = new CT[k*ad]; // centers
    index_type* asg = new index_type[n];
    size_t max_iter=1000;
    double epsilon=0;
    

    //initialization
    Lazy<T,CT, index_type> init;
    //note that here, d=ad
    init(v,n,d,ad,k,c,asg);

   
    Kmeans runner;
    kmeans_bench logger = kmeans_bench(n,d,k,max_iter,
    epsilon,"Lazy","Naive");
    logger.start_time();
    //true at the end suppresses logging
    runner.cluster_middle(v,n,d,ad,k,c,asg,D,logger,max_iter,epsilon,suppress_logging);
    logger.end_time();
     SCOPED_TRACE("dims: " + std::to_string(n) + " " + std::to_string(d) + " " + std::to_string(k)); //about to test centers
    assertCenteredCentroids<T,CT,index_type>(v,n,d,ad,k,c,asg);
    assertClosestPoints<T,CT,index_type>(v,n,d,ad,k,c,asg,D);
    auto rangn = parlay::iota(n);
    float msse = parlay::reduce(parlay::map(rangn,[&] (size_t i) { 
      float buf[2048];
      T* it = v+i*ad;
      for (size_t i = 0; i < d; i++) buf[i]=*(it++);
      return D.distance(buf,c+asg[i]*ad,d);
    }))/n; //calculate msse

    delete[] c;
    delete[] asg;

    return msse;

    

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
    //where does this error come from? 
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
    size_t k = 8;
    float* c = new float[k*ad]; // centers
    size_t* asg = new size_t[n];
    size_t max_iter=1000;
    double epsilon=0;
    

    //initialization
    Lazy<float,float,size_t> init;
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

    delete[] c;
    delete[] asg;

  }
 
  

  //run to converge and check values for several instances of n,d,k
  void naiveTestSeveral() {
    kmeansConvergenceTest<float,float,size_t,NaiveKmeans<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>>(v,1000,128,base_d,10,*D);

    kmeansConvergenceTest<float,float,size_t,NaiveKmeans<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>>(v,5000,100,base_d,3,*D);

    kmeansConvergenceTest<float,float,size_t,NaiveKmeans<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>>(v,2001,50,base_d,13,*D);

    kmeansConvergenceTest<float,float,size_t,NaiveKmeans<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>>(v,2001,50,base_d,1,*D);

    kmeansConvergenceTest<float,float,size_t,NaiveKmeans<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>>(v,2000,1,base_d,13,*D);
    
    
  }

  void naiveTestLong() {
    kmeansConvergenceTest<float,float,size_t,NaiveKmeans<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>>(v,100000,2,base_d,10,*D);

  }
 
};
//initialize static vals
float* NaiveData1::v = nullptr;
size_t NaiveData1::base_n = 0;
size_t NaiveData1::base_d = 0;
size_t NaiveData1::ad = 0;
Distance* NaiveData1::D = nullptr;



//Test fixture for NaiveKmeans Testing
//Open the datafile (so that we don't open it each test)
//NaiveData2 looks at a different data file
class NaiveData2 : public ::testing::Test {
  protected:

  static void SetUpTestSuite()  {
    auto [v2,n_int,d_int] = parse_uint8bin("/ssd1/anndata/bigann/base.1B.u8bin.crop_nb_1000000");
    v=v2;
    base_n = (size_t) n_int;
    base_d = (size_t) d_int;
    ad = base_d;
    
    D = new EuclideanDistanceFast();
  }

  static void TearDownTestSuite()  {

    //TODO warning: deleting object of polymorphic class type 'Distance' which has non-virtual destructor might cause undefined behavior [-Wdelete-non-virtual-dtor]
    //where does this error come from? 
    // delete D;

  }

  static uint8_t* v;
  static size_t base_n;
  static size_t base_d;
  static size_t ad;
  static Distance* D;

  //run all sorts of tests in here
  void naiveTestDimension() {
    EXPECT_EQ(base_n,1000000);
    EXPECT_EQ(base_d,128);
    EXPECT_EQ(3+5,8);
  }

};
//initialize static vals
uint8_t* NaiveData2::v = nullptr;
size_t NaiveData2::base_n = 0;
size_t NaiveData2::base_d = 0;
size_t NaiveData2::ad = 0;
Distance* NaiveData2::D = nullptr;



TEST_F(NaiveData1,Test1) {
  naiveTestDimension();
 // naiveTestOnce();
  naiveTestSeveral();

}
TEST_F(NaiveData1,TestNisK) {
  double msse = kmeansConvergenceTest<float,float,size_t,NaiveKmeans<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>>(v,1000,base_d,base_d,1000,*D);
  EXPECT_EQ(msse,0);
}

TEST_F(NaiveData1,Test2) {
  naiveTestLong();
}

TEST_F(NaiveData2,TestDimension) {
  naiveTestDimension();
  
  
  
}

TEST_F(NaiveData2,Test1) {
  kmeansConvergenceTest<uint8_t,float,size_t,NaiveKmeans<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>>(v,10000,2,base_d,20,*D);

}
TEST_F(NaiveData2,Test2) {
  kmeansConvergenceTest<uint8_t,float,size_t,NaiveKmeans<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>>(v,500,10,base_d,5,*D);
  
}
TEST_F(NaiveData2,Test3) {
  kmeansConvergenceTest<uint8_t,float,size_t,NaiveKmeans<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>>(v,1217,60,base_d,40,*D);
  
}
TEST_F(NaiveData2,Test4) {
  kmeansConvergenceTest<uint8_t,float,size_t,NaiveKmeans<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,128,base_d,40,*D);
}
TEST_F(NaiveData2,Testkis1) {
  kmeansConvergenceTest<uint8_t,float,size_t,NaiveKmeans<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,base_d,base_d,1,*D);
}
TEST_F(NaiveData2,Testdis1) {
  kmeansConvergenceTest<uint8_t,float,size_t,NaiveKmeans<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,1,base_d,100,*D);
}
TEST_F(NaiveData2,Test_n_is_k) {
  double msse = kmeansConvergenceTest<uint8_t,float,size_t,NaiveKmeans<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,base_d,base_d,1000,*D);
  EXPECT_EQ(msse,0);
}




//NaiveData3 looks at another different data file
class NaiveData3 : public ::testing::Test {
  protected:

  static void SetUpTestSuite()  {
    auto [v2,n_int,d_int] = parse_int8bin("/ssd1/anndata/MSSPACEV1B/spacev1b_base.i8bin.crop_nb_1000000");
    
    v=v2;
    base_n = (size_t) n_int;
    base_d = (size_t) d_int;
    ad = base_d;
    
    D = new EuclideanDistanceFast();
  }

  static void TearDownTestSuite()  {

    //TODO warning: deleting object of polymorphic class type 'Distance' which has non-virtual destructor might cause undefined behavior [-Wdelete-non-virtual-dtor]
    //where does this error come from? 
    // delete D;

  }

  static int8_t* v;
  static size_t base_n;
  static size_t base_d;
  static size_t ad;
  static Distance* D;

  //run all sorts of tests in here
  void naiveTestDimension() {
    EXPECT_EQ(base_n,1000000);
    EXPECT_EQ(base_d,100);
    EXPECT_EQ(3+5,8);
  }

};
//initialize static vals
int8_t* NaiveData3::v = nullptr;
size_t NaiveData3::base_n = 0;
size_t NaiveData3::base_d = 0;
size_t NaiveData3::ad = 0;
Distance* NaiveData3::D = nullptr;


TEST_F(NaiveData3,TestDimension) {
  naiveTestDimension();
  
  
  
}

TEST_F(NaiveData3,Test1) {
  kmeansConvergenceTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,10000,5,base_d,20,*D);

}
TEST_F(NaiveData3,Test2) {
  kmeansConvergenceTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,2000,base_d,base_d,500,*D);
  
}
TEST_F(NaiveData3,Test3) {
  kmeansConvergenceTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,900,60,base_d,21,*D);
  
}
TEST_F(NaiveData3,Test4) {
  kmeansConvergenceTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,base_d,base_d,10,*D);
  
}

TEST_F(NaiveData3,Test5) {
  kmeansConvergenceTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,99,base_d,10,*D);
  
}
TEST_F(NaiveData3,Testkis1) {
  kmeansConvergenceTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,base_d,base_d,1,*D);
  
}

TEST_F(NaiveData3,Testdis1) {
  kmeansConvergenceTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,1,base_d,10,*D);
  
}

TEST_F(NaiveData3,Test_n_is_k) {
  double msse = kmeansConvergenceTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,base_d,base_d,1000,*D);
  EXPECT_EQ(msse,0);
}
//higher dimensions, so this test takes a bit longer (6s)
TEST_F(NaiveData3,TestHigh_n) {
  kmeansConvergenceTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,100000,base_d,base_d,1000,*D);
  
}

