//This file tests naive and yy together, by running both and comparing the msse

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
#include "yy.h"

#include "../utils/point_range.h"
#include "../utils/euclidian_point.h"

#include <gtest/gtest.h>
#include "kmeansTests.h"

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
    delete D;

  }

  static float* v;
  static size_t base_n;
  static size_t base_d;
  static size_t ad;
  static Distance* D;

  //run all sorts of tests in here
  void naiveTestDimension() {
    
  }

 
};
//initialize static vals
float* NaiveData1::v = nullptr;
size_t NaiveData1::base_n = 0;
size_t NaiveData1::base_d = 0;
size_t NaiveData1::ad = 0;
Distance* NaiveData1::D = nullptr;


TEST_F(NaiveData1,TestDim) {
  EXPECT_EQ(base_n,1000000);
  EXPECT_EQ(base_d,200);

}
TEST_F(NaiveData1,Test1) {
   kmeansComparativeTest<float,float,size_t,NaiveKmeans<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>,Yinyang<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>>(v,1000,128,base_d,10,*D,10,0);


}
TEST_F(NaiveData1,Test2) {

    kmeansComparativeTest<float,float,size_t,NaiveKmeans<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>,Yinyang<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>>(v,5000,100,base_d,3,*D,10,0);


}
TEST_F(NaiveData1,Test3) {
   kmeansComparativeTest<float,float,size_t,NaiveKmeans<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>,Yinyang<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>>(v,2001,50,base_d,13,*D,10,0);
  
}
TEST_F(NaiveData1,Test4) {

    kmeansComparativeTest<float,float,size_t,NaiveKmeans<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>,Yinyang<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>>(v,2001,50,base_d,1,*D,10,0);

  
}
TEST_F(NaiveData1,Test5) {
      kmeansComparativeTest<float,float,size_t,NaiveKmeans<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>,Yinyang<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>>(v,2000,1,base_d,13,*D,10,0);
  
}



TEST_F(NaiveData1,TestLong) {
   kmeansComparativeTest<float,float,size_t,NaiveKmeans<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>,Yinyang<float,Euclidian_Point<float>,size_t,float,Euclidian_Point<float>>>(v,100000,2,base_d,10,*D,10,0);
}


//Test fixture for Yinyang Testing
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

    delete D;

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
  }

};
//initialize static vals
uint8_t* NaiveData2::v = nullptr;
size_t NaiveData2::base_n = 0;
size_t NaiveData2::base_d = 0;
size_t NaiveData2::ad = 0;
Distance* NaiveData2::D = nullptr;

TEST_F(NaiveData2,TestDimension) {
  naiveTestDimension();
}

TEST_F(NaiveData2,Test1) {
  kmeansComparativeTest<uint8_t,float,size_t,NaiveKmeans<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>,Yinyang<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>>(v,10000,2,base_d,20,*D,10,0);

}
TEST_F(NaiveData2,Test2) {
  kmeansComparativeTest<uint8_t,float,size_t,NaiveKmeans<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>,Yinyang<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>>(v,500,10,base_d,5,*D,10,0);
  
}
TEST_F(NaiveData2,Test3) {
  kmeansComparativeTest<uint8_t,float,size_t,NaiveKmeans<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>,Yinyang<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>>(v,1217,60,base_d,40,*D,10,0);
  
}
TEST_F(NaiveData2,Test4) {
  kmeansComparativeTest<uint8_t,float,size_t,NaiveKmeans<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>,Yinyang<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,128,base_d,40,*D,10,0);
}
TEST_F(NaiveData2,Testkis1) {
  kmeansComparativeTest<uint8_t,float,size_t,NaiveKmeans<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>,Yinyang<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,base_d,base_d,1,*D,10,0);
}
TEST_F(NaiveData2,Testdis1) {
  kmeansComparativeTest<uint8_t,float,size_t,NaiveKmeans<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>,Yinyang<uint8_t,Euclidian_Point<uint8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,1,base_d,100,*D,10,0);
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
    delete D;

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
  kmeansComparativeTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>,Yinyang<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,10000,5,base_d,20,*D,10,0);

}
TEST_F(NaiveData3,Test2) {
   kmeansComparativeTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>,Yinyang<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,2000,base_d,base_d,500,*D,10,0);
  
}
TEST_F(NaiveData3,Test3) {
   kmeansComparativeTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>,Yinyang<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,900,60,base_d,21,*D,10,0);
  
}
TEST_F(NaiveData3,Test4) {
   kmeansComparativeTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>,Yinyang<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,base_d,base_d,10,*D,10,0);
  
}

TEST_F(NaiveData3,Test5) {
   kmeansComparativeTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>,Yinyang<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,99,base_d,10,*D,10,0);
  
}
TEST_F(NaiveData3,Testkis1) {
   kmeansComparativeTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>,Yinyang<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,base_d,base_d,1,*D,10,0);
  
}

TEST_F(NaiveData3,Testdis1) {
   kmeansComparativeTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>,Yinyang<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,1000,1,base_d,10,*D,10,0);
  
}


//higher dimensions, so this test takes a bit longer (6s)
TEST_F(NaiveData3,TestHigh_n) {
   kmeansComparativeTest<int8_t,float,size_t,NaiveKmeans<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>,Yinyang<int8_t,Euclidian_Point<int8_t>,size_t,float,Euclidian_Point<float>>>(v,100000,base_d,base_d,1000,*D,10,0);
  
}

