//test the Distance function

#include <gtest/gtest.h>
#include "../distance.h"

//make sure that any d value works
//calculating distance on a vector filled with the same value for ease of test
TEST(EuclideanDistance,OneDistance) {
  std::vector<Distance*> distance_list = {new EuclideanDistanceSmall(), new EuclideanDistanceFast()};
  for (Distance* D : distance_list) {
    int N = 200;
    parlay::sequence<float> ones(N,1);
    parlay::sequence<float> twos(N,2);
    //distance between ones and twos should be the number of elements we are checking, which is i
    for (int i = 0; i <= N; i++) {
      EXPECT_EQ(D->distance(ones.begin(),twos.begin(),i),i);
    }

  }


}

TEST(EuclideanDistance,Size10PrintoutsFloat) {
  std::vector<Distance*> distance_list = {new EuclideanDistanceSmall(), new EuclideanDistanceFast()};
  for (Distance* D : distance_list) {
    parlay::sequence<float> ones(10,1);
    parlay::sequence<float> other_ones(10,1);
    parlay::sequence<float> twos(10,2);
    parlay::sequence<float> rang = parlay::tabulate(10,[&] (size_t i) {
        return static_cast<float>(i);
      });
    parlay::sequence<float> double_rang = parlay::tabulate(10,[&] (size_t i) {
        return static_cast<float>(2*i);
      });
    EXPECT_EQ(D->distance(parlay::make_slice(ones).begin(),
    parlay::make_slice(other_ones).begin(),10),0);

    EXPECT_EQ(1,1);

    EXPECT_EQ(D->distance(parlay::make_slice(ones).begin(),
    parlay::make_slice(twos).begin(),10),10);

    EXPECT_EQ(D->distance(parlay::make_slice(ones).begin(),
    parlay::make_slice(double_rang).begin(),10),970);

    //confirm float handling
    parlay::sequence<float> halves(10,.5);
    EXPECT_LE(
      std::abs(D->distance(parlay::make_slice(halves).begin(),
    parlay::make_slice(ones).begin(),10)-2.5),.001);

    parlay::sequence<float> shorty1 = parlay::sequence<float>(1,17);
    float* shorty2 = new float[1];
    shorty2[0] = 35;
    EXPECT_EQ(D->distance(parlay::make_slice(shorty1).begin(),
    shorty2,1),324);

    //ZERO DIST
    EXPECT_EQ(D->distance(parlay::make_slice(shorty1).begin(),
    shorty2,0),0);

    delete[] shorty2;
  }

   
}

TEST(EuclideanDistance,Size10PrintoutsInt) {
  std::vector<Distance*> distance_list = {new EuclideanDistanceSmall(), new EuclideanDistanceFast()};
  for (Distance* D : distance_list) {
    parlay::sequence<uint8_t> threes(10,3);
    parlay::sequence<uint8_t> other_ones(10,1);
    parlay::sequence<uint8_t> sevens(10,7);
    parlay::sequence<uint8_t> rang = parlay::tabulate(10,[&] (size_t i) {
        return static_cast<uint8_t>(i);
      });
    parlay::sequence<uint8_t> double_rang = parlay::tabulate(10,[&] (size_t i) {
        return static_cast<uint8_t>(2*i);
      });
    EXPECT_EQ(D->distance(parlay::make_slice(threes).begin(),
    parlay::make_slice(other_ones).begin(),10),40);

    EXPECT_EQ(1,1);

    EXPECT_EQ(D->distance(parlay::make_slice(threes).begin(),
    parlay::make_slice(sevens).begin(),10),160);

    EXPECT_EQ(D->distance(parlay::make_slice(threes).begin(),
    parlay::make_slice(double_rang).begin(),10),690);
  }

}

void nested_Dcall(Distance& D, std::string id, int num) {
  if (num == 0) {
    EXPECT_EQ(id,D.id());

  }
  else {
    nested_Dcall(D,id,num-1);
  }
}
void nested_Dcall2(Distance& D, std::string id, int num) {
  if (num==0) {
    EXPECT_EQ(id,D.id());
  }
  else {
    nested_Dcall2(D,id,num-1);
    nested_Dcall2(D,id,num-1);

  }
}
//make sure the Distance object can be passed by reference
TEST(EuclideanDistance,NestedCalls) {
  std::vector<Distance*> distance_list = {new EuclideanDistanceSmall(), new EuclideanDistanceFast()};
  for (Distance* D : distance_list) {
    nested_Dcall(*D,D->id(),200);
    nested_Dcall(*D,D->id(),50);
    nested_Dcall2(*D,D->id(),10);

  }

  for (Distance* D : distance_list) {
    delete D;


  }

    
  
}
