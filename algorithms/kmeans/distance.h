//expand on NSGDist.h

#ifndef DISTANCE_CLASS
#define DISTANCE_CLASS

#include <math.h>
#include <x86intrin.h>

#include <algorithm>
#include <iostream>
#include <type_traits>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "../utils/NSGDist.h"

class Distance{
  public:
    virtual std::string id(){return "generic";}
    virtual float distance(uint8_t *p, uint8_t *q, unsigned d){return 0;}
    virtual float distance(int8_t *p, int8_t *q, unsigned d){return 0;}
    virtual float distance(float *p, float *q, unsigned d){return 0;}
};


struct Mips_Distance : public Distance{

  std::string id(){return "mips";}

  float distance(uint8_t *p, uint8_t *q, unsigned d){
    int result = 0;
    for(unsigned i=0; i<d; i++){ //changing type of i to unsigned to avoid compiler warnings
      result += ((int32_t) q[i]) *
                    ((int32_t) p[i]);
    }
    return -((float) result);
  }

  float distance(int8_t *p, int8_t *q, unsigned d){
    int result = 0;
    for(unsigned i=0; i<d; i++){ //changed type of i to unsigned to avoid compiler warnings
      result += ((int32_t) q[i]) *
                    ((int32_t) p[i]);
    }
    return -((float) result);
  }

  float distance(float *p, float *q, unsigned d){
      float result = 0;
      for(unsigned i=0; i<d; i++){
        result += (q[i]) * (p[i]);
      }
      return -result;
  }

};

//EuclideanDistanceFast prohibits mixed type distance calls (much easier to debug), but used the fast distance implementation
struct EuclideanDistanceFast : public Distance {
    std::string id() {return "euclidean_fast";}

    float distance(uint8_t *p, uint8_t *q, unsigned d){
          // std::cout << "uint8: d: " << static_cast<int>(d) << std::endl;

    int result = 0;
    for(unsigned i=0; i<d; i++){
      result += ((int32_t)((int16_t) q[i] - (int16_t) p[i])) *
                    ((int32_t)((int16_t) q[i] - (int16_t) p[i]));
    }
    return (float) result;
  }

  float distance(int8_t *p, int8_t *q, unsigned d){
    int result = 0;
    for(unsigned i=0; i<d; i++){
      result += ((int32_t)((int16_t) q[i] - (int16_t) p[i])) *
                    ((int32_t)((int16_t) q[i] - (int16_t) p[i]));
    }
    return (float) result;
  }
  float distance(float* p, float* q, unsigned d) {
     efanna2e::DistanceL2 distfunc;
    return distfunc.compare(p, q, d);
  }

};

//Euclidian distance to use if d < 36 (I believe that 30 is the bound, but 
//just to be safe)
struct EuclideanDistanceSmall : public Distance {
    std::string id() {return "euclidean_small";}

    float distance(uint8_t *p, uint8_t *q, unsigned d){
          // std::cout << "uint8: d: " << static_cast<int>(d) << std::endl;

    int result = 0;
    for(unsigned i=0; i<d; i++){
      result += ((int32_t)((int16_t) q[i] - (int16_t) p[i])) *
                    ((int32_t)((int16_t) q[i] - (int16_t) p[i]));
    }
    return (float) result;
  }

  float distance(int8_t *p, int8_t *q, unsigned d){
    int result = 0;
    for(unsigned i=0; i<d; i++){
      result += ((int32_t)((int16_t) q[i] - (int16_t) p[i])) *
                    ((int32_t)((int16_t) q[i] - (int16_t) p[i]));
    }
    return (float) result;
  }
  float distance(float* p, float* q, unsigned d) {
    float result = 0;
    for (unsigned i = 0; i < d; i++) {
      result += (p[i]-q[i])*(p[i]-q[i]);
    }
    return result;
  }



};



#endif//DISTANCE_CLASS