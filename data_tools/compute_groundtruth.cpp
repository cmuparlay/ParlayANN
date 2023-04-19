#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include "utils/types.h"
#include "utils/NSGDist.h"
#include "utils/parse_files.h"

using pid = std::pair<int, float>;

template<typename T>
float mips_distance(T* p, T* q, unsigned d){
  // std::cout << "here1" << std::endl;
  float result = 0;
  for(int i=0; i<d; i++){
    float a = *(p+i);
    float b = *(q+i);
    result -= a*b;
  }
  // std::cout << "here" << std::endl; 
  return result;
}

template<typename T>
parlay::sequence<parlay::sequence<pid>> compute_groundtruth(parlay::sequence<Tvec_point<T>> &B, 
  parlay::sequence<Tvec_point<T>> &Q, int k, bool mips=false){
    unsigned d = (B[0].coordinates).size();
    size_t q = Q.size();
    size_t b = B.size();
    auto answers = parlay::tabulate(q, [&] (size_t i){  
        float topdist = 0;
        if(mips) topdist = -std::numeric_limits<float>::max();     
        int toppos;
        parlay::sequence<pid> topk;
        for(size_t j=0; j<b; j++){
            float dist;
            if(mips) dist = mips_distance<T>((Q[i].coordinates).begin(), (B[j].coordinates).begin(), d);
            else dist = distance((Q[i].coordinates).begin(), (B[j].coordinates).begin(), d);
            if(topk.size() < k){
                if(dist > topdist){
                    topdist = dist;   
                    toppos = topk.size();
                }
                topk.push_back(std::make_pair((int) j, dist));
            }
            else if(dist < topdist){
                float new_topdist=0;
                if(mips) new_topdist = -std::numeric_limits<float>::max();   
                int new_toppos=0;
                topk[toppos] = std::make_pair((int) j, dist);
                for(size_t l=0; l<topk.size(); l++){
                    if(topk[l].second > new_topdist){
                        new_topdist = topk[l].second;
                        new_toppos = (int) l;
                    }
                }
                topdist = new_topdist;
                toppos = new_toppos;
            }
        }
        return topk;
    });
    std::cout << "Done computing groundtruth" << std::endl;
    return answers;
}

void write_ivecs(parlay::sequence<parlay::sequence<pid>> &result, const std::string outFile, int k){
    std::cout << "Writing file with dimension " << result[0].size() << std::endl;
    std::cout << "File contains groundtruth for " << result.size() << " data points" << std::endl;

    auto less = [&] (pid a, pid b) {return a.second < b.second;};

    size_t n = result.size();
    auto vects = parlay::tabulate(result.size(), [&] (size_t i){
        parlay::sequence<int> data;
        data.push_back(k);

        auto sorted = parlay::sort(result[i], less);
        for(int j=0; j<k; j++){
          data.push_back(sorted[j].first);
        }
        return data;
    });

    parlay::sequence<int> to_write = parlay::flatten(vects);

    auto data = to_write.begin();
    std::ofstream writer;
    writer.open(outFile, std::ios::binary | std::ios::out);
    writer.write((char *) data, n * (k+1) * sizeof(int));
    writer.close();
}

void write_ibin(parlay::sequence<parlay::sequence<pid>> &result, const std::string outFile, int k){
    std::cout << "Writing file with dimension " << result[0].size() << std::endl;
    std::cout << "File contains groundtruth for " << result.size() << " data points" << std::endl;

    auto less = [&] (pid a, pid b) {return a.second < b.second;};
    parlay::sequence<int> preamble = {static_cast<int>(result.size()), static_cast<int>(result[0].size())};
    size_t n = result.size();
    parlay::parallel_for(0, result.size(), [&] (size_t i){
      parlay::sort_inplace(result[i], less);
    });
    auto ids = parlay::tabulate(result.size(), [&] (size_t i){
        parlay::sequence<int> data;
        for(int j=0; j<k; j++){
          data.push_back(static_cast<int>(result[i][j].first));
        }
        return data;
    });
    auto distances = parlay::tabulate(result.size(), [&] (size_t i){
        parlay::sequence<float> data;
        for(int j=0; j<k; j++){
          data.push_back(static_cast<float>(result[i][j].second));
        }
        return data;
    });
    parlay::sequence<int> flat_ids = parlay::flatten(ids);
    parlay::sequence<float> flat_dists = parlay::flatten(distances);

    auto pr = preamble.begin();
    auto id_data = flat_ids.begin();
    auto dist_data = flat_dists.begin();
    std::ofstream writer;
    writer.open(outFile, std::ios::binary | std::ios::out);
    writer.write((char *) pr, 2*sizeof(int));
    writer.write((char *) id_data, n * k * sizeof(int));
    writer.write((char *) dist_data, n * k * sizeof(float));
    writer.close();
}


int main(int argc, char* argv[]) {
  if (argc != 8) {
    std::cout << "usage: compute_groundtruth <base> <query> <filetype> <vectype> <k> <mips> <oFile>" << std::endl;
    return 1;
  }
  int k = std::atoi(argv[5]);
  int mips_choice = std::atoi(argv[6]);
  bool mips=false;
  if(mips_choice==1) mips=true;
  std::string ft = std::string(argv[3]);
  std::string tp = std::string(argv[4]);

  std::cout << "Computing the " << k << " nearest neighbors" << std::endl;
  if((ft != "bin") && (ft != "vec")){
    std::cout << "Error: file type not specified correctly, specify bin or vec" << std::endl;
    abort();
  }

  if((tp != "uint8") && (tp != "int8") && (tp != "float")){
    std::cout << "Error: vector type not specified correctly, specify int8, uint8, or float" << std::endl;
    abort();
  }

  if((ft == "vec") && (tp == "int8")){
    std::cout << "Error: incompatible file and vector types" << std::endl;
    abort();
  }

  int maxDeg = 0;

  parlay::sequence<parlay::sequence<pid>> answers;

  if(ft == "vec"){
    if(tp == "float"){
      std::cout << "Detected float coordinates" << std::endl;
      auto [md, B] = parse_fvecs(argv[1], NULL, maxDeg);
      auto [fd, Q] = parse_fvecs(argv[2], NULL, maxDeg);
      std::cout << "Base file size " << B.size() << std::endl;
      std::cout << "Query file size " << Q.size() << std::endl;
      answers = compute_groundtruth<float>(B, Q, k, mips);
    }else if(tp == "uint8"){
      std::cout << "Detected uint8 coordinates" << std::endl;
      auto [md, B] = parse_bvecs(argv[1], NULL, maxDeg);
      auto [fd, Q] = parse_bvecs(argv[2], NULL, maxDeg);
      std::cout << "Base file size " << B.size() << std::endl;
      std::cout << "Query file size " << Q.size() << std::endl;
      answers = compute_groundtruth<uint8_t>(B, Q, k, mips);
    }
    write_ivecs(answers, std::string(argv[7]), k);
  } else if(ft == "bin"){
    if(tp == "float"){
      std::cout << "Detected float coordinates" << std::endl;
      auto [md, B] = parse_fbin(argv[1], NULL, maxDeg);
      auto [fd, Q] = parse_fbin(argv[2], NULL, maxDeg);
      std::cout << "Base file size " << B.size() << std::endl;
      std::cout << "Query file size " << Q.size() << std::endl;
      answers = compute_groundtruth<float>(B, Q, k, mips);
    }else if(tp == "uint8"){
      std::cout << "Detected uint8 coordinates" << std::endl;
      auto [md, B] = parse_uint8bin(argv[1], NULL, maxDeg);
      auto [fd, Q] = parse_uint8bin(argv[2], NULL, maxDeg);
      std::cout << "Base file size " << B.size() << std::endl;
      std::cout << "Query file size " << Q.size() << std::endl;
      answers = compute_groundtruth<uint8_t>(B, Q, k, mips);
    }else if(tp == "int8"){
      std::cout << "Detected int8 coordinates" << std::endl;
      auto [md, B] = parse_int8bin(argv[1], NULL, maxDeg);
      auto [fd, Q] = parse_int8bin(argv[2], NULL, maxDeg);
      std::cout << "Base file size " << B.size() << std::endl;
      std::cout << "Query file size " << Q.size() << std::endl;
      answers = compute_groundtruth<int8_t>(B, Q, k, mips);
    }
    write_ibin(answers, std::string(argv[7]), k);
  }

  

  return 0;
}