// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <atomic>
#include <vector>
#include <limits>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/alloc.h"
#include "parlay/internal/file_map.h"
#include "../bench/parse_command_line.h"

#include "../bench/parse_command_line.h"
#include "types.h"
#include "epoch.h"
#include "point_range.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#pragma once

template<typename T, class Point, typename indexType>
struct DynamicPointRange{


  struct PointRange{

    long dimension(){return dims;}
    long aligned_dimension(){return aligned_dims;}

    using pT = Point;

    PointRange() : values(std::shared_ptr<T[]>(nullptr, std::free)) {n=0;}

    PointRange(size_t max_size, unsigned int dims) : max_size(max_size), dims(dims){
      n=0;
      aligned_dims = dim_round_up(dims, sizeof(T));
      if(aligned_dims != dims) std::cout << "Aligning dimension to " << aligned_dims << std::endl;
      std::cout << "Allocating memory for " << max_size << " vectors" << std::endl;
      values = std::shared_ptr<T[]>((T*) aligned_alloc(64, max_size*aligned_dims*sizeof(T)), std::free);
      slot_to_id = parlay::sequence<indexType>(max_size);
      slots = parlay::sequence<std::atomic<bool>>(max_size);
    }

    PointRange(char* filename) : values(std::shared_ptr<T[]>(nullptr, std::free)){
        if(filename == NULL) {
          max_size = 0;
          dims = 0;
          return;
        }
        std::ifstream reader(filename);
        assert(reader.is_open());

        //read num points and max degree
        unsigned int num_points;
        unsigned int d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        n = num_points;
        max_size = 2*n;
        reader.read((char*)(&d), sizeof(unsigned int));
        dims = d;
        std::cout << "Detected " << n << " points with dimension " << dims << std::endl;
        aligned_dims =  dim_round_up(dims, sizeof(T));
        if(aligned_dims != dims) std::cout << "Aligning dimension to " << aligned_dims << std::endl;
        std::cout << "Allocating memory for " << max_size << " vectors" << std::endl;
        values = std::shared_ptr<T[]>((T*) aligned_alloc(64, max_size*aligned_dims*sizeof(T)), std::free);
        slot_to_id = parlay::sequence<indexType>(max_size, std::numeric_limits<indexType>::max());
        slots = parlay::sequence<std::atomic<bool>>(max_size);
        parlay::parallel_for(0, max_size, [&] (size_t i){slots[i].store(false);});
        size_t BLOCK_SIZE = 1000000;
        size_t index = 0;
        while(index < num_points){
            size_t floor = index;
            size_t ceiling = index+BLOCK_SIZE <= n ? index+BLOCK_SIZE : n;
            T* data_start = new T[(ceiling-floor)*dims];
            reader.read((char*)(data_start), sizeof(T)*(ceiling-floor)*dims);
            T* data_end = data_start + (ceiling-floor)*dims;
            parlay::slice<T*, T*> data = parlay::make_slice(data_start, data_end);
            int data_bytes = dims*sizeof(T);
            parlay::random_generator gen(floor);
            std::uniform_int_distribution<indexType> dis(0, max_size-1);
            parlay::sequence<std::pair<indexType, indexType>> id_slot_pairs(ceiling-floor);
            parlay::parallel_for(floor, ceiling, [&] (size_t i){
              auto r = gen[i];
              indexType start = dis(r);
              bool found = false;
              indexType slot;
              while(!found){
                bool expected = false;
                bool success = slots[start].compare_exchange_strong(expected, true);
                if(success){
                  found = true;
                  slot = start;
                  break;
                }
                if(start < max_size-1) start++;
                else start = 0;
              }
              if(slot >= max_size){
                std::cout << "ERROR: slot " << slot << " too large" << std::endl;
              }
              slot_to_id[slot] = i;
              id_slot_pairs[i-floor] = std::make_pair(i, slot);
              T* destination = values.get() + aligned_dims*slot;
              std::memmove(data.begin() + (i-floor)*dims, destination, sizeof(T)*dims);
            });
            for(auto p : id_slot_pairs){
              id_to_slot[p.first] = p.second;
            }
            delete[] data_start;
            index = ceiling;
        }
    }

    size_t size() { return n; }
    
    Point operator [] (long i) {
      return Point(values.get()+i*aligned_dims, dims, aligned_dims, i);
    }

    indexType real_id(indexType id) {return slot_to_id[id];}

    void insert(parlay::slice<T*, T*> data, parlay::sequence<indexType> ids){
      if(n + ids.size() > max_size){
        std::cout << "ERROR: insertion would cause overflow" << std::endl;
        abort();
      }
      parlay::random_generator gen(ids[0]);
      std::uniform_int_distribution<indexType> dis(0, max_size-1);
      parlay::sequence<std::pair<indexType, indexType>> id_slot_pairs(ids.size());
      // parlay::parallel_for(0, ids.size(), [&] (size_t i){
      for(size_t i=0; i<ids.size(); i++){
        indexType slot;
        auto r = gen[i];
        indexType start = dis(r);
        bool found = false;
        while(!found){
          bool expected = false;
          bool success = slots[start].compare_exchange_strong(expected, true);
          if(success){
            found = true;
            slot = start;
            break;
          }
          if(start < max_size-1) start++;
          else start = 0;
        }
        slot_to_id[slot] = ids[i];
        id_slot_pairs[i] = std::make_pair(ids[i], slot);
        T* destination = values.get() + aligned_dims*slot;
        std::memmove(data.begin()+i*dims, destination, sizeof(T)*dims);
      // });
      }
      for(auto p : id_slot_pairs){
        id_to_slot[p.first] = p.second;
        if(slot_to_id[p.second] != p.first){
          std::cout << "ERROR: slot and id do not match" << std::endl;
          std::cout << "Id: " << p.first << " maps to slot " << p.second << " in table" << std::endl;
          std::cout << "But slots[" << p.second << "] maps to id " << slots[p.second] << std::endl;
        }if(slots[p.second].load() != true){
          std::cout << "ERROR: slot " << p.second << " should be full, but is empty " << std::endl;
        }
      }
      n += ids.size();
    }


    //TODO convert ids to slots
    //TODO make sure absent slots cannot be deleted?
    void delete_points(parlay::sequence<indexType> ids){
      std::cout << "Preparing to delete " << ids.size() << " elements" << std::endl;
      parlay::sequence<indexType> slots_to_delete = parlay::tabulate(ids.size(), [&] (size_t i) {return id_to_slot[ids[i]];});
      auto old_ids = parlay::tabulate(slots_to_delete.size(), [&] (size_t i){
        return pool.retire(slots_to_delete[i]);
      });
      //TODO check for duplicates here
      auto to_delete = parlay::flatten(old_ids);
      std::cout << "Got " << to_delete.size() << " points to delete, after removing duplicates size is ";
      parlay::remove_duplicates(to_delete);
      std::cout << to_delete.size() << std::endl;
      check();
      for(size_t i=0; i<to_delete.size(); i++){
      // parlay::parallel_for(0, to_delete.size(), [&] (size_t i) {
        //do a CAS here as a way to check for errors
        indexType j = to_delete[i];
        // bool test = slots[j].load();
        bool expected = true;
        bool success = slots[j].load();
        slots[j].store(false);
        if(!success){
          std::cout << "ERROR: compare-and-swap on slot " << j << " (id " << slot_to_id[j] << ") failed" << std::endl; 
          std::cout << "Id " << slot_to_id[j] << " matches to slot " << id_to_slot[slot_to_id[j]] << std::endl;
          std::cout << "Test val: " << success << std::endl;
        }
      // });
      }
      std::cout << "Before deletion size: " << n;
      n -= to_delete.size();
      std::cout << ", After deletion size: " << n << std::endl;
    }

    void check(){
      auto to_check = parlay::tabulate(max_size, [&] (size_t i){
        if(slots[i].load()) return 1;
        else return 0;
      });
      size_t num_points = parlay::reduce(to_check);
      if(num_points != n){
        std::cout << "ERROR: num occuped slots " << num_points << " not equal to n: " << n << std::endl;
      } else std::cout << "Check passed" << std::endl;
    }



  private:
    std::shared_ptr<T[]> values;
    unsigned int dims;
    unsigned int aligned_dims;
    size_t max_size;
    parlay::sequence<indexType> slot_to_id;
    std::unordered_map<indexType, indexType> id_to_slot;
    parlay::sequence<std::atomic<bool>> slots;
    size_t n;
    epoch::memory_pool<indexType> pool;

  }; //end PointRange

  DynamicPointRange(){}

  DynamicPointRange(size_t n, unsigned int dims){PR = PointRange(n, dims);}

  DynamicPointRange(char* iFile){PR = PointRange(iFile);}

  //TODO how to retain id from announcement when releasing?
  //could return it with Get() and then make it be passed in by Release()?
  //what other potential ideas?

  struct wp{
    int worker_id;
    PointRange& PR;

    wp(PointRange& PR, int id): PR(PR), worker_id(id) {}

  };

  PointRange& Get_PointRange() {
    return PR;
  }

  wp Get_PointRange_Read_Only() {
    auto& epoch = epoch::get_epoch();
    auto [not_in_epoch, id] = epoch.announce();
    return wp(PR, id);
  }

  void Release_PointRange(int worker_id){
    auto& epoch = epoch::get_epoch();
    epoch.unannounce(worker_id);
  }

  void Release_PointRange(){}

  void save(char* oFile){PR.save(oFile);}

  private:
   PointRange PR;

}; //end DynamicPointRange
