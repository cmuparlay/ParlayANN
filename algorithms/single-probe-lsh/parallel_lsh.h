#pragma once

#include <algorithm>
#include <bitset>
#include <map>

#include "parameters.h"

namespace grann {

  const int                       BITSET_MAX = 64;
  typedef std::bitset<BITSET_MAX> bitstring;

  // NOTE :: good efficiency when total_vec_size is integral multiple of 64
  inline void prefetch_vector(const char* vec, uint64_t vecsize) {
    uint64_t max_prefetch_size = (vecsize / 64) * 64;
    for (uint64_t d = 0; d < max_prefetch_size; d += 64)
      _mm_prefetch((const char*) vec + d, _MM_HINT_T0);
  }

  // NOTE :: good efficiency when total_vec_size is integral multiple of 64
  inline void prefetch_vector_l2(const char* vec, uint64_t vecsize) {
    uint64_t max_prefetch_size = (vecsize / 64) * 64;
    for (uint64_t d = 0; d < max_prefetch_size; d += 64)
      _mm_prefetch((const char*) vec + d, _MM_HINT_T1);
  }

  class HashTable {
   public:
    HashTable(uint32_t table_s, uint32_t vector_d) {
      if (table_s > BITSET_MAX) {
        perror("Input table size is too large");
        exit(1);
      }

      random_hps.reserve(table_s);

      vector_dim = vector_d;
      table_size = table_s;
    }
    ~HashTable() {}

    void generate_hps(size_t i) {
      std::random_device              r;
      std::default_random_engine      rng{r()};
      std::normal_distribution<float> gaussian_dist;
      for (size_t i = 0; i < table_size; i++) {
        std::vector<float> random_hp;
        random_hp.reserve(vector_dim);
        for (size_t j = 0; j < vector_dim; j++) {
          float add = gaussian_dist(rng);
          random_hp.push_back(add);
        }
        add_hp(random_hp);
      }
    }

    parlay::sequence<uint32_t>& get_bucket(bitstring bucket_id) {
      auto seq = parlay::delayed_seq<size_t>(buckets.size(), [&] (size_t i) { return buckets[i].first; });
      size_t index = parlay::internal::binary_search(seq, bucket_id.to_ulong(), std::less<size_t>());
      assert(bucket_id.to_ulong() == buckets[index].first);
      return buckets[index].second;
    }

    template<typename T>
    bitstring get_hash(const T *input_vector) {
      bitstring input_bits;
      for (size_t i = 0; i < table_size; i++) {
        // float dot_p = std::inner_product(random_hps[i].begin(),
        // random_hps[i].end(), input_vector, 0.0);
        float dot_p = 0.0;
        for (size_t j = 0; j < vector_dim; j++) {
          float x = random_hps[i][j] * input_vector[j];
          dot_p += x;
        }

        if (dot_p > 0)
          input_bits[i] = 1;
        else
          input_bits[i] = 0;
      }
      return input_bits;
    }

  void add_hp(std::vector<float> hp) {
    random_hps.push_back(hp);
  }

  void set_buckets(parlay::sequence<std::pair<size_t, parlay::sequence<uint32_t>>>&& B) {
    buckets = std::move(B);
  }

   protected:
    uint32_t vector_dim;  // dimension of points stored/each hp vector
    uint32_t table_size;  // number of hyperplanes
    std::vector<std::vector<float>>     random_hps;
    parlay::sequence<std::pair<size_t, parlay::sequence<uint32_t>>> buckets;
  };


  // Simple Neighbor with a flag, for remembering whether we already explored
  // out of a vertex or not.
  struct Neighbor {
    uint32_t  id = 0;
    float distance = std::numeric_limits<float>::max();
    bool  flag = true;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f = true)
        : id{id}, distance{distance}, flag(f) {
    }

    inline bool operator<(const Neighbor &other) const {
      return distance < other.distance;
    }
    inline bool operator==(const Neighbor &other) const {
      return (id == other.id);
    }
  };


  // Given a  neighbor array of size K starting with pointer addr which is
  // sorted by distance, and a new neighbor nn, insert it in correct place if it
  // can fit. Else dont do anything. If element is already in Pool, returns K+1.
  static inline unsigned InsertIntoPool(Neighbor *addr, unsigned K,
                                        Neighbor nn) {
    // find the location to insert
    unsigned left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
      memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
      addr[left] = nn;
      return left;
    }
    if (addr[right].distance < nn.distance) {
      addr[K] = nn;
      return K;
    }
    while (right > 1 && left < right - 1) {
      unsigned mid = (left + right) / 2;
      if (addr[mid].distance > nn.distance)
        right = mid;
      else
        left = mid;
    }
    // check equal ID

    while (left > 0) {
      if (addr[left].distance < nn.distance)
        break;
      if (addr[left].id == nn.id)
        return K + 1;
      left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)
      return K + 1;
    memmove((char *) &addr[right + 1], &addr[right],
            (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
  }


  template<typename T>
  class LSHIndex {
   public:

     LSHIndex(parlay::sequence<Tvec_point<T>*> &v) : v(v) {
       num_tables = 0;
       table_size = 0;
       vector_dim = 0;
     }

     ~LSHIndex() {
     }

    void build(const Parameters &params) {
      num_tables = params.Get<uint32_t>("num_tables");
      table_size = params.Get<uint32_t>("table_size");
      tables.reserve(num_tables);

      uint32_t dim = v[0]->coordinates.size();

      // 1. generate hyperplanes for the table
      for (size_t i = 0; i < num_tables; i++) {
        HashTable table = HashTable(table_size, dim);
        table.generate_hps(i);
        tables.push_back(table);
      }

      size_t j = 0;
      for (auto &table : tables) {
        std::cout << "Building table: " << j << std::endl;

        auto hash_ids = parlay::sequence<std::pair<size_t, uint32_t>>::from_function(v.size(), [&] (size_t i) {
          const T* cur_vec = v[i]->coordinates.begin();
          bitstring cur_vec_hash = table.get_hash(cur_vec);
          return std::make_pair(cur_vec_hash.to_ulong(), i);
        });
        auto buckets = parlay::group_by_key_ordered(hash_ids);

        table.set_buckets(std::move(buckets));

        ++j;
      }
    }

  // Returns the number of neighbors retrieved.
  uint32_t search(T *query, uint32_t dim, uint32_t res_count,
                           const Parameters &search_params, uint32_t *indices,
                           float *distances, size_t* comps) {
    std::vector<uint32_t> candidates;
    for (auto &table : tables) {
      bitstring         query_hash = table.get_hash(query);
      parlay::sequence<uint32_t>& curr_bucket = table.get_bucket(query_hash);
      candidates.insert(candidates.end(), curr_bucket.begin(),
                        curr_bucket.end());
    }

    std::vector<Neighbor> best_candidates(res_count + 1);
    uint32_t                  curr_size = 0;
    uint32_t                  max_size = res_count;
    uint32_t                  cmps = 0;
    std::set<uint32_t>  inserted;

    process_candidates_into_best_candidates_pool(
        query, dim, candidates, best_candidates, max_size, curr_size, inserted,
        cmps);
    comps[0]=(size_t) cmps;

    res_count = curr_size < res_count ? curr_size : res_count;
    for (uint32_t i = 0; i < res_count; i++) {
      indices[i] = best_candidates[i].id;

      if (distances != nullptr) {
        distances[i] = best_candidates[i].distance;
      }
    }

    // TODO
    *comps = 0;

    return res_count;
  }

  uint32_t process_candidates_into_best_candidates_pool(
      T* &node_coords, uint32_t dim, std::vector<uint32_t> &cand_list,
      std::vector<Neighbor> &top_L_candidates, const uint32_t maxListSize,
      uint32_t &curListSize, std::set<uint32_t> &already_inserted,
      uint32_t &total_comparisons) {
    uint32_t best_inserted_position = maxListSize;

    for (unsigned m = 0; m < cand_list.size(); ++m) {
      unsigned id = cand_list[m];
      if (already_inserted.find(id) == already_inserted.end()) {
        already_inserted.insert(id);

        if ((m + 1) < cand_list.size()) {
          auto nextn = cand_list[m + 1];
          prefetch_vector(
              (const char *) v[nextn]->coordinates.begin(),
              sizeof(T) * dim);
        }

        total_comparisons++;
        Euclidian_Distance D = Euclidian_Distance();
        float dist = D.distance(node_coords, v[id]->coordinates.begin(), dim);

        if (curListSize > 0 &&
            dist >= top_L_candidates[curListSize - 1].distance &&
            (curListSize == maxListSize))
          continue;

        Neighbor nn(id, dist, true);
        unsigned r = InsertIntoPool(top_L_candidates.data(), curListSize, nn);
        if (curListSize < maxListSize)
          ++curListSize;  // candidate_list has grown by +1
        if (r < best_inserted_position)
          best_inserted_position = r;
      }
    }
    return best_inserted_position;
  }

   protected:
    const parlay::sequence<Tvec_point<T>*> &v;
    uint32_t                   num_tables;
    uint32_t                   table_size;
    uint32_t                   vector_dim;
    std::vector<HashTable> tables;
  };

}  // namespace grann
