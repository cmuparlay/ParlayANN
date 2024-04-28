#include <atomic>
#include <vector>
#include <limits>

#include "parlay/alloc.h"
#include "parlay/primitives.h"

#pragma once





// ***************************
// epoch structure
// ***************************

namespace epoch {

  constexpr int max_workers = 1024;

  inline int worker_id() { return parlay::my_thread_id(); }

  inline int num_workers() {
    return parlay::num_thread_ids();
  }

 
struct alignas(64) epoch_s {
        
  
  struct alignas(64) announce_slot {
    std::atomic<long> last;
    announce_slot() : last(-1l) {}
  };

  std::vector<announce_slot> announcements;
  std::atomic<long> current_epoch;
  epoch_s() {
    announcements = std::vector<announce_slot>(max_workers);
    current_epoch = 0;
  }

  void print_announce() {
    for(auto& ann : announcements)
      std::cout << ann.last << " ";
    std::cout << std::endl;
  }

  void clear_announce() {
    for(auto& ann : announcements)
      ann.last = -1;
  }

  long get_current() {
    return current_epoch.load();
  }
  
  long get_my_epoch() {
    return announcements[worker_id()].last;
  }

  void set_my_epoch(long e) {
    announcements[worker_id()].last = e;
  }

  std::pair<bool,int> announce() {
      size_t id = worker_id();
      // by the time it is announced current_e could be out of date, but that should be OK
      if (announcements[id].last.load() == -1) {
        announcements[id].last = get_current();
        return std::pair(true, id);
      } else {
        return std::pair(false, id);
      }
  }

  void unannounce(size_t id) {
    assert(announcements[id].last.load() != -1l);
    announcements[id].last.store(-1l, std::memory_order_release);
  }

  void update_epoch() {
    size_t id = worker_id();
    if (id >= max_workers) {
      std::cerr << id << " is too many threads for the epoch-based collector" << std::endl;
      abort();
    }
    int workers = num_workers();
    long current_e = get_current();
    bool all_there = true;
    // check if everyone is done with earlier epochs
    for (int i=0; i < workers; i++)
      if ((announcements[i].last != -1l) && announcements[i].last < current_e) {
        all_there = false;
        break;
      }
    // if so then increment current epoch
    if (all_there) {
      if (current_epoch.compare_exchange_strong(current_e, current_e+1)) {
      }
    }
  }

};

  extern inline epoch_s& get_epoch() {
    static epoch_s epoch;
    return epoch;
  }

// ***************************
// epoch pools
// *************************** 

template<typename indexType>
struct alignas(64) memory_pool {

    private:

      static constexpr double milliseconds_between_epoch_updates = 20.0;
      long update_threshold;
      using sys_time = std::chrono::time_point<std::chrono::system_clock>;

      // each thread keeps one of these
      struct old_current {
        std::vector<indexType> old;
        std::vector<indexType> current; // linked list of retired items from current epoch
        long epoch; // epoch on last retire, updated on a retire
        long count; // number of retires so far, reset on updating the epoch
        sys_time time; // time of last epoch update
        old_current() : epoch(0) {}
      };

      std::vector<old_current> pools;
      int workers;

      std::vector<indexType> add_to_current_list(indexType p) {
        auto i = parlay::worker_id();
        auto &pid = pools[i];
        auto ids = advance_epoch(i, pid);
        pid.current.push_back(p);
        return ids;
      }

      // returns list of objects to delete
      std::vector<indexType> clear_list(std::vector<indexType> &ids) {
        std::vector<indexType> id_copy = ids;
        ids.clear();
        return id_copy;
      }

      std::vector<indexType> advance_epoch(int i, old_current& pid) {
        epoch::epoch_s& epoch = epoch::get_epoch();
        std::vector<indexType> ids;
        if (pid.epoch + 1 < epoch.get_current()) {
          ids = clear_list(pid.old);
          pid.old = pid.current;
          pid.current.clear();
          pid.epoch = epoch.get_current();
        }
        // a heuristic
        auto now = std::chrono::system_clock::now();
        if (++pid.count == update_threshold  ||
            std::chrono::duration_cast<std::chrono::milliseconds>(now - pid.time).count() >
            milliseconds_between_epoch_updates * (1 + ((float) i)/workers)) {
          pid.count = 0;
          pid.time = now;
          epoch.update_epoch();
        }
        return ids;
      }


      
    public:
      
      memory_pool() {
        workers = epoch::max_workers;
        update_threshold = 2*workers;
        pools = std::vector<old_current>(workers);
        for (int i = 0; i < workers; i++) {
          pools[i].count = parlay::hash64(i) % update_threshold;
          pools[i].time = std::chrono::system_clock::now();
        }
      }

      memory_pool(const memory_pool&) = delete;
      ~memory_pool() {} 


      void stats() {
        epoch::get_epoch().print_announce();
        std::cout << "epoch number: " << epoch::get_epoch().get_current() << std::endl;
        for (int i=0; i < pools.size(); i++) {
          std::cout << "pool[" << i << "] = " << pools[i].old.size() << ", " << pools[i].current.size() << std::endl;
        }
      }

      std::vector<indexType> retire(indexType p){
        return add_to_current_list(p);
      }

        
    }; //end struct memory_pool


template <typename indexType>
extern inline memory_pool<indexType>& get_pool() {
  static memory_pool<indexType> pool;
  return pool;
}






} // end namespace epoch

