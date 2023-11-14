//Yinyang method for accelerating exact kmeans

#ifndef YY
#define YY

#include "parlay/primitives.h"
#include "parlay/internal/get_time.h"
#include "distance.h"
#include "initialization.h"
#include "naive.h"
#include "kmeans_bench.h"

template<typename T, typename Point, typename index_type, typename CT, typename CenterPoint>
struct Yinyang : KmeansInterface<T,Point,index_type,CT,CenterPoint> {

  //note that a point does not own its lbs, instead they are stored in a matrix
  struct point {
    index_type best; // the index of the best center for the point
    parlay::slice<T*, T*> coordinates; // the coordinates of the point

    index_type id; //an id used for the point, for debugging purposes
    float ub;
    float global_lb; //global lower bound
    index_type old_best; //the previous best

    point(index_type id, index_type chosen, parlay::slice<T*,T*> coordinates) : best(chosen),
    coordinates(coordinates.begin(),coordinates.end()),  id(id),
    ub(std::numeric_limits<float>::max()) {

    }

  };

  struct center {
    index_type id; // a unique identifier for the center
    index_type group_id; //the id of the group that the center belongs to
    parlay::sequence<CT> coordinates; // the coordinates of the center
    float delta;
    index_type old_num_members; //how many points belonged to that center
    index_type new_num_members; //the number of members this iter
    bool has_changed; //check if the center has changed

    center(index_type id, parlay::sequence<CT> coordinates) : id(id),group_id(-1),coordinates(coordinates),delta(0),old_num_members(0),new_num_members(0),has_changed(true) {

    }

  
  };
  
  struct group {
    index_type id;
    //store the ids of all the centers belonging to this group
    parlay::sequence<index_type> center_ids;

    float max_drift;
  };

  //initialize the groups by running a naive kmeans a few times
  void init_groups( size_t d,size_t ad, size_t k,  CT* c, parlay::sequence<center>& centers, parlay::sequence<group>& groups, size_t t, Distance& D) {

    //cluster on the groups initially using NaiveKmeans
    //not bothering with aligned_dimension for clustering centers into groups
    CT* group_centers = new CT[t * ad];
    size_t* group_asg = new size_t[k];

    Lazy<CT,CT,size_t> init;
    init(c,k,d,ad,t,group_centers,group_asg);

    kmeans_bench logger = kmeans_bench(k,d,t,5,0,"Lazy currently", "Internal Naive");
    logger.start_time();
    
    
    NaiveKmeans<CT,Euclidian_Point<CT>,size_t,CT,Euclidian_Point<CT>> run;
    run.cluster_middle(c,k,d,ad,t,
    group_centers, group_asg,D,logger, 5, 0.0001,true);
    logger.end_time();

    parlay::parallel_for(0,k,[&] (size_t i) {
      centers[i].group_id = group_asg[i];
    });

    //set group ids
    for (size_t i = 0; i < t; i++) groups[i].id = i;
    
    //TODO do in parallel?
    //sequential assigning the groups their centers
    for (size_t i = 0; i < k; i++) {
      groups[centers[i].group_id].center_ids.push_back(i);
      
    }

    delete[] group_centers; //memory cleanup
    delete[] group_asg;


  }

  //confirm that the groups are nonempty
  void assert_proper_group_size(parlay::sequence<group>& groups, 
   size_t t, bool DEBUG_FLAG=false) {
    
    for (size_t i =0 ;i < t; i++) {
      if (groups[i].center_ids.size() == 0) {
        std::cout << 
        "Group assignment went wrong, group is wrong size"
        << std::endl;
        std::cout << groups[i].center_ids.size() << std::endl;
        abort();
      }
    }
  }

  //computes the drift, then initializes the new centers
  float update_centers_drift(size_t d, size_t ad,
  size_t k, CT* center_calc_float, parlay::sequence<center>& centers, 
  parlay::sequence<group>& groups, size_t t, Distance& D) {

    std::cout << "enter ucd" <<std::endl;
   
    // Check convergence
    parlay::parallel_for (0,k,[&] (size_t i) { 
      centers[i].delta = sqrt_dist(
        parlay::make_slice(centers[i].coordinates).begin(), 
      center_calc_float+i*ad,d,D);
    });
        std::cout << "seg?" <<std::endl;

    //max_diff is the largest center movement
    float max_diff = *parlay::max_element(parlay::map(centers,[&] (center& cen) {
      return cen.delta;
    }));
        std::cout << "seg?2" <<std::endl;


    //Copy over new centers
    parlay::parallel_for(0,k,[&] (size_t j) {
      for (size_t coord = 0; coord < d; coord++) {
         centers[j].coordinates[coord] = center_calc_float[j*ad+coord];

      }
     
    });
        std::cout << "seg?3" <<std::endl;


    //for each group, get max drift for group
    parlay::parallel_for(0,t,[&] (size_t i) {
      auto drifts = parlay::map(groups[i].center_ids, [&] (size_t j) {
      return centers[j].delta; });
      
      groups[i].max_drift = *max_element(drifts);

    });
        std::cout << "seg?4" <<std::endl;


    return max_diff;

  }

  //update the lb's of a point given the drifts of each group
  void set_point_global_lb(point& p, parlay::sequence<group>& groups,
  size_t t,parlay::sequence<parlay::sequence<float>>& lbs) {
    p.global_lb = std::numeric_limits<float>::max();
    for (size_t j = 0; j < t; j++) {

      lbs[p.id][j] = std::max(static_cast<float>(0), lbs[p.id][j]-groups[j].max_drift);
      //reduce the global lower bound if possible
      //TODO which is better, if check or a min (given that the if check may prevent a write, which is good) (given that the min is more concise which looks nice)
      if (p.global_lb < lbs[p.id][j]) {
        p.global_lb=lbs[p.id][j];
      }
      //p.global_lb = std::min(p.global_lb,p.lb[j]);
   
    }

  }

  //yinyang needs sqrt dist for triangle inequality to work
  float sqrt_dist(CT* a, CT* b, size_t d, Distance& D) {
    return std::sqrt(D.distance(a,b,d));
  }

  //make sure that the centers collectively own exactly n points (assertion)
  void assert_members_n(size_t n, size_t k, parlay::sequence<center>& centers) {
    size_t elt_counter = parlay::reduce(parlay::map(centers,[&] (center& my_center) {
      return my_center.old_num_members;
    }));
  
    if (elt_counter != n) {
      std::cout << "error in num_members assignment: " << elt_counter << std::endl;
      abort();
    }
  }

  //run yy
  void cluster_middle(T* v, size_t n, size_t d, size_t ad, size_t k, CT* c, index_type* asg, 
  Distance& D, kmeans_bench& logger, size_t max_iter, double epsilon,bool suppress_logging=false) {

    std::cout << "initial assign: " << std::endl;
    for (size_t i = 0; i < n; i++ ) {
      std::cout << asg[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "init cen: " << std::endl;
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < d; j++) {
        std::cout << c[i*ad+j] << " ";
      }
      std::cout << std::endl;
    }

    parlay::internal::timer tim = parlay::internal::timer();
    tim.start();
    float assignment_time = 0;
    float update_time = 0;
    float setup_time = 0;

    //when we do no iterations, nothing needs to happen at all
    if (max_iter == 0) return;

    //we copy a point into a float buffer of fixed size 2048, so we don't support d > 2048
    if (d > 2048) {
      std::cout << "d greater than 2048, too big, printing d: " << 
      d << std::endl;
      abort();
    }
    
    //create the centers
    parlay::sequence<center> centers = parlay::tabulate<center>(k, [&] (size_t i) {
    return center(i, parlay::sequence<CT>(d));
    });

    //fill in the centers
    parlay::parallel_for(0,k, [&] (size_t i) {
      for (size_t j = 0; j < d; j++) {
        centers[i].coordinates[j] = *(c + i*ad + j);
      }
    });
    
    //t is the number of groups (notation following paper)
    //We want t to be big without overloading memory because our memory cost contains O(nt) 
    //leaving at t=k/10 for now to allow for more consistent benching, but should fiddle with
    //TODO optimize
    //we take the max with 1 to make sure that t is positive (because for k < 10, k/10 = 0)
    size_t t=std::max(static_cast<size_t>(1),k/10);
    
    //initialize the groups
    parlay::sequence<group> groups(t);
    std::cout << "hi" << std::endl;
    init_groups(d,ad,k,c,centers,groups,t,D);
    std::cout << "bye" <<std::endl;
    assert_proper_group_size(groups,t,false); //confirm groups all nonempty

    //Init the points
    parlay::sequence<point> pts = parlay::tabulate<point>(n, [&] (size_t i) {
      //is new needed here? TODO delete later 
      //TODO does point need a slice of d or ad? think d
      return point(i,asg[i],parlay::slice(v+i*ad, v+i*ad + ad));

    });

    std::cout << "echo " << std::endl;

    //per recommendation using a matrix of lbs instead of initializing independently
    parlay::sequence<parlay::sequence<float>> lbs(n,parlay::sequence<float>(t,std::numeric_limits<float>::max()));
    //Init the point bounds
    std::cout << "echo2 " << std::endl;

    parlay::parallel_for(0,n,[&] (size_t i) {
  
      //first, we find the closest center to each point
      CT buf[2048];
      T* it = pts[i].coordinates.begin();
      for (size_t j = 0; j < d; j++) buf[j]= *(it++);
        
      auto distances = parlay::delayed::map(centers, [&](center& q) {
          return D.distance(buf, make_slice(q.coordinates).begin(),d);
      });

      pts[i].best = min_element(distances) - distances.begin();
      pts[i].old_best = pts[i].best;
      pts[i].ub = std::sqrt(distances[pts[i].best]);

      //TODO sadly linear in k, improve?
      for (size_t j = 0; j < k; j++) {
        float new_lb = std::sqrt(distances[j]);
        //TODO using an if check instead of a min, faster?
        if (j != pts[i].best && new_lb < lbs[i][centers[j].group_id]) lbs[i][centers[j].group_id] = new_lb;
        
      }
    });

   std::cout << "echo3 " << std::endl;
   std::cout << "printing state" << std::endl;
   for (size_t i = 0; i < n; i++) {
    std::cout << i << " " << pts[i].best << std::endl;
   }
   std::cout << "print lbs: " << std::endl;
   for (size_t i = 0; i < n; i++ ) {
    for (size_t j = 0; j < t; j++) {
      std::cout << lbs[i][j] << " ";
    }
    std::cout << std::endl;
   }


    assignment_time = tim.next_time();
  

    //compute num_members for each center
    //TODO which is faster, histogram by key or integer sort type method?
    //center_member_dist stores how many members (points) each center owns
    auto center_member_dist = parlay::histogram_by_key(parlay::map(pts,[&] (point& p) {
      return p.best;
    }));
    parlay::parallel_for(0,k,[&] (size_t i) {
      centers[i].has_changed = true;
      centers[i].new_num_members = 0;
      centers[i].old_num_members = 0;
    });

    //caution: can't use k here! some centers get no points*
    parlay::parallel_for(0,center_member_dist.size(),[&] (size_t i) {
      
      centers[center_member_dist[i].first].new_num_members=center_member_dist[i].second;
      centers[center_member_dist[i].first].old_num_members=center_member_dist[i].second;

    });

    //debugging (confirm all points belong to a center)
    std::cout << "errr" << std::endl;
     auto center_member_dist3 = parlay::histogram_by_key(parlay::map(pts,[&] (point& p) {
      return p.best;
    }));
    assert_members_n(n,k,centers);
    
    //iters start at 1 as we have already done a closest point check
    size_t iters = 1; 
    float max_diff = 0.0;
    //keep track of the number of distance calculations
    parlay::sequence<size_t> distance_calculations(n,k); 
    //keep track of the number of points reassigned in an iteration
    parlay::sequence<uint8_t> center_reassignments(k,1);
   
    //for center calculation
    CT* new_centers = new CT[k*ad];
  
  

    setup_time = tim.next_time();
    std::cout << "before iters "<< std::endl;

    //our iteration loop, will stop when we've done max_iter iters, or if we converge (within epsilon)
    while (true) {

      //TODO use yy-style comparative compute_centers in future iterations (once it is actually faster)
      //copying over to c array to use a shared compute_centers function with naive. Copying this should be relatively cheap so not concerned TODO is this actually cheap?
      parlay::parallel_for(0,k,[&] (size_t i) {
        for (size_t j = 0; j < d; j++) {
          c[i*ad+j]=centers[i].coordinates[j];
        }
      });
      parlay::parallel_for(0,n,[&] (size_t i) {
        asg[i]=pts[i].best;
      });
      std::cout << "before comcen "<< std::endl;

      this->compute_centers(v,n,d,ad,k,c,new_centers,asg);
      std::cout << "before drift "<< std::endl;

      max_diff = update_centers_drift(d,ad,k,new_centers,centers,groups,t,D);
     
           std::cout << "before msse "<< std::endl;

      //TODO this is not a correct calculation for the msse because ub may not be tight, but for purposes of logging we accept (because taking the true distance would be expensive and throw off the accuracy of benching)
      float msse = parlay::reduce(parlay::delayed_tabulate(n,[&] (size_t i) {
        return pts[i].ub * pts[i].ub;
      }))/n; 

      update_time = tim.next_time();
      std::cout << "before logger "<< std::endl;
 auto center_member_dist5 = parlay::histogram_by_key(parlay::map(pts,[&] (point& p) {
      return p.best;
    }));
      //end of iteration stat updating
      if (!suppress_logging) {
         logger.add_iteration(assignment_time,update_time,msse,parlay::reduce(distance_calculations),
      parlay::reduce(center_reassignments),parlay::map(centers,[&] (center& cen) {
        return cen.delta;
      }),setup_time);
      }

      assignment_time=0;
      update_time=0;
      setup_time=0;
      //convergence check
      if (iters >= max_iter || max_diff <= epsilon) break;
      std::cout << "new iter "<< std::endl;

      iters += 1; //start a new iteration
      parlay::parallel_for(0,n,[&] (size_t i) {
        distance_calculations[i]=0;
        center_reassignments[i]=0;
      });
     
      if (!suppress_logging) {
        std::cout << "iter: " << iters << std::endl;
      }

      //set centers changed to false, update old_num_members
      parlay::parallel_for(0,k,[&] (size_t i) {
        centers[i].has_changed = false;
        centers[i].old_num_members=centers[i].new_num_members;
      });

            std::cout << "pre asg "<< std::endl;

 auto center_member_dist7 = parlay::histogram_by_key(parlay::map(pts,[&] (point& p) {
      return p.best;
    }));
      //3.2: Group filtering (assign step)      
      parlay::parallel_for(0,n,[&](size_t i) {

        //update bounds and old_best
        pts[i].ub += centers[pts[i].best].delta; 
        pts[i].old_best = pts[i].best; 
        set_point_global_lb(pts[i],groups,t,lbs);

        //nothing happens if our closest center can't change
        if (pts[i].global_lb >= pts[i].ub) {
          return;
        }

        //copy point to float buffer
        CT buf[2048];
        T* it = pts[i].coordinates.begin();
        for (size_t coord = 0; coord < d; coord++) buf[coord]=* (it++);
        
        //tighten the upper bound
        pts[i].ub = sqrt_dist(buf,
        parlay::make_slice(centers[pts[i].best].coordinates).begin(),
        d,D);
        distance_calculations[i] += 1;

        //again, nothing happens if our closest center can't change
        if (pts[i].global_lb >= pts[i].ub) {
          return;
        }

        //for each group
        for (size_t j = 0; j < t; j++) {
          //if group j is too far away we don't look at it
          if (pts[i].ub <= lbs[i][j]) {
            continue;
          }
                   
          //reset the lower bound, make it smallest distance we calculate that's not the closest distance away
          lbs[i][j] = std::numeric_limits<float>::max(); 
            
          //for each group member (center)
          for (size_t l = 0; l < groups[j].center_ids.size(); l++) {

              //don't do a distance comparison with the previous best
              if (pts[i].old_best == groups[j].center_ids[l]) {
                continue;
              }

              //find distance to center l in group j
              float new_d = sqrt_dist(
              buf,
              parlay::make_slice(
                centers[groups[j].center_ids[l]].coordinates).begin(),
              d,D);

              //increment distance calc counter for this pt
              distance_calculations[i]++; 

              //note that the ub is tight rn,
              //that ub IS the distance to the previously 
              //closest center
              //So if our new dist is less than ub, we have
              //a new closest point
              if (pts[i].ub > new_d) {
                //the group with the previous center gets a slightly
                //lower bound, because the distance to the old center can 
                //become a lower bound
                //minus needed because of the center change from this iter
                //TODO is this a correct adjustment of the lower bound?? consider again
                lbs[i][centers[pts[i].best].group_id]=
                std::max(pts[i].ub-centers[pts[i].best].delta, std::min(pts[i].ub,
                lbs[i][centers[pts[i].best].group_id] - centers[pts[i].best].delta));
                
                pts[i].best=groups[j].center_ids[l];
                //new ub is tight
                pts[i].ub = new_d;

                //log center reassign
                //TODO read before write (reading cheaper)
                center_reassignments[i] = 1;
                //mark centers have changed. yes this is a race, but because we are setting false to true this is fine
                centers[pts[i].best].has_changed = true;
                centers[pts[i].old_best].has_changed=true;
                  
              }
              else {
                //if this center is not the closest, use it to improve the lower bound
                if (lbs[i][j] > new_d) {
                  lbs[i][j]=new_d;
                }
              
              }
          }  
        }
      }); //gran 1? I think helps. TODO
            std::cout << "post asg "<< std::endl;
 auto center_member_dist9 = parlay::histogram_by_key(parlay::map(pts,[&] (point& p) {
      return p.best;
    }));
      assignment_time = tim.next_time();
            std::cout << "about to create "<< std::endl;

            std::cout << "printing best"<<std::endl;
            // for (size_t i = 0; i < pts.size(); i++) {
            //   std::cout << i << " " << pts[i].best << std::endl;
            // }

            parlay::parallel_for(0,n,[&] (size_t i) {
        asg[i]=pts[i].best;
      });

      std::cout << "transferred to asg" << std::endl;
      auto rangn = parlay::delayed_tabulate(n,[&] (size_t i) { return i;});
      auto new_center_member_dist = parlay::histogram_by_key(parlay::map(rangn,[&] (size_t i) {return asg[i];}));

      //record num_new_members for each center
      // auto new_center_member_dist = parlay::histogram_by_key(parlay::map(pts,[&] (point& p) {
      //   return p.best;
      // }));
      

                  std::cout << "post create "<< std::endl;

      parlay::parallel_for(0,new_center_member_dist.size(),[&] (size_t i) {
        centers[new_center_member_dist[i].first].new_num_members = new_center_member_dist[i].second;

      });
   
      
      setup_time = tim.next_time();
                  std::cout << "finished round"<< std::endl;


    }
   
    //copy back over coordinates
    //put our data back 
    parlay::parallel_for(0,k,[&] (size_t i) {
      for (size_t j = 0; j < d; j++) {
        c[i*ad + j] = centers[i].coordinates[j];
      }
    });
    parlay::parallel_for(0,n,[&] (size_t i) {
        asg[i] = pts[i].best;
    });


    delete[] new_centers;

  }

};

#endif //YYIMP