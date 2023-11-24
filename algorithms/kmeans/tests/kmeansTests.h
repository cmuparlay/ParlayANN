//header for kmeans testing
//includes tests that should be run for any k-means implementation


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
template<typename CT>
bool equal_centers(CT* c, size_t i, size_t j, size_t d, size_t ad) {
  for (size_t coord = 0; coord < d; coord++) {
    if (std::abs(c[i*ad+coord] - c[j*ad+coord]) > .1) {
      return false; //centers not equal
    }
  }
  return true;
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
      EXPECT_EQ( (bests[i]==asg[i]) || equal_centers(c,bests[i],asg[i],d,ad),true) << "asg fail " << bests[i] << " " << asg[i];
      if (bests[i] != asg[i]) break; //prevent overprinting
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