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
//WARNING this test requires for every center to own points, even though a valid k-means output can include a few empty centers
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
      std::cout << "Centers diff: " << c[i*ad+coord] << " " << c[j*ad+coord] << std::endl;
      return false; //centers not equal
    }
  }
  return true;
}

template<typename T, typename CT>
bool equidistant_centers(T* v, CT* c, size_t pt_id, size_t cid1, size_t cid2, size_t d, size_t ad, Distance& D) {
   CT buf[2048];
    T* it = v+pt_id*ad;
    for (size_t el = 0; el < d; el++) buf[el]=*(it++);
  auto dist1 = D.distance(buf,c+cid1*ad,d);
  auto dist2 = D.distance(buf,c+cid2*ad,d);
  if ( std::abs(dist1-dist2) < .1) {
    return true;
  }
  else {
    return false;
  }
 
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
      EXPECT_EQ( (bests[i]==asg[i]) || equal_centers(c,bests[i],asg[i],d,ad) || equidistant_centers(v,c,i,bests[i],asg[i],d,ad,D),true) << "asg fail " << bests[i] << " " << asg[i] << std::endl;
      if (bests[i] != asg[i]){
        std::cout << "printing point that failed in float form" << std::endl;
        for (size_t coord = 0; coord < d; coord++) {
          std::cout << static_cast<float>(v[i*ad+coord]) << " ";
        }
        std::cout << std::endl;
        std::cout << "printed point that failed " << std::endl;
        break; //prevent overprinting
      }
    }

    delete[] bests; //memory cleanup
}


//given a kmeans method, n, d, k, run to convergence, and make sure that the result is a valid kmeans result
//by setting k=1, we get the k is 1 test implicitly (don't need a separate function) (by k is 1 test, I mean that the result of clustering with k=1 is that the single center is the average of all the points)
template<typename T, typename CT, typename index_type, typename Kmeans>
double kmeansConvergenceTest(T* v, size_t n, size_t d, size_t ad, size_t k, Distance& D, bool suppress_logging=true) {
    CT* c = new CT[k*ad]; // centers
    index_type* asg = new index_type[n];
    size_t max_iter=1000; //is 1000 iters sufficient? TODO
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

//given two kmeans methods, run both, and compare the output msse's
template<typename T, typename CT, typename index_type, typename Kmeans, typename Kmeans2>
void kmeansComparativeTest(T* v, size_t n, size_t d, size_t ad, size_t k,  Distance& D, size_t max_iter, double epsilon, bool suppress_logging=true) {
    CT* c = new CT[k*ad]; // centers
    index_type* asg = new index_type[n];
    

    //initialization
    Lazy<T,CT, index_type> init;
    //note that here, d=ad
    init(v,n,d,ad,k,c,asg);

    CT* c2 = new CT[k*ad];
    index_type* asg2 = new index_type[n];
    //copy over values from initialization into c2, asg2
    parlay::parallel_for(0,k*ad,[&] (size_t i) {
      c2[i]=c[i];
    });
    parlay::parallel_for(0,n,[&] (size_t i) {
      asg2[i]=asg[i];
    });

   
    Kmeans runner;
    kmeans_bench logger = kmeans_bench(n,d,k,max_iter,
    epsilon,"Lazy",runner.name());
    logger.start_time();
    //true at the end suppresses logging
    runner.cluster_middle(v,n,d,ad,k,c,asg,D,logger,max_iter,epsilon,suppress_logging);
    logger.end_time();

    Kmeans2 runner2;

    kmeans_bench logger2 = kmeans_bench(n,d,k,max_iter,epsilon,"Lazy",runner2.name());
    logger2.start_time();
    //true at the end suppresses logging
    runner2.cluster_middle(v,n,d,ad,k,c2,asg2,D,logger,max_iter,epsilon,suppress_logging);
    logger2.end_time();
    
    auto rangn = parlay::iota(n);
    float msse1 = parlay::reduce(parlay::map(rangn,[&] (size_t i) { 
      float buf[2048];
      T* it = v+i*ad;
      for (size_t i = 0; i < d; i++) buf[i]=*(it++);
      return D.distance(buf,c+asg[i]*ad,d);
    }))/n; //calculate msse

    float msse2 = parlay::reduce(parlay::map(rangn,[&] (size_t i) { 
      float buf[2048];
      T* it = v+i*ad;
      for (size_t i = 0; i < d; i++) buf[i]=*(it++);
      return D.distance(buf,c2+asg2[i]*ad,d);
    }))/n; //calculate msse

    //avoid weird division by very small numbers? TODO (or is this >0 instead of >0.0001 okay)
    //make sure that the msse values are within 1% of each other
    if (msse1 > 0 && msse2 > 0) {
      EXPECT_LE(std::abs(msse1-msse2)/msse1, 0.01 ) << runner.name() << " : " << msse1 << ", " << runner2.name() << " : " << msse2 << "\n";

    }
    else {
      EXPECT_LE(std::abs(msse1-msse2),.0001);
    }

    delete[] c;
    delete[] asg;
    delete[] c2;
    delete[] asg2;


    

  }