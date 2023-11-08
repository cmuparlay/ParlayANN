# Quickstart

The following is a crash course in quickly building and querying an index using ParlayANN.

First, download a 100K slice of the BIGANN dataset.

```bash
mkdir -p data && cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
```

Next, convert it from the .fvecs format to binary format:

```bash
cd ../data_tools
make vec_to_bin
./vec_to_bin float ../data/sift/sift_learn.fvecs ../data/sift/sift_learn.fbin
./vec_to_bin float ../data/sift/sift_query.fvecs ../data/sift/sift_query.fbin
```

Next, calculate its ground truth up to $k=100$. See the README in the data_tools folder for an explanation of each parameter.

```bash
cd ../data_tools
make compute_groundtruth
./compute_groundtruth -base_path ../data/sift/sift_learn.fbin -query_path ../data/sift/sift_query.fbin -data_type float -k 100 -dist_func Euclidian -gt_path ../data/sift/sift-100K
```

To build an index using Vamana and write it to an outfile, use the following commandline:
```bash
cd ../algorithms/vamana
make
./neighbors -R 32 -L 64 -alpha 1.2 two_pass 0 -graph_outfile ../../data/sift/sift_learn_32_64 -data_type float -dist_func Euclidian -base_path ../../data/sift/sift_learn.fbin
```

You should see the following output; timing will vary based on your machine and these times are taken from a machine with 72 cores:

```bash
Detected 100000 points with dimension 128
Building graph...
Pass 10% complete
Pass 20% complete
Pass 30% complete
Pass 40% complete
Pass 50% complete
Pass 60% complete
Pass 70% complete
Pass 80% complete
Pass 90% complete
Pass 100% complete
beam search time: total: 0.3436
bidirect time: total: 0.0557
prune time: total: 0.3751
Average visited: 68, Tail visited: 76
Vamana graph built with 100000 points and parameters R = 32, L = 64
Graph has average degree 26.94 and maximum degree 32
Graph built in 0.8123 seconds
Parlay time: 0.8138
Writing graph with 100000 points and max degree 32
```

To load and then query the index, use the following command:
```bash
cd ../algorithms/vamana
make
./neighbors -R 32 -L 64 -a 1.2 -graph_path ../../data/sift/sift_learn_32_64 -query_path ../../data/sift/sift_query.fbin -gt_path ../../data/sift/sift-100K -data_type float -dist_func Euclidian -base_path ../../data/sift/sift_learn.fbin
```

You should see an output similar to the following; timings were taken using a machine with 72 cores.

```bash
Detected 10000 points with num results 100
Detected 100000 points with dimension 128
Detected 10000 points with dimension 128
Detected 100000 points with max degree 32
Average visited: 0, Tail visited: 0
Vamana graph built with 100000 points and parameters R = 32, L = 64
Graph has average degree 26.9416 and maximum degree 32
Graph built in 0 seconds
For 10@10 recall = 0.11027, QPS = 5.05561e+06, Q = 10, cut = 1.35, visited limit = 6, degree limit: 16, average visited = 6, average cmps = 89
For 10@10 recall = 0.2032, QPS = 4.40141e+06, Q = 10, cut = 1.35, visited limit = 8, degree limit: 16, average visited = 8, average cmps = 115
For 10@10 recall = 0.36414, QPS = 2.58598e+06, Q = 10, cut = 1.35, visited limit = 10, degree limit: 19, average visited = 10, average cmps = 169
For 10@10 recall = 0.45495, QPS = 2.68528e+06, Q = 10, cut = 1.35, visited limit = 10, degree limit: 22, average visited = 10, average cmps = 195
For 10@10 recall = 0.59687, QPS = 2.35627e+06, Q = 10, cut = 1.35, visited limit = 10, degree limit: 25, average visited = 10, average cmps = 219
For 10@10 recall = 0.61139, QPS = 2.07857e+06, Q = 13, cut = 1.35, visited limit = 13, degree limit: 22, average visited = 13, average cmps = 245
For 10@10 recall = 0.7209, QPS = 1.86532e+06, Q = 11, cut = 1.35, visited limit = 11, degree limit: 28, average visited = 11, average cmps = 261
For 10@10 recall = 0.75036, QPS = 1.90006e+06, Q = 13, cut = 1.35, visited limit = 13, degree limit: 25, average visited = 13, average cmps = 274
For 10@10 recall = 0.81667, QPS = 1.80538e+06, Q = 15, cut = 1.35, visited limit = 15, degree limit: 25, average visited = 15, average cmps = 310
For 10@10 recall = 0.86956, QPS = 1.55304e+06, Q = 15, cut = 1.35, visited limit = 15, degree limit: 28, average visited = 15, average cmps = 339
For 10@10 recall = 0.92219, QPS = 1.4652e+06, Q = 15, cut = 1.35, visited limit = 15, degree limit: 32, average visited = 15, average cmps = 372
For 10@10 recall = 0.95779, QPS = 1.15009e+06, Q = 12, cut = 1.35, visited limit = 100000, degree limit: 32, average visited = 18, average cmps = 436
For 10@10 recall = 0.97133, QPS = 955658, Q = 17, cut = 1.35, visited limit = 100000, degree limit: 32, average visited = 22, average cmps = 529
For 10@10 recall = 0.98078, QPS = 775014, Q = 24, cut = 1.35, visited limit = 100000, degree limit: 32, average visited = 29, average cmps = 656
For 10@10 recall = 0.99151, QPS = 473530, Q = 45, cut = 1.35, visited limit = 100000, degree limit: 32, average visited = 49, average cmps = 1026
For 10@10 recall = 0.99509, QPS = 351296, Q = 70, cut = 1.35, visited limit = 100000, degree limit: 32, average visited = 71, average cmps = 1356
For 10@10 recall = 0.99912, QPS = 186532, Q = 180, cut = 1.35, visited limit = 100000, degree limit: 32, average visited = 145, average cmps = 2279
For 10@10 recall = 0.9995, QPS = 151930, Q = 250, cut = 1.35, visited limit = 100000, degree limit: 32, average visited = 174, average cmps = 2546
For 10@10 recall = 0.99995, QPS = 13560.6, Q = 1000, cut = 10, visited limit = 100000, degree limit: 32, average visited = 1003, average cmps = 7885
For 10@10 recall = 0.99995, QPS = 13560.6, Q = 1000, cut = 10, visited limit = 100000, degree limit: 32, average visited = 1003, average cmps = 7885
```

