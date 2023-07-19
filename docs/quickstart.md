# Quickstart

The following is a crash course in quickly building and querying an index using ParlayANN.

First, download a 100K slice of the BIGANN dataset.

```bash
mkdir -p data && cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
```

Next, calculate its ground truth up to $k=100$. See the README in the data_tools folder for an explanation of each parameter.

```bash
cd ../data_tools
make compute_groundtruth
./compute_groundtruth -base_path ../data/sift/sift_learn.fvecs -query_path ../data/sift/sift_query.fvecs -file_type vec -data_type float -k 100 -dist_func Euclidian -gt_path ../data/sift/sift-100K
```

To build an index using Vamana and write it to an outfile, use the following commandline:
```bash
cd ../algorithms/vamana
make
./neighbors -R 32 -L 64 -a 1.2 -graph_outfile ../../data/sift/sift_learn_32_64 -data_type float -file_type vec -dist_func Euclidian -base_path ../../data/sift/sift_learn.fvecs
```

You should see the following output; timing will vary based on your machine and these times are taken from a machine with 48 cores:

```bash
Building graph...
Medoid ID: 29429
Index build 10% complete
Index build 20% complete
Index build 30% complete
Index build 40% complete
Index build 50% complete
Index build 60% complete
Index build 70% complete
Index build 80% complete
Index build 90% complete
Index build 100% complete
ANN: Built index: 0.7683
Index built with average degree 26.9 and max degree 32
ANN: stats: 0.0007
Parlay time: 0.7710
Writing graph with 100000 points and max degree 32
```

To load and then query the index, use the following command:
```bash
cd ../algorithms/vamana
make
./neighbors -R 32 -L 64 -a 1.2 -graph_path ../../data/sift/sift_learn_32_64 -query_path ../../data/sift/sift_query.fvecs -gt_path ../../data/sift/sift-100K -data_type float -file_type vec -dist_func Euclidian -base_path ../../data/sift/sift_learn.fvecs
```

You should see an output similar to the following; timings were taken using a machine with 48 cores.

```bash
Medoid ID: 29429
Average visited: 0, Tail visited: 0
Vamana graph built with 100000 points and parameters R = 32, L = 64
Graph has average degree 26.8981 and maximum degree 32
Graph built in 0 seconds
For recall = 0.32854, QPS = 3.97456e+06
For recall = 0.67768, QPS = 2.84738e+06
For recall = 0.83234, QPS = 2.2604e+06
For recall = 0.89642, QPS = 1.87793e+06
For recall = 0.92684, QPS = 1.60514e+06
For recall = 0.94427, QPS = 1.41263e+06
For recall = 0.95558, QPS = 1.2427e+06
For recall = 0.97583, QPS = 901469
For recall = 0.99146, QPS = 535762
For recall = 0.99603, QPS = 344851
For recall = 0.99939, QPS = 32946.8
Parlay time: 1.7924
```

