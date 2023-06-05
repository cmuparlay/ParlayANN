# Algorithms

The algorithms in this folder share a common main file and thus a common commandline interface. The commandline interface allows the user to build an ANNS graph and write it to an outfile, load a graph and search it, or build and search in one shot. It contains several "generic" parameters that can be repurposed for a new benchmark. In the following examples, we provide instructions for building indices using bash. The instructions assume that the user has downloaded and built groundtruth for the 100K slice of the BIGANN dataset, as shown in the quickstart instructions.

### Universal Parameters

#### Parameters for building:
1. **-graph_outfile**: if graph is not already built, path the graph is written to. This is optional; if not provided, the graph will be built and will print timing and statistics before terminating.
2. **-file_type**: the type of the base, query, and groundtruth files. Current options are "vec" for .vecs and "bin" for .bin.
3. **-data_type**: type of the base and query vectors. Currently "float", "int8", and "uint8" are supported.
4. **-dist_func**: the distance function to use when calculating nearest neighbors. Currently Euclidian distance ("euclidian") and maximum inner product search ("mips") are supported.
5. **-base_path**: path to the base file.

#### Parameters for searching:

1. **-gt_path**: path to the ground truth, either in .ivecs or .ibin format.
2. **-graph_path**: path to the ANNS graph in the case of using an already built graph.
3. **-query_path**: path to the queries in .bin or .vecs format.
4. **-res_path**: path where a CSV file of results can be written (it is written to in append form, so it can be used to collect results of multiple runs).


### Generic Parameters

The following commandline parameters are designed to allow users to use the existing interface 

1. **-R**: an integer.
2. **-L**: an integer.
3. **-a**: a double.
4. **-d**: a double.
5. **-memory_flag**: a flag used to indicate how much memory to allocate for the graph. Setting to 0 means that each point will be allocated $R$ edges, while setting it to 1 indicates allocating $L*R$ edges per point.

## Vamana (DiskANN)

Vamana, also known as DiskANN, is an algorithm introduced in [DiskANN: Fast Accurate Billion-point Nearest
Neighbor Search on a Single Node](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf) by Subramanya et al., with original code in the [DiskANN repo](https://github.com/microsoft/DiskANN). It builds a graph incrementally, and its insert procedure does a variant on greedy search or beam search with a frontier size $L$ on the existing graph and uses the nodes visited during the search as edge candidates. The visited nodes are pruned to a list of size $R$ by pruning out points that are likely to become long edges of triangles, with a parameter $a$ that is used to control how aggressive the prune step is. 

1. **R**: the degree bound.
2. **L**: the beam width to use when building the graph.
3. **a**: the pruning parameter.

The memory flag is set to zero.

To build a Vamana graph on BIGANN-100K and save it to memory, use the following commandline:

```bash
cd vamana
make
./neighbors -R 32 -L 64 -a 1.2 -graph_outfile ../../data/sift/sift_learn_32_64 -data_type float -file_type vec -dist_func Euclidian -base_path ../../data/sift/sift_learn.fvecs
```

To load an already built graph and query it, use the following:
```bash
cd vamana
make
./neighbors -R 32 -L 64 -a 1.2 -graph_path ../../data/sift/sift_learn_32_64 -query_path ../../data/sift/sift_query.fvecs -gt_path ../../data/sift/sift-100K -res_path ../../workflows/vamana_res.csv -data_type float -file_type vec -dist_func Euclidian -base_path ../../data/sift/sift_learn.fvecs
```

To build, query, and save to memory, use the following:
```bash
cd vamana
make
./neighbors -R 32 -L 64 -a 1.2 -graph_outfile ../../data/sift/sift_learn_32_64 -query_path ../../data/sift/sift_query.fvecs -gt_path ../../data/sift/sift-100K -res_path ../../workflows/vamana_res.csv -data_type float -file_type vec -dist_func Euclidian -base_path ../../data/sift/sift_learn.fvecs
```

## HCNNG

HCNNG is an algorithm taken from [Hierarchical Clustering-Based Graphs for Large Scale Approximate Nearest Neighbor Search](https://www.researchgate.net/publication/334477189_Hierarchical_Clustering-Based_Graphs_for_Large_Scale_Approximate_Nearest_Neighbor_Search) by Munoz et al. and original implemented in [this repository](https://github.com/jalvarm/hcnng). Roughly, it builds a tree by recursively partitioning the data using random partitions until it reaches a leaf size of at most 1000 points, and then builds a bounded-degree MST with the points in each leaf. The edges from the MST are used as the edges in the graph. The algorithm repeats this process a total of $L$ times and merges the edges into the graph on each iteration. Its parameters are as follows:

1. **R**: the degree bound of the graph built by each individual cluster tree.
2. **L**: the number of cluster trees.
3. **a**: the leaf size of each cluster tree.

The memory flag is set to 1 since the total graph degree is $L*R$. A commandline with suggested parameters for HCNNG for the BIGANN-100K dataset is as follows:

```bash
cd HCNNG
make
./neighbors -R 3 -L 10 -a 1000 -memory_flag 1 -graph_outfile ../../data/sift/sift_learn_3_10 -query_path ../../data/sift/sift_query.fvecs -gt_path ../../data/sift/sift-100K -res_path ../../workflows/vamana_res.csv -data_type float -file_type vec -dist_func Euclidian -base_path ../../data/sift/sift_learn.fvecs
```

## pyNNDescent

[pyNNDescent](https://pynndescent.readthedocs.io/en/latest/) is an ANNS algorithm by Leland McInnes. It works based on the principle that in a k-nearest neighbor graph, a neighbor of a neighbor is likely to be a neighbor. It finds an approximate nearest neighbor graph by building some number of random clustering trees and calculating exhaustive nearest neighbors at the leaves. Then, it proceeds in rounds, connecting each vertex to the neighbors of each neighbors and keeping the $R$ closest neighbors on each round. After terminating, it prunes out long edges of triangles; in our version, we add a pruning parameter $d$ to control for a denser graph if desired.

1. **R**: the graph degree bound.
2. **a**: the number of cluster trees to use when initializing the graph.
3. **L**: the leaf size of the cluster trees.
4. **d**: the pruning parameter for the final pruning step.

The memory flag is set to 0 since the graph degree bound is $R$.

```bash
cd pyNNDescent
make
./neighbors -R 30 -L 100 -a 10 -d 1.2 -graph_outfile ../../data/sift/sift_learn_30 -query_path ../../data/sift/sift_query.fvecs -gt_path ../../data/sift/sift-100K -res_path ../../workflows/vamana_res.csv -data_type float -file_type vec -dist_func Euclidian -base_path ../../data/sift/sift_learn.fvecs
```

## Searching

Each graph is searched using a version of the greedy search/beam search algorithm described in [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf). It also incorporates the optimization suggested in [Pruned Bi-Directed K-Nearest Neighbor Graph for Proximity Search](https://link.springer.com/chapter/10.1007/978-3-319-46759-7_2) of pruning the search frontier when it includes points that are far away from the current $k$-nearest neighbor. Instead of taking in parameters specified by the user, the search routine tries a wide variety of parameter choices and reports those that maximize QPS for a given recall value. 