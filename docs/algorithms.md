# Algorithms

The algorithms in this folder share a common main file and thus a common commandline interface. The commandline interface allows the user to build an ANNS graph and write it to an outfile, load a graph and search it, or build and search in one shot. It contains several "generic" parameters that can be repurposed for a new benchmark. In the following examples, we provide instructions for building indices using bash. The instructions assume that the user has downloaded, converted, and built groundtruth for the 100K slice of the BIGANN dataset, as shown in the quickstart instructions. If you want to use range searching, we also provide instructions for computing range groundtruth in `data_tools.md`.

### Universal Parameters

#### Parameters for building:
1. **-graph_outfile** (optional): if graph is not already built, path the graph is written to. This is optional; if not provided, the graph will be built and will print timing and statistics before terminating.
2. **-data_type**: type of the base and query vectors. Currently "float", "int8", and "uint8" are supported.
3. **-dist_func**: the distance function to use when calculating nearest neighbors. Currently Euclidian distance ("euclidian") and maximum inner product search ("mips") are supported.
4. **-base_path**: path to the base file. We only work with files in the .bin format; for your convenience, a converter from the popular .vecs format has been provided in the data tools folder.

#### Parameters for searching:

1. **-gt_path**: path to the ground truth, in .ibin format.
2. **-graph_path** (optional): path to the ANNS graph in the case of using an already built graph.
3. **-query_path**: path to the queries in .bin format.
4. **-res_path** (optional): path where a CSV file of results can be written (it is written to in append form, so it can be used to collect results of multiple runs).
5. **-k** (`long`): the number of nearest neighbors to search for.


### Algorithms

Next we provide some descriptions and example commandline arguments for each algorithm in the implementation.

## Vamana (DiskANN)

Vamana, also known as DiskANN, is an algorithm introduced in [DiskANN: Fast Accurate Billion-point Nearest
Neighbor Search on a Single Node](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf) by Subramanya et al., with original code in the [DiskANN repo](https://github.com/microsoft/DiskANN). It builds a graph incrementally, and its insert procedure does a variant on greedy search or beam search with a frontier size $L$ on the existing graph and uses the nodes visited during the search as edge candidates. The visited nodes are pruned to a list of size $R$ by pruning out points that are likely to become long edges of triangles, with a parameter $a$ that is used to control how aggressive the prune step is. 

1. **R** (`long`): the degree bound.
2. **L** (`long`): the beam width to use when building the graph.
3. **alpha** (`double`): the pruning parameter.
4. **two_pass** (`bool`): optional argument that allows the user to build the graph with two passes or just one (two passes approximately doubles the build time, but provides higher accuracy).

To build a Vamana graph on BIGANN-100K and save it to memory, use the following commandline:

```bash
cd vamana
make
./neighbors -R 32 -L 64 -alpha 1.2 -graph_outfile ../../data/sift/sift_learn_32_64 -data_type float -dist_func Euclidian -base_path ../../data/sift/sift_learn.fbin
```

To load an already built graph and query it, use the following:
```bash
cd vamana
make
./neighbors -R 32 -L 64 -alpha 1.2 -graph_path ../../data/sift/sift_learn_32_64 -query_path ../../data/sift/sift_query.fbin -gt_path ../../data/sift/sift-100K -res_path ../../data/vamana_res.csv -data_type float  -dist_func Euclidian -base_path ../../data/sift/sift_learn.fbin
```

To build, query, and save to memory, use the following:
```bash
cd vamana
make
./neighbors -R 32 -L 64 -alpha 1.2 -graph_outfile ../../data/sift/sift_learn_32_64 -query_path ../../data/sift/sift_query.fbin -gt_path ../../data/sift/sift-100K -res_path ../../data/vamana_res.csv -data_type float -dist_func Euclidian -base_path ../../data/sift/sift_learn.fbin
```

To execute range search using Vamana, use the following commandline. Note that range searching currently does not support exporting data to a CSV file: 

```bash
cd ../rangeSearch/vamanaRange
make
./range -R 32 -L 64 -alpha 1.2 -graph_outfile ../../data/sift/sift_learn_32_64 -query_path ../../data/sift/sift_query.fbin -gt_path ../../data/sift/sift-100K-range -data_type float -dist_func Euclidian -base_path ../../data/sift/sift_learn.fbin
```

## HNSW

HNSW is an algorithm proposed in [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://dl.acm.org/doi/10.1109/TPAMI.2018.2889473) by Yu et al., of which an implementation is available at [hnswlib](https://github.com/nmslib/hnswlib) and is maintained by the paper authors. The HNSW incrementally builds a hierarchical structure consisting of multiple layers, where each layer is a proximity graphs with the Navigable Small World (NSW) property. The lower layers are always the supersets of the upper ones, and the bottom layer contains all the base points. In the process of constructions, each point is randomly assigned with a height in a logarithmic distribution and repeatedly inserted into all the layers below. As the two points incident to an edge in higher layers has longer distance, the hierarchical structures allows to quickly approach the query point at high layers first and then do fine-grained search at low layers.
Its parameters are as follows:

1. **m** (`long`): the degree bound. Typically between 16 and 64. The graph at the bottom layer (layer0) has the degree bound of $2m$ while graphs at upper layers have degree bound of $m$.
2. **efc** (`long`): the beam width to use when building the graph. Should be set at least $2.5m$, and up to 500.
3. **alpha** (`double`): the pruning parameter. Should be set between 1.0 and 1.15 for similarity measures that are not metrics (e.g. maximum inner product), and between 0.8 and 1.0 for metric spaces. 
4. **ml** (`double`): optional argument to control the number of layers (height). Increasing $ml$ results in more layers which increases the build time but potentially improve the query performance; however, improper settings of $ml$ (too high or too low) can incur much work of query thus impacting the query performance. It should be set around $1/log~m$.

A commandline with suggested parameters for HNSW for the BIGANN-100K dataset is as follows:
```bash
cd HNSW
make
./neighbors -m 20 -efc 50 -alpha 0.9 -ml 0.34 -graph_outfile ../../data/sift/sift_learn_20_50_034 -query_path ../../data/sift/sift_query.fbin -gt_path ../../data/sift/sift-100K -res_path ../../data/hnsw_res.csv -data_type float -dist_func Euclidian -base_path ../../data/sift/sift_learn.fbin
```

## HCNNG

HCNNG is an algorithm taken from [Hierarchical Clustering-Based Graphs for Large Scale Approximate Nearest Neighbor Search](https://www.researchgate.net/publication/334477189_Hierarchical_Clustering-Based_Graphs_for_Large_Scale_Approximate_Nearest_Neighbor_Search) by Munoz et al. and original implemented in [this repository](https://github.com/jalvarm/hcnng). Roughly, it builds a tree by recursively partitioning the data using random partitions until it reaches a leaf size of at most 1000 points, and then builds a bounded-degree MST with the points in each leaf. The edges from the MST are used as the edges in the graph. The algorithm repeats this process a total of $L$ times and merges the edges into the graph on each iteration. Its parameters are as follows:

1. **mst_deg** (`long`): the degree bound of the graph built by each individual cluster tree.
2. **num_clusters** (`long`): the number of cluster trees.
3. **cluster_size** (`long`): the leaf size of each cluster tree.

A commandline with suggested parameters for HCNNG for the BIGANN-100K dataset is as follows:

```bash
cd HCNNG
make
./neighbors -cluster_size 1000 -mst_deg 3 -num_clusters 30  -graph_outfile ../../data/sift/sift_learn_3_10 -query_path ../../data/sift/sift_query.fbin -gt_path ../../data/sift/sift-100K -res_path ../../data/hcnng_res.csv -data_type float -dist_func Euclidian -base_path ../../data/sift/sift_learn.fbin
```

## pyNNDescent

[pyNNDescent](https://pynndescent.readthedocs.io/en/latest/) is an ANNS algorithm by Leland McInnes. It works based on the principle that in a k-nearest neighbor graph, a neighbor of a neighbor is likely to be a neighbor. It finds an approximate nearest neighbor graph by building some number of random clustering trees and calculating exhaustive nearest neighbors at the leaves. Then, it proceeds in rounds, connecting each vertex to the neighbors of each neighbors and keeping the $R$ closest neighbors on each round. After terminating, it prunes out long edges of triangles; in our version, we add a pruning parameter $d$ to control for a denser graph if desired.

1. **R** (`long`): the graph degree bound.
2. **num_clusters** (`long`): the number of cluster trees to use when initializing the graph.
3. **cluster_size** (`long`): the leaf size of the cluster trees.
4. **alpha** (`double`): the pruning parameter for the final pruning step.
5. **delta** (`double`): the early stopping parameter for the nnDescent process.


```bash
cd pyNNDescent
make
./neighbors -R 40 -cluster_size 100 -num_clusters 10 -alpha 1.2 -delta 0.05 -graph_outfile ../../data/sift/sift_learn_30 -query_path ../../data/sift/sift_query.fbin -gt_path ../../data/sift/sift-100K -res_path ../../data/pynn_res.csv -data_type float -dist_func Euclidian -base_path ../../data/sift/sift_learn.fbin
```

## Searching

Each graph is searched using a version of the greedy search/beam search algorithm described in [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf). It also incorporates the optimization suggested in [Pruned Bi-Directed K-Nearest Neighbor Graph for Proximity Search](https://link.springer.com/chapter/10.1007/978-3-319-46759-7_2) of pruning the search frontier when it includes points that are far away from the current $k$-nearest neighbor. Instead of taking in parameters specified by the user, the search routine tries a wide variety of parameter choices and reports those that maximize QPS for a given recall value. The search parameters (see `types.h` in the utils folder) can be tuned if you are developing your own algorithm and are as follows:

1. **Q** (`long`): the beam width. Must be set at least $k$. Controls the number of candidate neighbors retained at any point in the search and is for the most part the chief determinant of accuracy and speed of the search. 
2. **k** (`long`): number of nearest neighbors to search for. 
3. **cut** (`double`): controls pruning the frontier of points that are far away from the current $k$ nearest neighbors. Used only for distance functions that are true metrics (as opposed to similarities that may not obey the triangle inequality, etc.)
4. **visited limit** (`long`): controls the maximum number of vertices visited during the beam search. Used for low accuracy searches; set to the number of vertices in the graph if you don't want any limit.
5. **degree limit** (`long`): controls the maximum number of out-neighbors read when visiting a vertex. Also useful for low accuracy searches. Note that if the out-neighbors are not sorted in order of distance, it does not make sense to use this parameter. 


