# Range Search 

Range search is defined as finding every points within a radius among dataset, which is presented in our paper  [Range Retrieval with Graph-Based Indices](https://arxiv.org/abs/2502.13245). We implemented algorithms that are used in this paper.

## Sample commandline and parameters

Range groundtruth file should be computed before running these commands. These tools are provided in data_tools library. For further explanation, see [Data Tools](https://cmuparlay.github.io/ParlayANN/data_tools)

To run a range search on sift run:
```
cd rangeSearch/vamanaRange
R=../../data/sift
make
./range  -alpha 1.15 -R 64 -L 128 -r 10000 -base_path $R/sift_learn.fbin -data_type uint8 -dist_func Euclidian -query_path $R/sift_query.fbin  -gt_path $R/range_gt_1M_10000 -search_mode beamSearch -early_stop -graph_path $R/graph1M  -early_stopping_radius 30000
```

All other parameters are same as in  [Algorithms](https://cmuparlay.github.io/ParlayANN/algorithms). Here we add descriptions for parameters that are new.

1. **-r**(`double`): radius that we want to search.
2. **-search_mode**(`string`): The search mode to use can be specified. Possible options are ['doubling', 'greedy', 'beam'], corresponding to the three algorithms introduced in our paper. The default option is beam search.
3. **-early_stop**(optional): Flag for early stopping. With this flag on, range search would stop early based on early stopping radius.
4. **-early_stopping_radius**(`double`): Radius for early stopping. Typically larger than the range search radius.
