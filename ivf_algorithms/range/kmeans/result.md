./rangeGraph -base_path $P/base.1B.u8bin.crop_nb_10000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_1M_100 -res_path bigann_ivf_range_graph_10M.csv -gt_path $P/range_gt_10M_10000 -data_type uint8 -num_clusters 100 -dist_func Euclidian -k 10 -r 10000
Detected 10000 points with num matches 152958
Detected 10000000 points with dimension 128
Detected 10000 points with dimension 128
Building with 100 clusters
Calculating clusters
KMeans run on: 2000000 many points to obtain: 100 many clusters.
Beginning iteration 0...
Beginning iteration 1...
Beginning iteration 2...
Beginning iteration 3...
Beginning iteration 4...
Beginning iteration 5...
Beginning iteration 6...
Beginning iteration 7...
Beginning iteration 8...
Beginning iteration 9...
Beginning iteration 10...
Beginning iteration 11...
Beginning iteration 12...
Beginning iteration 13...
Beginning iteration 14...
Beginning iteration 15...
Beginning iteration 16...
Beginning iteration 17...
Beginning iteration 18...
Beginning iteration 19...
KMeansClustering Time: 2.59025
ClusterStats: num_points: 10000000 num_clusters: 100 Min: 68989 Max: 172903 Avg: 100000
Calculating centroids
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
beam search time: total: 0.0045
bidirect time: total: 0.0015
prune time: total: 0.0835
After build: 13.49, 29
Found 100 centroids
Index built in 5.546 s
For pointwise recall = 0.9094 and cumulative recall = 0.9376, QPS = 4372, n_probes = 1, average cmps = 102608
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 893.8, n_probes = 5, average cmps = 76341
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 457.6, n_probes = 10, average cmps = 144493
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 236.7, n_probes = 20, average cmps = 269171
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 98.54, n_probes = 50, average cmps = 179830
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 50.51, n_probes = 100, average cmps = 121707
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 50.62, n_probes = 200, average cmps = 121707
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 50.59, n_probes = 500, average cmps = 121707
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 50.57, n_probes = 1000, average cmps = 121707
Saving index...
Parlay time: 975.6869


./rangeGraph -base_path $P/base.1B.u8bin.crop_nb_10000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_1M_100 -res_path bigann_ivf_range_graph_10M.csv -gt_path $P/range_gt_10M_10000 -data_type uint8 -num_clusters 1000 -dist_func Euclid
ian -k 10 -r 10000
Detected 10000 points with num matches 152958
Detected 10000000 points with dimension 128
Detected 10000 points with dimension 128
Building with 1000 clusters
Calculating clusters
KMeans run on: 2000000 many points to obtain: 1000 many clusters.
Beginning iteration 0...
Beginning iteration 1...
Beginning iteration 2...
Beginning iteration 3...
Beginning iteration 4...
Beginning iteration 5...
Beginning iteration 6...
Beginning iteration 7...
Beginning iteration 8...
Beginning iteration 9...
Beginning iteration 10...
Beginning iteration 11...
Beginning iteration 12...
Beginning iteration 13...
Beginning iteration 14...
Beginning iteration 15...
Beginning iteration 16...
Beginning iteration 17...
Beginning iteration 18...
Beginning iteration 19...
KMeansClustering Time: 11.3064
ClusterStats: num_points: 10000000 num_clusters: 1000 Min: 1958 Max: 26646 Avg: 10000
Calculating centroids
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
beam search time: total: 0.0458
bidirect time: total: 0.0190
prune time: total: 0.0288
After build: 19.95, 32
Found 1000 centroids
Index built in 14.21 s
For pointwise recall = 0.788 and cumulative recall = 0.7695, QPS = 4.164e+04, n_probes = 1, average cmps = 10553
For pointwise recall = 0.9971 and cumulative recall = 0.9991, QPS = 8628, n_probes = 5, average cmps = 52265
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 4361, n_probes = 10, average cmps = 103611
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 2212, n_probes = 20, average cmps = 204831
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 917.9, n_probes = 50, average cmps = 73773
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 466.6, n_probes = 100, average cmps = 135970
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 240.9, n_probes = 200, average cmps = 251763
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 97.86, n_probes = 500, average cmps = 150740
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 48.28, n_probes = 1000, average cmps = 122600
Saving index...
Parlay time: 405.6553


./rangeGraph -base_path $P/base.1B.u8bin.crop_nb_10000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_1M_100 -res_path bigann_ivf_range_graph_10M.csv -gt_path $P/range_gt_10M_10000 -data_type uint8 -num_clusters 10000 -dist_func Eucli
dian -k 10 -r 10000
Detected 10000 points with num matches 152958
Detected 10000000 points with dimension 128
Detected 10000 points with dimension 128
Building with 10000 clusters
Calculating clusters
KMeans run on: 2000000 many points to obtain: 10000 many clusters.
Beginning iteration 0...
Beginning iteration 1...
Beginning iteration 2...
Beginning iteration 3...
Beginning iteration 4...
Beginning iteration 5...
Beginning iteration 6...
Beginning iteration 7...
Beginning iteration 8...
Beginning iteration 9...
Beginning iteration 10...
Beginning iteration 11...
Beginning iteration 12...
Beginning iteration 13...
Beginning iteration 14...
Beginning iteration 15...
Beginning iteration 16...
Beginning iteration 17...
Beginning iteration 18...
Beginning iteration 19...
KMeansClustering Time: 101.021
ClusterStats: num_points: 10000000 num_clusters: 10000 Min: 0 Max: 3352 Avg: 1000
Calculating centroids
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
beam search time: total: 0.0315
bidirect time: total: 0.0300
prune time: total: 0.0090
After build: 23.95, 32
Found 10000 centroids
Index built in 103.9 s
For pointwise recall = 0.5105 and cumulative recall = 0.3385, QPS = 4.645e+05, n_probes = 1, average cmps = 1161
For pointwise recall = 0.9381 and cumulative recall = 0.8378, QPS = 9.807e+04, n_probes = 5, average cmps = 5459
For pointwise recall = 0.987 and cumulative recall = 0.9645, QPS = 4.832e+04, n_probes = 10, average cmps = 10760
For pointwise recall = 0.9978 and cumulative recall = 0.9964, QPS = 2.495e+04, n_probes = 20, average cmps = 21223
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 1.022e+04, n_probes = 50, average cmps = 52088
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 5173, n_probes = 100, average cmps = 102463
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 2592, n_probes = 200, average cmps = 201587
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 1054, n_probes = 500, average cmps = 64903
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 533.9, n_probes = 1000, average cmps = 118312
Saving index...
Parlay time: 139.6791


For pointwise recall = 0.4989 and cumulative recall = 0.3465, QPS = 4.471e+05, n_probes = 1, average cmps = 1164
For pointwise recall = 0.7352 and cumulative recall = 0.5478, QPS = 2.207e+05, n_probes = 2, average cmps = 2256
For pointwise recall = 0.8496 and cumulative recall = 0.6966, QPS = 1.334e+05, n_probes = 3, average cmps = 3333
For pointwise recall = 0.9069 and cumulative recall = 0.7919, QPS = 1.189e+05, n_probes = 4, average cmps = 4405
For pointwise recall = 0.9434 and cumulative recall = 0.8517, QPS = 9.752e+04, n_probes = 5, average cmps = 5474
For pointwise recall = 0.987 and cumulative recall = 0.9647, QPS = 5e+04, n_probes = 10, average cmps = 10784
For pointwise recall = 0.9982 and cumulative recall = 0.9971, QPS = 2.505e+04, n_probes = 20, average cmps = 21267
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 1.029e+04, n_probes = 50, average cmps = 52172
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 5213, n_probes = 100, average cmps = 102618
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 2605, n_probes = 200, average cmps = 201861
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 1060, n_probes = 500, average cmps = 495086
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 536.8, n_probes = 1000, average cmps = 978903

./range -base_path $P/base.1B.u8bin.crop_nb_10000000 -query_path $P/query.public.10K.u8bin -cluster_ou
tfile $P/range_graph_1M_100 -res_path bigann_ivf_range_graph_10M.csv -gt_path $P/range_gt_10M_10000 -data_type uint8 -num_clusters 10000 -dist_func Euclidian 
-k 10 -r 10000

For pointwise recall = 0.5509 and cumulative recall = 0.344, QPS = 4.363e+05, n_probes = 1, average cmps = 1165
For pointwise recall = 0.9434 and cumulative recall = 0.8477, QPS = 9.846e+04, n_probes = 5, average cmps = 5466
For pointwise recall = 0.9899 and cumulative recall = 0.9643, QPS = 5.037e+04, n_probes = 10, average cmps = 10752
For pointwise recall = 0.9977 and cumulative recall = 0.9971, QPS = 2.546e+04, n_probes = 20, average cmps = 21217
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 1.037e+04, n_probes = 50, average cmps = 52070
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 5265, n_probes = 100, average cmps = 102400
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 2621, n_probes = 200, average cmps = 201446
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 1072, n_probes = 500, average cmps = 64373
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 536.3, n_probes = 1000, average cmps = 116950
Saving index...
Parlay time: 139.1565