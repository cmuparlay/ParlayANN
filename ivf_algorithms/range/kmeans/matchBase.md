./range -base_path $P/base.1B.u8bin.crop_nb_100000000 -query_path $P/query.public.10K.u8bin -cluster_o
utfile $P/range_graph_100M_100 -res_path bigann_ivf_range_graph_10M.csv -gt_path $P/range_gt_100M_10000 -data_type uint8 -num_clusters 100000 -dist_func Eucli
dian -k 10 -r 10000
Detected 10000 points with num matches 1530449
Detected 100000000 points with dimension 128
Detected 10000 points with dimension 128
Building with 100000 clusters
Calculating clusters
KMeans run on: 20000000 many points to obtain: 100000 many clusters.
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
KMeansClustering Time: 10189.2
ClusterStats: num_points: 100000000 num_clusters: 100000 Min: 0 Max: 5105 Avg: 1000
Calculating centroids
Found 100000 centroids
Index built in 10216.8 s
For all points: 
Sweeping once with regular beam search
For pointwise recall = 0.387742 and cumulative recall = 0.0978451, QPS = 44188.2, n_probes = 1, average cmps = 101069
For pointwise recall = 0.561266 and cumulative recall = 0.175599, QPS = 44792, n_probes = 2, average cmps = 102142
For pointwise recall = 0.66662 and cumulative recall = 0.2459, QPS = 40916.9, n_probes = 3, average cmps = 103212
For pointwise recall = 0.73177 and cumulative recall = 0.300805, QPS = 38382.9, n_probes = 4, average cmps = 104285
For pointwise recall = 0.77516 and cumulative recall = 0.34755, QPS = 35427.7, n_probes = 5, average cmps = 105357
For pointwise recall = 0.855015 and cumulative recall = 0.460149, QPS = 29431.4, n_probes = 8, average cmps = 108562
For pointwise recall = 0.892933 and cumulative recall = 0.518185, QPS = 26125.1, n_probes = 10, average cmps = 110696
For pointwise recall = 0.956281 and cumulative recall = 0.705322, QPS = 17121.6, n_probes = 20, average cmps = 121300
For pointwise recall = 0.975261 and cumulative recall = 0.807951, QPS = 12511.8, n_probes = 30, average cmps = 131835
For pointwise recall = 0.984908 and cumulative recall = 0.873529, QPS = 9820.85, n_probes = 40, average cmps = 142312
For pointwise recall = 0.99065 and cumulative recall = 0.913539, QPS = 7985.29, n_probes = 50, average cmps = 152760
For pointwise recall = 0.997988 and cumulative recall = 0.986025, QPS = 4066.39, n_probes = 100, average cmps = 204546
For pointwise recall = 0.999075 and cumulative recall = 0.999208, QPS = 2013.15, n_probes = 200, average cmps = 306766
For pointwise recall = 0.999117 and cumulative recall = 0.999643, QPS = 746.198, n_probes = 500, average cmps = 608058
For pointwise recall = 0.999117 and cumulative recall = 0.999643, QPS = 285.214, n_probes = 1000, average cmps = 1102271
For pointwise recall = 0.999117 and cumulative recall = 0.999643, QPS = 94.182, n_probes = 2000, average cmps = 2077436
For pointwise recall = 0.999117 and cumulative recall = 0.999643, QPS = 47.8329, n_probes = 3000, average cmps = 3044411


Trying again with two-round search
For pointwise recall = 0.387742 and cumulative recall = 0.0978451, QPS = 47860.6, n_probes = 1, average cmps = 101069
For pointwise recall = 0.561266 and cumulative recall = 0.175599, QPS = 42383.7, n_probes = 2, average cmps = 102142
For pointwise recall = 0.66662 and cumulative recall = 0.2459, QPS = 40294.6, n_probes = 3, average cmps = 103212
For pointwise recall = 0.73177 and cumulative recall = 0.300805, QPS = 37459.5, n_probes = 4, average cmps = 104285
For pointwise recall = 0.77516 and cumulative recall = 0.34755, QPS = 35189.2, n_probes = 5, average cmps = 105357
For pointwise recall = 0.855015 and cumulative recall = 0.460149, QPS = 29560.3, n_probes = 8, average cmps = 108562
For pointwise recall = 0.892933 and cumulative recall = 0.518185, QPS = 26356.6, n_probes = 10, average cmps = 110696
For pointwise recall = 0.956281 and cumulative recall = 0.705322, QPS = 17036.6, n_probes = 20, average cmps = 121300
For pointwise recall = 0.975261 and cumulative recall = 0.807951, QPS = 12667.4, n_probes = 30, average cmps = 131835
For pointwise recall = 0.984908 and cumulative recall = 0.873529, QPS = 9897.71, n_probes = 40, average cmps = 142312
For pointwise recall = 0.99065 and cumulative recall = 0.913539, QPS = 8107.23, n_probes = 50, average cmps = 152760
For pointwise recall = 0.997988 and cumulative recall = 0.986025, QPS = 4088.33, n_probes = 100, average cmps = 204546
For pointwise recall = 0.999075 and cumulative recall = 0.999208, QPS = 2023.57, n_probes = 200, average cmps = 306766
For pointwise recall = 0.999117 and cumulative recall = 0.999643, QPS = 745.958, n_probes = 500, average cmps = 608058
For pointwise recall = 0.999117 and cumulative recall = 0.999643, QPS = 286.569, n_probes = 1000, average cmps = 1102271
For pointwise recall = 0.999117 and cumulative recall = 0.999643, QPS = 94.5963, n_probes = 2000, average cmps = 2077436
For pointwise recall = 0.999117 and cumulative recall = 0.999643, QPS = 47.7449, n_probes = 3000, average cmps = 3044411

For all 9413 points with zero results: 
Sweeping once with regular beam search
For pointwise recall = -nan and cumulative recall = -nan, QPS = 50747.8, n_probes = 1, average cmps = 95108
For pointwise recall = -nan and cumulative recall = -nan, QPS = 47555.4, n_probes = 2, average cmps = 96089
For pointwise recall = -nan and cumulative recall = -nan, QPS = 44468.4, n_probes = 3, average cmps = 97068
For pointwise recall = -nan and cumulative recall = -nan, QPS = 40569.9, n_probes = 4, average cmps = 98051
For pointwise recall = -nan and cumulative recall = -nan, QPS = 37811.9, n_probes = 5, average cmps = 99035
For pointwise recall = -nan and cumulative recall = -nan, QPS = 31592.2, n_probes = 8, average cmps = 101977
For pointwise recall = -nan and cumulative recall = -nan, QPS = 28013.4, n_probes = 10, average cmps = 103937
For pointwise recall = -nan and cumulative recall = -nan, QPS = 18418.6, n_probes = 20, average cmps = 113704
For pointwise recall = -nan and cumulative recall = -nan, QPS = 13685.4, n_probes = 30, average cmps = 123436
For pointwise recall = -nan and cumulative recall = -nan, QPS = 10701.1, n_probes = 40, average cmps = 133135
For pointwise recall = -nan and cumulative recall = -nan, QPS = 8770.78, n_probes = 50, average cmps = 142820
For pointwise recall = -nan and cumulative recall = -nan, QPS = 4416.78, n_probes = 100, average cmps = 190954
For pointwise recall = -nan and cumulative recall = -nan, QPS = 2182.1, n_probes = 200, average cmps = 286343
For pointwise recall = -nan and cumulative recall = -nan, QPS = 798.515, n_probes = 500, average cmps = 568743
For pointwise recall = -nan and cumulative recall = -nan, QPS = 303.02, n_probes = 1000, average cmps = 1033587
For pointwise recall = -nan and cumulative recall = -nan, QPS = 99.7019, n_probes = 2000, average cmps = 1952954
For pointwise recall = -nan and cumulative recall = -nan, QPS = 50.4971, n_probes = 3000, average cmps = 2865683

For all 263 points with 1 to 20 results
Sweeping once with regular beam search
For pointwise recall = 0.514893 and cumulative recall = 0.516243, QPS = 927730, n_probes = 1, average cmps = 2664
For pointwise recall = 0.701319 and cumulative recall = 0.70339, QPS = 944822, n_probes = 2, average cmps = 2700
For pointwise recall = 0.791598 and cumulative recall = 0.793785, QPS = 723798, n_probes = 3, average cmps = 2735
For pointwise recall = 0.843656 and cumulative recall = 0.850989, QPS = 741015, n_probes = 4, average cmps = 2770
For pointwise recall = 0.873803 and cumulative recall = 0.883475, QPS = 740686, n_probes = 5, average cmps = 2805
For pointwise recall = 0.927084 and cumulative recall = 0.935734, QPS = 613497, n_probes = 8, average cmps = 2908
For pointwise recall = 0.963096 and cumulative recall = 0.962571, QPS = 531293, n_probes = 10, average cmps = 2978
For pointwise recall = 0.992542 and cumulative recall = 0.989407, QPS = 336542, n_probes = 20, average cmps = 3318
For pointwise recall = 0.995074 and cumulative recall = 0.993644, QPS = 259720, n_probes = 30, average cmps = 3647
For pointwise recall = 0.996677 and cumulative recall = 0.996469, QPS = 209442, n_probes = 40, average cmps = 3969
For pointwise recall = 0.998578 and cumulative recall = 0.997175, QPS = 172423, n_probes = 50, average cmps = 4288
For pointwise recall = 0.998849 and cumulative recall = 0.997881, QPS = 97205.3, n_probes = 100, average cmps = 5840
For pointwise recall = 0.998849 and cumulative recall = 0.997881, QPS = 48326, n_probes = 200, average cmps = 8802
For pointwise recall = 0.998849 and cumulative recall = 0.997881, QPS = 17343.9, n_probes = 500, average cmps = 17183
For pointwise recall = 0.998849 and cumulative recall = 0.997881, QPS = 6664.88, n_probes = 1000, average cmps = 30391
For pointwise recall = 0.998849 and cumulative recall = 0.997881, QPS = 1993.98, n_probes = 2000, average cmps = 55637
For pointwise recall = 0.998849 and cumulative recall = 0.997881, QPS = 1018.93, n_probes = 3000, average cmps = 80218


Trying again with two-round search
For pointwise recall = 0.514893 and cumulative recall = 0.516243, QPS = 687191, n_probes = 1, average cmps = 2664
For pointwise recall = 0.701319 and cumulative recall = 0.70339, QPS = 972668, n_probes = 2, average cmps = 2700
For pointwise recall = 0.791598 and cumulative recall = 0.793785, QPS = 875810, n_probes = 3, average cmps = 2735
For pointwise recall = 0.843656 and cumulative recall = 0.850989, QPS = 766577, n_probes = 4, average cmps = 2770
For pointwise recall = 0.873803 and cumulative recall = 0.883475, QPS = 700084, n_probes = 5, average cmps = 2805
For pointwise recall = 0.927084 and cumulative recall = 0.935734, QPS = 577234, n_probes = 8, average cmps = 2908
For pointwise recall = 0.963096 and cumulative recall = 0.962571, QPS = 509970, n_probes = 10, average cmps = 2978
For pointwise recall = 0.992542 and cumulative recall = 0.989407, QPS = 339720, n_probes = 20, average cmps = 3318
For pointwise recall = 0.995074 and cumulative recall = 0.993644, QPS = 259155, n_probes = 30, average cmps = 3647
For pointwise recall = 0.996677 and cumulative recall = 0.996469, QPS = 203915, n_probes = 40, average cmps = 3969
For pointwise recall = 0.998578 and cumulative recall = 0.997175, QPS = 166581, n_probes = 50, average cmps = 4288
For pointwise recall = 0.998849 and cumulative recall = 0.997881, QPS = 79202.9, n_probes = 100, average cmps = 5840
For pointwise recall = 0.998849 and cumulative recall = 0.997881, QPS = 49511.3, n_probes = 200, average cmps = 8802
For pointwise recall = 0.998849 and cumulative recall = 0.997881, QPS = 17296.1, n_probes = 500, average cmps = 17183
For pointwise recall = 0.998849 and cumulative recall = 0.997881, QPS = 6604.13, n_probes = 1000, average cmps = 30391
For pointwise recall = 0.998849 and cumulative recall = 0.997881, QPS = 1926.72, n_probes = 2000, average cmps = 55637
For pointwise recall = 0.998849 and cumulative recall = 0.997881, QPS = 859.801, n_probes = 3000, average cmps = 80218

For all 324 points with greater than 20 results
Sweeping once with regular beam search
For pointwise recall = 0.28453 and cumulative recall = 0.0974577, QPS = 611060, n_probes = 1, average cmps = 3295
For pointwise recall = 0.44758 and cumulative recall = 0.175111, QPS = 701164, n_probes = 2, average cmps = 3352
For pointwise recall = 0.565172 and cumulative recall = 0.245393, QPS = 602809, n_probes = 3, average cmps = 3407
For pointwise recall = 0.640949 and cumulative recall = 0.300296, QPS = 563444, n_probes = 4, average cmps = 3463
For pointwise recall = 0.695088 and cumulative recall = 0.347053, QPS = 504617, n_probes = 5, average cmps = 3516
For pointwise recall = 0.796514 and cumulative recall = 0.459708, QPS = 387072, n_probes = 8, average cmps = 3676
For pointwise recall = 0.835981 and cumulative recall = 0.517774, QPS = 343430, n_probes = 10, average cmps = 3780
For pointwise recall = 0.926847 and cumulative recall = 0.705059, QPS = 235482, n_probes = 20, average cmps = 4277
For pointwise recall = 0.959179 and cumulative recall = 0.807779, QPS = 171151, n_probes = 30, average cmps = 4751
For pointwise recall = 0.975356 and cumulative recall = 0.873415, QPS = 140298, n_probes = 40, average cmps = 5207
For pointwise recall = 0.984215 and cumulative recall = 0.913462, QPS = 118366, n_probes = 50, average cmps = 5651
For pointwise recall = 0.997289 and cumulative recall = 0.986014, QPS = 67298.4, n_probes = 100, average cmps = 7752
For pointwise recall = 0.999258 and cumulative recall = 0.999209, QPS = 35782.8, n_probes = 200, average cmps = 11619
For pointwise recall = 0.999334 and cumulative recall = 0.999644, QPS = 12891.8, n_probes = 500, average cmps = 22130
For pointwise recall = 0.999334 and cumulative recall = 0.999644, QPS = 5017.33, n_probes = 1000, average cmps = 38293
For pointwise recall = 0.999334 and cumulative recall = 0.999644, QPS = 1534.86, n_probes = 2000, average cmps = 68844
For pointwise recall = 0.999334 and cumulative recall = 0.999644, QPS = 706.801, n_probes = 3000, average cmps = 98509


Trying again with two-round search
For pointwise recall = 0.28453 and cumulative recall = 0.0974577, QPS = 622394, n_probes = 1, average cmps = 3295
For pointwise recall = 0.44758 and cumulative recall = 0.175111, QPS = 663834, n_probes = 2, average cmps = 3352
For pointwise recall = 0.565172 and cumulative recall = 0.245393, QPS = 570256, n_probes = 3, average cmps = 3407
For pointwise recall = 0.640949 and cumulative recall = 0.300296, QPS = 540833, n_probes = 4, average cmps = 3463
For pointwise recall = 0.695088 and cumulative recall = 0.347053, QPS = 452468, n_probes = 5, average cmps = 3516
For pointwise recall = 0.796514 and cumulative recall = 0.459708, QPS = 381185, n_probes = 8, average cmps = 3676
For pointwise recall = 0.835981 and cumulative recall = 0.517774, QPS = 354786, n_probes = 10, average cmps = 3780
For pointwise recall = 0.926847 and cumulative recall = 0.705059, QPS = 218584, n_probes = 20, average cmps = 4277
For pointwise recall = 0.959179 and cumulative recall = 0.807779, QPS = 179022, n_probes = 30, average cmps = 4751
For pointwise recall = 0.975356 and cumulative recall = 0.873415, QPS = 142412, n_probes = 40, average cmps = 5207
For pointwise recall = 0.984215 and cumulative recall = 0.913462, QPS = 115823, n_probes = 50, average cmps = 5651
For pointwise recall = 0.997289 and cumulative recall = 0.986014, QPS = 67111.4, n_probes = 100, average cmps = 7752
For pointwise recall = 0.999258 and cumulative recall = 0.999209, QPS = 36608.6, n_probes = 200, average cmps = 11619
For pointwise recall = 0.999334 and cumulative recall = 0.999644, QPS = 12930.2, n_probes = 500, average cmps = 22130
For pointwise recall = 0.999334 and cumulative recall = 0.999644, QPS = 4257.12, n_probes = 1000, average cmps = 38293
For pointwise recall = 0.999334 and cumulative recall = 0.999644, QPS = 1557.03, n_probes = 2000, average cmps = 68844
For pointwise recall = 0.999334 and cumulative recall = 0.999644, QPS = 734.998, n_probes = 3000, average cmps = 98509

For pointwise recall = 0.387742 and cumulative recall = 0.0978451, QPS = 41499.3, n_probes = 1, average cmps = 101069
For pointwise recall = 0.561266 and cumulative recall = 0.175599, QPS = 39843.7, n_probes = 2, average cmps = 102142
For pointwise recall = 0.66662 and cumulative recall = 0.2459, QPS = 39490.9, n_probes = 3, average cmps = 103212
For pointwise recall = 0.73177 and cumulative recall = 0.300805, QPS = 35746.8, n_probes = 4, average cmps = 104285
For pointwise recall = 0.77516 and cumulative recall = 0.34755, QPS = 33830.1, n_probes = 5, average cmps = 105357
For pointwise recall = 0.892933 and cumulative recall = 0.518185, QPS = 26135.5, n_probes = 10, average cmps = 110696
For pointwise recall = 0.956281 and cumulative recall = 0.705322, QPS = 17283.2, n_probes = 20, average cmps = 121300
For pointwise recall = 0.99065 and cumulative recall = 0.913539, QPS = 8131.7, n_probes = 50, average cmps = 152760
For pointwise recall = 0.997988 and cumulative recall = 0.986025, QPS = 4123.84, n_probes = 100, average cmps = 204546
For pointwise recall = 0.999075 and cumulative recall = 0.999208, QPS = 2025.57, n_probes = 200, average cmps = 306766
For pointwise recall = 0.999117 and cumulative recall = 0.999643, QPS = 751.113, n_probes = 500, average cmps = 608058
For pointwise recall = 0.999117 and cumulative recall = 0.999643, QPS = 285.181, n_probes = 1000, average cmps = 1102271
Saving index...
Parlay time: 11471.3272