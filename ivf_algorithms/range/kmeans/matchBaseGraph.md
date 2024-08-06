./rangeGraph -base_path $P/base.1B.u8bin.crop_nb_10000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_1M_100 -res_path bigann_ivf_range_graph_10M.csv -gt_path $P/range_gt_10M_10000 -data_type uint8 -num_clusters 10000 -dist_func Euclidian -k 10 -r 10000
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
KMeansClustering Time: 100.991
ClusterStats: num_points: 10000000 num_clusters: 10000 Min: 0 Max: 3795 Avg: 1000
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
beam search time: total: 0.0411
bidirect time: total: 0.0300
prune time: total: 0.0100
After build: 23.95, 32
Found 10000 centroids
Index built in 103.8 s
For all points: 
Sweeping once with regular beam search
For pointwise recall = 0.5496 and cumulative recall = 0.3626, QPS = 4.421e+05, n_probes = 1, average cmps = 1181
For pointwise recall = 0.7796 and cumulative recall = 0.587, QPS = 2.339e+05, n_probes = 2, average cmps = 2266
For pointwise recall = 0.8708 and cumulative recall = 0.7244, QPS = 1.61e+05, n_probes = 3, average cmps = 3346
For pointwise recall = 0.9189 and cumulative recall = 0.8074, QPS = 1.209e+05, n_probes = 4, average cmps = 4424
For pointwise recall = 0.9442 and cumulative recall = 0.8682, QPS = 9.695e+04, n_probes = 5, average cmps = 5495
For pointwise recall = 0.9829 and cumulative recall = 0.9517, QPS = 6.14e+04, n_probes = 8, average cmps = 8691
For pointwise recall = 0.9899 and cumulative recall = 0.9747, QPS = 4.985e+04, n_probes = 10, average cmps = 10808
For pointwise recall = 0.9955 and cumulative recall = 0.9976, QPS = 2.525e+04, n_probes = 20, average cmps = 21293
For pointwise recall = 0.9986 and cumulative recall = 0.9994, QPS = 1.699e+04, n_probes = 30, average cmps = 31668
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 1.279e+04, n_probes = 40, average cmps = 41984
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 1.031e+04, n_probes = 50, average cmps = 52248
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 5226, n_probes = 100, average cmps = 102696
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 2606, n_probes = 200, average cmps = 201924
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 1064, n_probes = 500, average cmps = 494752
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 540.4, n_probes = 1000, average cmps = 977223
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 269.7, n_probes = 2000, average cmps = 1936435
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 176.8, n_probes = 3000, average cmps = 2894187


Trying again with two-round search
For pointwise recall = 0.5496 and cumulative recall = 0.3626, QPS = 4.604e+05, n_probes = 1, average cmps = 1181
For pointwise recall = 0.7796 and cumulative recall = 0.587, QPS = 2.423e+05, n_probes = 2, average cmps = 2266
For pointwise recall = 0.8708 and cumulative recall = 0.7244, QPS = 1.642e+05, n_probes = 3, average cmps = 3346
For pointwise recall = 0.9189 and cumulative recall = 0.8074, QPS = 1.24e+05, n_probes = 4, average cmps = 4424
For pointwise recall = 0.9442 and cumulative recall = 0.8682, QPS = 9.803e+04, n_probes = 5, average cmps = 5495
For pointwise recall = 0.9829 and cumulative recall = 0.9517, QPS = 6.208e+04, n_probes = 8, average cmps = 8691
For pointwise recall = 0.9899 and cumulative recall = 0.9747, QPS = 4.989e+04, n_probes = 10, average cmps = 10808
For pointwise recall = 0.9955 and cumulative recall = 0.9976, QPS = 2.54e+04, n_probes = 20, average cmps = 21293
For pointwise recall = 0.9986 and cumulative recall = 0.9994, QPS = 1.703e+04, n_probes = 30, average cmps = 31668
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 1.286e+04, n_probes = 40, average cmps = 41984
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 1.034e+04, n_probes = 50, average cmps = 52248
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 5244, n_probes = 100, average cmps = 102696
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 2610, n_probes = 200, average cmps = 201924
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 1068, n_probes = 500, average cmps = 494752
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 540.4, n_probes = 1000, average cmps = 977223
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 269.6, n_probes = 2000, average cmps = 1936435
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 176.5, n_probes = 3000, average cmps = 2894187

For all 9590 points with zero results: 
Sweeping once with regular beam search
For pointwise recall = -nan and cumulative recall = -nan, QPS = 4.904e+05, n_probes = 1, average cmps = 1103
For pointwise recall = -nan and cumulative recall = -nan, QPS = 2.584e+05, n_probes = 2, average cmps = 2118
For pointwise recall = -nan and cumulative recall = -nan, QPS = 1.735e+05, n_probes = 3, average cmps = 3129
For pointwise recall = -nan and cumulative recall = -nan, QPS = 1.315e+05, n_probes = 4, average cmps = 4140
For pointwise recall = -nan and cumulative recall = -nan, QPS = 1.053e+05, n_probes = 5, average cmps = 5145
For pointwise recall = -nan and cumulative recall = -nan, QPS = 6.667e+04, n_probes = 8, average cmps = 8150
For pointwise recall = -nan and cumulative recall = -nan, QPS = 5.356e+04, n_probes = 10, average cmps = 10143
For pointwise recall = -nan and cumulative recall = -nan, QPS = 2.692e+04, n_probes = 20, average cmps = 20059
For pointwise recall = -nan and cumulative recall = -nan, QPS = 1.796e+04, n_probes = 30, average cmps = 29906
For pointwise recall = -nan and cumulative recall = -nan, QPS = 1.356e+04, n_probes = 40, average cmps = 39714
For pointwise recall = -nan and cumulative recall = -nan, QPS = 1.088e+04, n_probes = 50, average cmps = 49481
For pointwise recall = -nan and cumulative recall = -nan, QPS = 5502, n_probes = 100, average cmps = 97617
For pointwise recall = -nan and cumulative recall = -nan, QPS = 2728, n_probes = 200, average cmps = 192561
For pointwise recall = -nan and cumulative recall = -nan, QPS = 1113, n_probes = 500, average cmps = 473318
For pointwise recall = -nan and cumulative recall = -nan, QPS = 562.4, n_probes = 1000, average cmps = 936315
For pointwise recall = -nan and cumulative recall = -nan, QPS = 279.9, n_probes = 2000, average cmps = 1857161
For pointwise recall = -nan and cumulative recall = -nan, QPS = 183.8, n_probes = 3000, average cmps = 2776193

For all 209 points with 1 to 20 results
Sweeping once with regular beam search
For pointwise recall = 0.5785 and cumulative recall = 0.5889, QPS = 6.427e+06, n_probes = 1, average cmps = 34
For pointwise recall = 0.8128 and cumulative recall = 0.7963, QPS = 5.534e+06, n_probes = 2, average cmps = 66
For pointwise recall = 0.8939 and cumulative recall = 0.8839, QPS = 3.723e+06, n_probes = 3, average cmps = 98
For pointwise recall = 0.9352 and cumulative recall = 0.929, QPS = 2.738e+06, n_probes = 4, average cmps = 129
For pointwise recall = 0.9515 and cumulative recall = 0.953, QPS = 2.206e+06, n_probes = 5, average cmps = 160
For pointwise recall = 0.985 and cumulative recall = 0.9871, QPS = 1.228e+06, n_probes = 8, average cmps = 248
For pointwise recall = 0.9902 and cumulative recall = 0.9917, QPS = 9.684e+05, n_probes = 10, average cmps = 305
For pointwise recall = 0.9925 and cumulative recall = 0.9954, QPS = 5.523e+05, n_probes = 20, average cmps = 580
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 3.744e+05, n_probes = 30, average cmps = 838
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 2.842e+05, n_probes = 40, average cmps = 1090
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 2.346e+05, n_probes = 50, average cmps = 1333
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 1.341e+05, n_probes = 100, average cmps = 2496
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 7.128e+04, n_probes = 200, average cmps = 4656
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 3.085e+04, n_probes = 500, average cmps = 10803
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 1.577e+04, n_probes = 1000, average cmps = 20772
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 7726, n_probes = 2000, average cmps = 40436
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 4838, n_probes = 3000, average cmps = 60211


Trying again with two-round search
For pointwise recall = 0.5785 and cumulative recall = 0.5889, QPS = 6.627e+06, n_probes = 1, average cmps = 34
For pointwise recall = 0.8128 and cumulative recall = 0.7963, QPS = 2.458e+06, n_probes = 2, average cmps = 66
For pointwise recall = 0.8939 and cumulative recall = 0.8839, QPS = 3.862e+06, n_probes = 3, average cmps = 98
For pointwise recall = 0.9352 and cumulative recall = 0.929, QPS = 2.667e+06, n_probes = 4, average cmps = 129
For pointwise recall = 0.9515 and cumulative recall = 0.953, QPS = 2.006e+06, n_probes = 5, average cmps = 160
For pointwise recall = 0.985 and cumulative recall = 0.9871, QPS = 1.418e+06, n_probes = 8, average cmps = 248
For pointwise recall = 0.9902 and cumulative recall = 0.9917, QPS = 9.421e+05, n_probes = 10, average cmps = 305
For pointwise recall = 0.9925 and cumulative recall = 0.9954, QPS = 5.88e+05, n_probes = 20, average cmps = 580
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 3.895e+05, n_probes = 30, average cmps = 838
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 2.99e+05, n_probes = 40, average cmps = 1090
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 2.008e+05, n_probes = 50, average cmps = 1333
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 1.306e+05, n_probes = 100, average cmps = 2496
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 6.967e+04, n_probes = 200, average cmps = 4656
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 3.051e+04, n_probes = 500, average cmps = 10803
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 1.569e+04, n_probes = 1000, average cmps = 20772
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 7741, n_probes = 2000, average cmps = 40436
For pointwise recall = 0.9978 and cumulative recall = 0.9972, QPS = 4865, n_probes = 3000, average cmps = 60211

For all 201 points with greater than 20 results
Sweeping once with regular beam search
For pointwise recall = 0.5194 and cumulative recall = 0.361, QPS = 5.549e+06, n_probes = 1, average cmps = 42
For pointwise recall = 0.7451 and cumulative recall = 0.5855, QPS = 4.564e+06, n_probes = 2, average cmps = 81
For pointwise recall = 0.8469 and cumulative recall = 0.7233, QPS = 3.405e+06, n_probes = 3, average cmps = 118
For pointwise recall = 0.9018 and cumulative recall = 0.8065, QPS = 2.846e+06, n_probes = 4, average cmps = 154
For pointwise recall = 0.9365 and cumulative recall = 0.8676, QPS = 2.271e+06, n_probes = 5, average cmps = 189
For pointwise recall = 0.9807 and cumulative recall = 0.9514, QPS = 1.494e+06, n_probes = 8, average cmps = 292
For pointwise recall = 0.9895 and cumulative recall = 0.9746, QPS = 1.164e+06, n_probes = 10, average cmps = 358
For pointwise recall = 0.9987 and cumulative recall = 0.9976, QPS = 6.053e+05, n_probes = 20, average cmps = 652
For pointwise recall = 0.9995 and cumulative recall = 0.9994, QPS = 3.853e+05, n_probes = 30, average cmps = 923
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 3.159e+05, n_probes = 40, average cmps = 1180
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 2.508e+05, n_probes = 50, average cmps = 1432
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 1.474e+05, n_probes = 100, average cmps = 2582
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 7.367e+04, n_probes = 200, average cmps = 4706
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 3.146e+04, n_probes = 500, average cmps = 10629
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 1.614e+04, n_probes = 1000, average cmps = 20134
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 8075, n_probes = 2000, average cmps = 38838
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 4977, n_probes = 3000, average cmps = 57783


Trying again with two-round search
For pointwise recall = 0.5194 and cumulative recall = 0.361, QPS = 6.223e+06, n_probes = 1, average cmps = 42
For pointwise recall = 0.7451 and cumulative recall = 0.5855, QPS = 3.931e+06, n_probes = 2, average cmps = 81
For pointwise recall = 0.8469 and cumulative recall = 0.7233, QPS = 3.376e+06, n_probes = 3, average cmps = 118
For pointwise recall = 0.9018 and cumulative recall = 0.8065, QPS = 2.917e+06, n_probes = 4, average cmps = 154
For pointwise recall = 0.9365 and cumulative recall = 0.8676, QPS = 2.278e+06, n_probes = 5, average cmps = 189
For pointwise recall = 0.9807 and cumulative recall = 0.9514, QPS = 1.361e+06, n_probes = 8, average cmps = 292
For pointwise recall = 0.9895 and cumulative recall = 0.9746, QPS = 1.173e+06, n_probes = 10, average cmps = 358
For pointwise recall = 0.9987 and cumulative recall = 0.9976, QPS = 6.1e+05, n_probes = 20, average cmps = 652
For pointwise recall = 0.9995 and cumulative recall = 0.9994, QPS = 4.006e+05, n_probes = 30, average cmps = 923
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 3.128e+05, n_probes = 40, average cmps = 1180
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 2.509e+05, n_probes = 50, average cmps = 1432
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 1.338e+05, n_probes = 100, average cmps = 2582
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 7.356e+04, n_probes = 200, average cmps = 4706
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 3.148e+04, n_probes = 500, average cmps = 10629
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 1.605e+04, n_probes = 1000, average cmps = 20134
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 7912, n_probes = 2000, average cmps = 38838
For pointwise recall = 0.9995 and cumulative recall = 0.9996, QPS = 4936, n_probes = 3000, average cmps = 57783

For pointwise recall = 0.5496 and cumulative recall = 0.3626, QPS = 4.443e+05, n_probes = 1, average cmps = 1181
For pointwise recall = 0.7796 and cumulative recall = 0.587, QPS = 2.442e+05, n_probes = 2, average cmps = 2266
For pointwise recall = 0.8708 and cumulative recall = 0.7244, QPS = 1.604e+05, n_probes = 3, average cmps = 3346
For pointwise recall = 0.9189 and cumulative recall = 0.8074, QPS = 1.23e+05, n_probes = 4, average cmps = 4424
For pointwise recall = 0.9442 and cumulative recall = 0.8682, QPS = 9.847e+04, n_probes = 5, average cmps = 5495
For pointwise recall = 0.9899 and cumulative recall = 0.9747, QPS = 4.968e+04, n_probes = 10, average cmps = 10808
For pointwise recall = 0.9955 and cumulative recall = 0.9976, QPS = 2.507e+04, n_probes = 20, average cmps = 21293
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 1.025e+04, n_probes = 50, average cmps = 52248
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 5197, n_probes = 100, average cmps = 102696
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 2594, n_probes = 200, average cmps = 201924
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 1062, n_probes = 500, average cmps = 494752
For pointwise recall = 0.9986 and cumulative recall = 0.9996, QPS = 537.6, n_probes = 1000, average cmps = 977223
Saving index...
Parlay time: 545.2579


./rangeGraph -base_path $P/base.1B.u8bin.crop_nb_100000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_100M_100 -res_path bigann_ivf_range_graph_10M.csv -gt_path $P/range_gt_100M_10000 -data_type uint8 -num_clusters 100000 -dist_func Euclidian -k 10 -r 10000
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
KMeansClustering Time: 10189.9
ClusterStats: num_points: 100000000 num_clusters: 100000 Min: 0 Max: 5238 Avg: 1000
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
beam search time: total: 0.1327
bidirect time: total: 0.0476
prune time: total: 0.0850
After build: 26.93, 32
Found 100000 centroids
Index built in 1.022e+04 s
For all points: 
Sweeping once with regular beam search
For pointwise recall = 0.3572 and cumulative recall = 0.09316, QPS = 4.279e+05, n_probes = 1, average cmps = 1207
For pointwise recall = 0.5438 and cumulative recall = 0.1762, QPS = 2.207e+05, n_probes = 2, average cmps = 2320
For pointwise recall = 0.6515 and cumulative recall = 0.2441, QPS = 1.463e+05, n_probes = 3, average cmps = 3421
For pointwise recall = 0.7248 and cumulative recall = 0.3006, QPS = 1.12e+05, n_probes = 4, average cmps = 4514
For pointwise recall = 0.7689 and cumulative recall = 0.3444, QPS = 9.107e+04, n_probes = 5, average cmps = 5609
For pointwise recall = 0.8628 and cumulative recall = 0.4562, QPS = 5.613e+04, n_probes = 8, average cmps = 8871
For pointwise recall = 0.8965 and cumulative recall = 0.5208, QPS = 4.594e+04, n_probes = 10, average cmps = 11049
For pointwise recall = 0.9546 and cumulative recall = 0.7018, QPS = 2.307e+04, n_probes = 20, average cmps = 21825
For pointwise recall = 0.9762 and cumulative recall = 0.8086, QPS = 1.553e+04, n_probes = 30, average cmps = 32529
For pointwise recall = 0.9856 and cumulative recall = 0.8744, QPS = 1.178e+04, n_probes = 40, average cmps = 43195
For pointwise recall = 0.991 and cumulative recall = 0.9148, QPS = 9451, n_probes = 50, average cmps = 53800
For pointwise recall = 0.9981 and cumulative recall = 0.9862, QPS = 4795, n_probes = 100, average cmps = 106196
For pointwise recall = 0.9991 and cumulative recall = 0.9992, QPS = 2394, n_probes = 200, average cmps = 209338
For pointwise recall = 0.9991 and cumulative recall = 0.9996, QPS = 982.5, n_probes = 500, average cmps = 512970
For pointwise recall = 0.9991 and cumulative recall = 0.9996, QPS = 498.8, n_probes = 1000, average cmps = 1010030
For pointwise recall = 0.9991 and cumulative recall = 0.9996, QPS = 251.5, n_probes = 2000, average cmps = 1989415
For pointwise recall = 0.9991 and cumulative recall = 0.9996, QPS = 166, n_probes = 3000, average cmps = 2959675


Trying again with two-round search
For pointwise recall = 0.3572 and cumulative recall = 0.09316, QPS = 4.265e+05, n_probes = 1, average cmps = 1207
For pointwise recall = 0.5438 and cumulative recall = 0.1762, QPS = 2.241e+05, n_probes = 2, average cmps = 2320
For pointwise recall = 0.6515 and cumulative recall = 0.2441, QPS = 1.527e+05, n_probes = 3, average cmps = 3421
For pointwise recall = 0.7248 and cumulative recall = 0.3006, QPS = 1.134e+05, n_probes = 4, average cmps = 4514
For pointwise recall = 0.7689 and cumulative recall = 0.3444, QPS = 9.199e+04, n_probes = 5, average cmps = 5609
For pointwise recall = 0.8628 and cumulative recall = 0.4562, QPS = 5.78e+04, n_probes = 8, average cmps = 8871
For pointwise recall = 0.8965 and cumulative recall = 0.5208, QPS = 4.63e+04, n_probes = 10, average cmps = 11049
For pointwise recall = 0.9546 and cumulative recall = 0.7018, QPS = 2.353e+04, n_probes = 20, average cmps = 21825
For pointwise recall = 0.9762 and cumulative recall = 0.8086, QPS = 1.574e+04, n_probes = 30, average cmps = 32529
For pointwise recall = 0.9856 and cumulative recall = 0.8744, QPS = 1.184e+04, n_probes = 40, average cmps = 43195
For pointwise recall = 0.991 and cumulative recall = 0.9148, QPS = 9596, n_probes = 50, average cmps = 53800
For pointwise recall = 0.9981 and cumulative recall = 0.9862, QPS = 4855, n_probes = 100, average cmps = 106196
For pointwise recall = 0.9991 and cumulative recall = 0.9992, QPS = 2412, n_probes = 200, average cmps = 209338
For pointwise recall = 0.9991 and cumulative recall = 0.9996, QPS = 988.6, n_probes = 500, average cmps = 512970
For pointwise recall = 0.9991 and cumulative recall = 0.9996, QPS = 500.1, n_probes = 1000, average cmps = 1010030
For pointwise recall = 0.9991 and cumulative recall = 0.9996, QPS = 252, n_probes = 2000, average cmps = 1989415
For pointwise recall = 0.9991 and cumulative recall = 0.9996, QPS = 166.5, n_probes = 3000, average cmps = 2959675

For all 9413 points with zero results: 
Sweeping once with regular beam search
For pointwise recall = -nan and cumulative recall = -nan, QPS = 4.932e+05, n_probes = 1, average cmps = 1109
For pointwise recall = -nan and cumulative recall = -nan, QPS = 2.425e+05, n_probes = 2, average cmps = 2128
For pointwise recall = -nan and cumulative recall = -nan, QPS = 1.647e+05, n_probes = 3, average cmps = 3138
For pointwise recall = -nan and cumulative recall = -nan, QPS = 1.256e+05, n_probes = 4, average cmps = 4140
For pointwise recall = -nan and cumulative recall = -nan, QPS = 9.88e+04, n_probes = 5, average cmps = 5147
For pointwise recall = -nan and cumulative recall = -nan, QPS = 6.267e+04, n_probes = 8, average cmps = 8146
For pointwise recall = -nan and cumulative recall = -nan, QPS = 5.076e+04, n_probes = 10, average cmps = 10147
For pointwise recall = -nan and cumulative recall = -nan, QPS = 2.555e+04, n_probes = 20, average cmps = 20076
For pointwise recall = -nan and cumulative recall = -nan, QPS = 1.725e+04, n_probes = 30, average cmps = 29966
For pointwise recall = -nan and cumulative recall = -nan, QPS = 1.291e+04, n_probes = 40, average cmps = 39844
For pointwise recall = -nan and cumulative recall = -nan, QPS = 1.033e+04, n_probes = 50, average cmps = 49686
For pointwise recall = -nan and cumulative recall = -nan, QPS = 5232, n_probes = 100, average cmps = 98407
For pointwise recall = -nan and cumulative recall = -nan, QPS = 2587, n_probes = 200, average cmps = 194697
For pointwise recall = -nan and cumulative recall = -nan, QPS = 1057, n_probes = 500, average cmps = 479362
For pointwise recall = -nan and cumulative recall = -nan, QPS = 534.6, n_probes = 1000, average cmps = 946978
For pointwise recall = -nan and cumulative recall = -nan, QPS = 267.7, n_probes = 2000, average cmps = 1870552
For pointwise recall = -nan and cumulative recall = -nan, QPS = 176.4, n_probes = 3000, average cmps = 2786510

For all 263 points with 1 to 20 results
Sweeping once with regular beam search
For pointwise recall = 0.4679 and cumulative recall = 0.4605, QPS = 7.435e+06, n_probes = 1, average cmps = 38
For pointwise recall = 0.6697 and cumulative recall = 0.6737, QPS = 3.74e+06, n_probes = 2, average cmps = 74
For pointwise recall = 0.7722 and cumulative recall = 0.7839, QPS = 2.982e+06, n_probes = 3, average cmps = 110
For pointwise recall = 0.8412 and cumulative recall = 0.8475, QPS = 2.142e+06, n_probes = 4, average cmps = 145
For pointwise recall = 0.877 and cumulative recall = 0.8814, QPS = 1.662e+06, n_probes = 5, average cmps = 181
For pointwise recall = 0.9549 and cumulative recall = 0.9484, QPS = 1.094e+06, n_probes = 8, average cmps = 285
For pointwise recall = 0.9755 and cumulative recall = 0.9661, QPS = 9.397e+05, n_probes = 10, average cmps = 354
For pointwise recall = 0.9892 and cumulative recall = 0.9816, QPS = 4.856e+05, n_probes = 20, average cmps = 697
For pointwise recall = 0.9958 and cumulative recall = 0.9951, QPS = 3.226e+05, n_probes = 30, average cmps = 1029
For pointwise recall = 0.9969 and cumulative recall = 0.9972, QPS = 2.079e+05, n_probes = 40, average cmps = 1355
For pointwise recall = 0.9988 and cumulative recall = 0.9979, QPS = 2e+05, n_probes = 50, average cmps = 1675
For pointwise recall = 0.9988 and cumulative recall = 0.9979, QPS = 1.078e+05, n_probes = 100, average cmps = 3231
For pointwise recall = 0.9988 and cumulative recall = 0.9979, QPS = 5.74e+04, n_probes = 200, average cmps = 6209
For pointwise recall = 0.9988 and cumulative recall = 0.9979, QPS = 2.5e+04, n_probes = 500, average cmps = 14625
For pointwise recall = 0.9988 and cumulative recall = 0.9979, QPS = 1.282e+04, n_probes = 1000, average cmps = 27876
For pointwise recall = 0.9988 and cumulative recall = 0.9979, QPS = 6581, n_probes = 2000, average cmps = 53132
For pointwise recall = 0.9988 and cumulative recall = 0.9979, QPS = 4304, n_probes = 3000, average cmps = 77769


Trying again with two-round search
For pointwise recall = 0.4679 and cumulative recall = 0.4605, QPS = 6.423e+06, n_probes = 1, average cmps = 38
For pointwise recall = 0.6697 and cumulative recall = 0.6737, QPS = 3.398e+06, n_probes = 2, average cmps = 74
For pointwise recall = 0.7722 and cumulative recall = 0.7839, QPS = 2.621e+06, n_probes = 3, average cmps = 110
For pointwise recall = 0.8412 and cumulative recall = 0.8475, QPS = 2.15e+06, n_probes = 4, average cmps = 145
For pointwise recall = 0.877 and cumulative recall = 0.8814, QPS = 1.691e+06, n_probes = 5, average cmps = 181
For pointwise recall = 0.9549 and cumulative recall = 0.9484, QPS = 1.124e+06, n_probes = 8, average cmps = 285
For pointwise recall = 0.9755 and cumulative recall = 0.9661, QPS = 6.17e+05, n_probes = 10, average cmps = 354
For pointwise recall = 0.9892 and cumulative recall = 0.9816, QPS = 4.648e+05, n_probes = 20, average cmps = 697
For pointwise recall = 0.9958 and cumulative recall = 0.9951, QPS = 3.293e+05, n_probes = 30, average cmps = 1029
For pointwise recall = 0.9969 and cumulative recall = 0.9972, QPS = 2.464e+05, n_probes = 40, average cmps = 1355
For pointwise recall = 0.9988 and cumulative recall = 0.9979, QPS = 2.023e+05, n_probes = 50, average cmps = 1675
For pointwise recall = 0.9988 and cumulative recall = 0.9979, QPS = 1.108e+05, n_probes = 100, average cmps = 3231
For pointwise recall = 0.9988 and cumulative recall = 0.9979, QPS = 5.731e+04, n_probes = 200, average cmps = 6209
For pointwise recall = 0.9988 and cumulative recall = 0.9979, QPS = 2.49e+04, n_probes = 500, average cmps = 14625
For pointwise recall = 0.9988 and cumulative recall = 0.9979, QPS = 1.285e+04, n_probes = 1000, average cmps = 27876
For pointwise recall = 0.9988 and cumulative recall = 0.9979, QPS = 6662, n_probes = 2000, average cmps = 53132
For pointwise recall = 0.9988 and cumulative recall = 0.9979, QPS = 4323, n_probes = 3000, average cmps = 77769

For all 324 points with greater than 20 results
Sweeping once with regular beam search
For pointwise recall = 0.2674 and cumulative recall = 0.09282, QPS = 4.112e+06, n_probes = 1, average cmps = 59
For pointwise recall = 0.4415 and cumulative recall = 0.1757, QPS = 2.408e+06, n_probes = 2, average cmps = 116
For pointwise recall = 0.5536 and cumulative recall = 0.2436, QPS = 1.74e+06, n_probes = 3, average cmps = 172
For pointwise recall = 0.6304 and cumulative recall = 0.3001, QPS = 1.395e+06, n_probes = 4, average cmps = 228
For pointwise recall = 0.6811 and cumulative recall = 0.3439, QPS = 1.113e+06, n_probes = 5, average cmps = 281
For pointwise recall = 0.788 and cumulative recall = 0.4558, QPS = 7.024e+05, n_probes = 8, average cmps = 440
For pointwise recall = 0.8323 and cumulative recall = 0.5204, QPS = 5.789e+05, n_probes = 10, average cmps = 547
For pointwise recall = 0.9265 and cumulative recall = 0.7016, QPS = 2.928e+05, n_probes = 20, average cmps = 1052
For pointwise recall = 0.9603 and cumulative recall = 0.8084, QPS = 1.994e+05, n_probes = 30, average cmps = 1534
For pointwise recall = 0.9763 and cumulative recall = 0.8743, QPS = 1.676e+05, n_probes = 40, average cmps = 1995
For pointwise recall = 0.9846 and cumulative recall = 0.9148, QPS = 1.411e+05, n_probes = 50, average cmps = 2438
For pointwise recall = 0.9974 and cumulative recall = 0.9862, QPS = 7.613e+04, n_probes = 100, average cmps = 4557
For pointwise recall = 0.9993 and cumulative recall = 0.9992, QPS = 4.193e+04, n_probes = 200, average cmps = 8431
For pointwise recall = 0.9993 and cumulative recall = 0.9996, QPS = 1.823e+04, n_probes = 500, average cmps = 18982
For pointwise recall = 0.9993 and cumulative recall = 0.9996, QPS = 9908, n_probes = 1000, average cmps = 35175
For pointwise recall = 0.9993 and cumulative recall = 0.9996, QPS = 5072, n_probes = 2000, average cmps = 65730
For pointwise recall = 0.9993 and cumulative recall = 0.9996, QPS = 3391, n_probes = 3000, average cmps = 95396


Trying again with two-round search
For pointwise recall = 0.2674 and cumulative recall = 0.09282, QPS = 4.054e+06, n_probes = 1, average cmps = 59
For pointwise recall = 0.4415 and cumulative recall = 0.1757, QPS = 2.395e+06, n_probes = 2, average cmps = 116
For pointwise recall = 0.5536 and cumulative recall = 0.2436, QPS = 1.728e+06, n_probes = 3, average cmps = 172
For pointwise recall = 0.6304 and cumulative recall = 0.3001, QPS = 1.289e+06, n_probes = 4, average cmps = 228
For pointwise recall = 0.6811 and cumulative recall = 0.3439, QPS = 9.333e+05, n_probes = 5, average cmps = 281
For pointwise recall = 0.788 and cumulative recall = 0.4558, QPS = 7.358e+05, n_probes = 8, average cmps = 440
For pointwise recall = 0.8323 and cumulative recall = 0.5204, QPS = 5.514e+05, n_probes = 10, average cmps = 547
For pointwise recall = 0.9265 and cumulative recall = 0.7016, QPS = 2.824e+05, n_probes = 20, average cmps = 1052
For pointwise recall = 0.9603 and cumulative recall = 0.8084, QPS = 2.227e+05, n_probes = 30, average cmps = 1534
For pointwise recall = 0.9763 and cumulative recall = 0.8743, QPS = 1.73e+05, n_probes = 40, average cmps = 1995
For pointwise recall = 0.9846 and cumulative recall = 0.9148, QPS = 1.413e+05, n_probes = 50, average cmps = 2438
For pointwise recall = 0.9974 and cumulative recall = 0.9862, QPS = 7.542e+04, n_probes = 100, average cmps = 4557
For pointwise recall = 0.9993 and cumulative recall = 0.9992, QPS = 4.176e+04, n_probes = 200, average cmps = 8431
For pointwise recall = 0.9993 and cumulative recall = 0.9996, QPS = 1.832e+04, n_probes = 500, average cmps = 18982
For pointwise recall = 0.9993 and cumulative recall = 0.9996, QPS = 9712, n_probes = 1000, average cmps = 35175
For pointwise recall = 0.9993 and cumulative recall = 0.9996, QPS = 4988, n_probes = 2000, average cmps = 65730
For pointwise recall = 0.9993 and cumulative recall = 0.9996, QPS = 3340, n_probes = 3000, average cmps = 95396

For pointwise recall = 0.3572 and cumulative recall = 0.09316, QPS = 4.147e+05, n_probes = 1, average cmps = 1207
For pointwise recall = 0.5438 and cumulative recall = 0.1762, QPS = 2.239e+05, n_probes = 2, average cmps = 2320
For pointwise recall = 0.6515 and cumulative recall = 0.2441, QPS = 1.493e+05, n_probes = 3, average cmps = 3421
For pointwise recall = 0.7248 and cumulative recall = 0.3006, QPS = 1.131e+05, n_probes = 4, average cmps = 4514
For pointwise recall = 0.7689 and cumulative recall = 0.3444, QPS = 8.936e+04, n_probes = 5, average cmps = 5609
For pointwise recall = 0.8965 and cumulative recall = 0.5208, QPS = 4.646e+04, n_probes = 10, average cmps = 11049
For pointwise recall = 0.9546 and cumulative recall = 0.7018, QPS = 2.332e+04, n_probes = 20, average cmps = 21825
For pointwise recall = 0.991 and cumulative recall = 0.9148, QPS = 9488, n_probes = 50, average cmps = 53800
For pointwise recall = 0.9981 and cumulative recall = 0.9862, QPS = 4813, n_probes = 100, average cmps = 106196
For pointwise recall = 0.9991 and cumulative recall = 0.9992, QPS = 2400, n_probes = 200, average cmps = 209338
For pointwise recall = 0.9991 and cumulative recall = 0.9996, QPS = 983.3, n_probes = 500, average cmps = 512970
For pointwise recall = 0.9991 and cumulative recall = 0.9996, QPS = 498.5, n_probes = 1000, average cmps = 1010030
Saving index...
Parlay time: 10695.0668

