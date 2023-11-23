from random import randint
import pandas as pd
import numpy as np
from datetime import datetime
import time
from collections import defaultdict
import matplotlib.pyplot as plt

import os
import _ParlayANNpy as pann
import numpy as np
import wrapper as wp
from scipy.sparse import csr_matrix

FERN_DATA_DIR = "/ssd1/anndata/bigann/"
AWARE_DATA_DIR = "/ssd1/data/bigann/"

DATA_DIR = FERN_DATA_DIR
POINTS_PATH = DATA_DIR + "data/yfcc100M/base.10M.u8bin.crop_nb_10000000"
FILTER_PATH = DATA_DIR + 'data/yfcc100M/base.metadata.10M.spmat'

ds = wp.FilteredDataset(POINTS_PATH, FILTER_PATH)

samples = 2
# try multiprocessing - spawn 192 processes, each one samples a point

columns = ['Point', 'Filter1 Avg Dist', 'Filter2 Avg Dist', 'Join Avg Dist', 'Filter 1', 'Filter 2']
df = pd.DataFrame(index=range(samples), columns=columns, dtype=float)


# Repeat the following steps many times:
for i in range(samples):
    # 1. Select a random point (try the query set after the general dataset)
    point = randint(0, ds.size()-1)
    filters = ds.get_point_filters(point)
    while len(filters) < 2:
        point = randint(0, ds.size()-1) 
        filters = ds.get_point_filters(point)

    # 2. Select two random filters associated with that point
    filter1 = randint(0, len(filters) - 1)
    filter2 = randint(0, len(filters) - 1)
    while filter1 == filter2:
        filter2 = randint(0, len(filters) - 1)
    filter1 = filters[filter1]
    filter2 = filters[filter2]

    # 3. Compute the average distance between this point and Filter 1
    filter1_points = ds.get_filter_points(filter1)
    avg_dist1 = 0
    for p in filter1_points:
        if p != point:
            avg_dist1 += ds.distance(point, p)
    avg_dist1 /= (len(filter1_points)-1)
    print(avg_dist1)

    # 4. Compute the average distance between this point and Filter 2
    filter2_points = ds.get_filter_points(filter2)
    avg_dist2 = 0
    for p in filter2_points:
        if p != point:
            avg_dist2 += ds.distance(point, p)
    avg_dist2 /= (len(filter2_points)-1)
    print(avg_dist2)

    # 5. Compute the average distance between this point and points in Filter 1 AND Filter 2
    join_filters_points = ds.get_filter_intersection(filter1, filter2)
    avg_join_dist = 0
    for p in join_filters_points:
        if p != point:
            avg_join_dist += ds.distance(p, point)
    if len(join_filters_points) > 1:
        avg_join_dist /= (len(join_filters_points)-1)
    print(avg_join_dist)

    # 6. Store this data into a pandas dataframe along with as much other information as possible
    df.at[i, 'Point'] = point
    df.at[i, 'Filter1 Avg Dist'] = avg_dist1
    df.at[i, 'Filter2 Avg Dist'] = avg_dist2
    df.at[i, 'Join Avg Dist'] = avg_join_dist
    df.at[i, 'Filter 1'] = filter1
    df.at[i, 'Filter 2'] = filter2
# Create histogram of the average distances (using matplotlib or seaborn)
df['Point'] = df['Point'].astype(int)
df['Filter 1'] = df['Filter 1'].astype(int)
df['Filter 2'] = df['Filter 2'].astype(int)
print(df)

x = np.arange(1, samples+1)

y = list(df['Filter1 Avg Dist'])
z = list(df['Filter2 Avg Dist'])
k = list(df['Join Avg Dist'])

print(y,z,k)

ax = plt.subplot(111)
w = 0.3
ax.bar(x-w, y, width=w, color='b', align='center')
ax.bar(x, z, width=w, color='g', align='center')
ax.bar(x+w, k, width=w, color='r', align='center')
ax.xaxis_date()
ax.autoscale(tight=True)

plt.show()
