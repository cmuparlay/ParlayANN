import pandas as pd
import numpy as np
from datetime import datetime
import time
from collections import defaultdict

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


# Repeat the following steps many times:

# 1. Select a random point

# 2. Select two random filters associated with that point

# 3. Compute the average distance between this point and Filter 1

# 4. Compute the average distance between this point and Filter 2

# 5. Compute the average distance between this point and points in Filter 1 AND Filter 2

# 6. Store this data into a pandas dataframe along with as much other information as possible

# Create histogram of the average distances (using matplotlib or seaborn)

