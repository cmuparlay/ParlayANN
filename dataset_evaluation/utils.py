import argparse
import os
import sys
import multiprocessing
import math 

import csv
import matplotlib as mpl
# mpl.use('Agg')
mpl.rcParams['grid.linestyle'] = ":"
mpl.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import statistics as st
from dataset_info import mk, dsinfo, data_options

already_ran = set()

def runstring(op, outfile):
    if op in already_ran:
        return
    already_ran.add(op)
    os.system("echo \"" + op + "\"")
    os.system("echo \"" + op + "\" >> " + outfile)
    x = os.system(op + " >> " + outfile)
    if (x) :
        if (os.WEXITSTATUS(x) == 0) : raise NameError("  aborted: " + op)
        os.system("echo Failed")
    
def runtest(dataset_name, outfile) :
    ds = data_options[dataset_name]
    op = "./../rangeSearch/vamanaRange/range"
    op += " -base_path " + ds["base"] 
    op += " -gt_path " + ds["gt"] 
    op += " -query_path " + ds["query"] 
    op += " -data_type " + ds["data_type"] 
    op += " -dist_func " + ds["dist_fn"] 
    op += " -r " + str(ds["radius"] )
    op += " -early_stopping_radius " + str(ds["esr"] )
    op += " -alpha " + str(ds["alpha"] )
    op += " -R " + str(64)
    op += " -L " + str(128)
    runstring(op, outfile)