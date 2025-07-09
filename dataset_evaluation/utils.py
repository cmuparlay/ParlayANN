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

def string_to_list(s):
  s = s.strip().strip('[').strip(']').split(',')
  return [ss.strip() for ss in s]

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
    
def runtest(dataset_name, outfile, search_type, early_stop) :
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
    op += " -R " + str(ds["R"] )
    op += " -L " + str(ds["L"] )
    op += " -quantize_mode " + str(ds["quantize_mode"] )
    if early_stop:
        op += " -early_stop"
    op += " -search_mode " + search_type
    if os.path.exists(ds["graph"]):
        op += " -graph_path " + ds["graph"]
    else:
        op += " -graph_outfile " + ds["graph"]
    runstring(op, outfile)

def pareto_frontier(Xs, Ys, maxX = True):
    """
    Finds the Pareto frontier of a set of points.

    Parameters:
    Xs: Array of x-coordinates of the points.
    Ys: Array of y-coordinates of the points.
    maxX: If True, maximizes the x-coordinate; otherwise, minimizes it.

    Returns:
    x_pareto: Array of x-coordinates on the Pareto frontier.
    y_pareto: Array of y-coordinates on the Pareto frontier.
    """

    myList = sorted([(x, y) for (x, y) in zip(Xs, Ys)], reverse=maxX)
    x_pareto = [myList[0][0]]
    y_pareto = [myList[0][1]]
    for i in range(1, len(myList)):
        if maxX:
            if myList[i][1] > y_pareto[-1]:
                x_pareto.append(myList[i][0])
                y_pareto.append(myList[i][1])
        else:
            if myList[i][1] < y_pareto[-1]:
                x_pareto.append(myList[i][0])
                y_pareto.append(myList[i][1])

    return x_pareto, y_pareto

def export_legend(legend, filename="legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)