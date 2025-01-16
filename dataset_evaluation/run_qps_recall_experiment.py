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



parser = argparse.ArgumentParser()
parser.add_argument("-groups", nargs='+', help="specify which groups to plot recall for: all, zeros, onetwos, or threeplus")
parser.add_argument("-datasets", help="dataset list")
parser.add_argument("-g","--graphs_only", help="graphs only",action="store_true")
parser.add_argument("-graph_name", help="graphs name")

args = parser.parse_args()
print("datasets " + args.datasets)
print("groups:", args.groups)


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

def string_to_list(s):
  s = s.strip().strip('[').strip(']').split(',')
  return [ss.strip() for ss in s]

pt_groupings = ["All", "Zeros", "Onetwos", "Threeplus"]
search_types = ["Beam Search", "Greedy Search", "Early Stopping"]
recall_types = ["Pointwise Recall", "Cumulative Recall"]

valid_groups = []
for g in pt_groupings:
  for s in search_types:
    for r in recall_types:
      valid_groups.append(g + ", " + s + ", " + r)

groups = args.groups[0]
groups = eval(groups)


for group in groups:
  if group not in valid_groups:
    print("Error: group", group, "not valid")
    exit(1)

if '[' in args.datasets:
  datasets = string_to_list(args.datasets)
else:
  print('invalid argument')
  exit(1)



if not args.graphs_only:
  for dataset_name in datasets:
      directory = "qps_recall_results/"
      os.makedirs(directory, exist_ok=True)
      outFile = directory + "/" + str(dataset_name) + ".txt"
      # clear output file
      os.system("echo \"\" > " + outFile)
      runtest(dataset_name, outFile)



'''
Results file is understood to include the QPS/Recall curves for *four* different groups
and *three* different algorithms
Groups are: all, zeros, onetwos, threepluses
Algorithms are: beam search, beam search + range search, beam search + range search + early stopping
Also includes *two* different recall choices: cumulative and pointwise
Returns list of recalls and list of QPS
'''
def readResultsFile(dataset_name, group):
    result_filename = "qps_recall_results/" + str(dataset_name) + ".txt"
    file = open(result_filename, 'r')
    categories = string_to_list(group)
    if categories[0] == "Zeros": # recall is all recall if group is zeros
      qps_line = categories[0] + ", " + categories[1] + ", QPS"
      recall_line = "All, " + categories[1] + ", " + categories[2]
    else:
      qps_line = categories[0] + ", " + categories[1] + ", QPS"
      recall_line = group 
    for line in file.readlines():
      if line.find(recall_line) != -1:
          recall = line.split(': ')[1]
          recall = eval(recall)
      if line.find(qps_line) != -1:
          qps = line.split(': ')[1]
          qps = eval(qps)
    return qps, recall

result_data = {}

for dataset_name in datasets:
    result_data[dataset_name] = {}
    for group in groups:
        qps, recall = readResultsFile(dataset_name, group)
        result_data[dataset_name][group + ", QPS"] = qps
        result_data[dataset_name][group] = recall

# print(result_data)

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

linestyles = {"Beam Search": "solid",
              "Greedy Search": "dashed",
              "Early Stopping": "dotted",
            }

def plot_qps_recall_graph(result_data, graph_name, paper_ver=False):
  os.makedirs("graphs/qps_recall", exist_ok=True)

  outputFile = 'graphs/qps_recall/' + graph_name.replace('.', '') + '.pdf'
  mpl.rcParams.update({'font.size': 25})

  print(outputFile)
  xmin = 1.0
  xmax = 0


  fig, axs = plt.subplots()
  # fig = plt.figure()
  
  rects = {}
  
  for dataset_name, dataset_data in result_data.items():
    alginfo = dsinfo[dataset_name]
    
    for group in groups:

      algorithm = string_to_list(group)[1]
      recall_type = string_to_list(group)[2]
      linestyle = linestyles[algorithm]
      recall_raw = dataset_data[group]
      qps_raw = dataset_data[group + ", QPS"]
      qps, recall = pareto_frontier(qps_raw, recall_raw, True)
      xmin = min(min(recall), xmin)
      xmax = max(max(recall), xmax)
      if "alpha" in alginfo:
        alpha = alginfo["alpha"]
      else:
        alpha = 1.0
      rects[dataset_name] = axs.plot(recall, qps,
        alpha=alpha,
        color=alginfo["color"],
        linewidth=3.0,
        linestyle=linestyle,
        marker=alginfo["marker"],
        markersize=14,
        label=dataset_name+", " + algorithm)


  # axs.set_xscale('log')

  alpha = 4.0

  def fun(x):
      return 1 - (1 - x) ** (1 / alpha)

  def inv_fun(x):
      return 1 - (1 - x) ** alpha

  axs.set_xscale("function", functions=(fun, inv_fun))
  plt.xlim(xmin=xmin-.02)
  plt.xlim(xmax=xmax)
  if alpha <= 3:
      ticks = [inv_fun(x) for x in np.arange(0, 1.2, 0.2)]
      plt.xticks(ticks)
  if alpha > 3:
      from matplotlib import ticker

      axs.xaxis.set_major_formatter(ticker.LogitFormatter())
      # plt.xticks(ticker.LogitLocator().tick_values(xmin, xmax))
      xticks = [0, .5, .9, .99, .999, .9999, 1.0]
      real_xticks=[]
      for i in range(len(xticks)):
        if (xticks[i] < xmin) & (i < len(xticks)-1):
          if (xticks[i+1] < xmin):
            continue
        elif (xticks[i] > xmax) & (i > 0):
          if (xticks[i-1] > xmax):
            continue
        real_xticks.append(xticks[i])
      plt.xticks(real_xticks)

  axs.set_yscale('log')
  plt.xlabel(recall_type, fontsize=14)
  plt.xticks(rotation=45)


  plt.ylabel('QPS', fontsize=14)
  # axs.tick_params(axis='x', labelsize=14) 
  plt.tick_params(axis='both', which='both', labelsize=14)


  legend_x = 1
  legend_y = 0.5 
  plt.grid()
  if not paper_ver:
    plt.title(graph_name)
    plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y))
  plt.savefig(outputFile, bbox_inches='tight')

  if paper_ver:
    nc = 8
    if len(algs) == 2:
      ncol = 1
    elif len(algs) == 5:
      ncol = 3
    legend = plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y), ncol=nc, framealpha=0.0)
    export_legend(legend, 'graphs/' + graph_name + '_legend.pdf')
  plt.close('all')


plot_qps_recall_graph(result_data, args.graph_name)


