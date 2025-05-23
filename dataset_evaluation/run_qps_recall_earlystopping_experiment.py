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
from utils import runtest, pareto_frontier, export_legend



parser = argparse.ArgumentParser()
parser.add_argument("-groups", nargs='+', help="specify which groups to plot recall for: all, zeros, onetwos, or threeplus")
parser.add_argument("-datasets", help="dataset list")
parser.add_argument("-g","--graphs_only", help="graphs only",action="store_true")
parser.add_argument("-p","--paper_version", help="paper_version",action="store_true")
parser.add_argument("-graph_name", help="graphs name")

args = parser.parse_args()
print("datasets " + args.datasets)
print("groups:", args.groups)


def string_to_list(s):
  s = s.strip().strip('[').strip(']').split(',')
  return [ss.strip() for ss in s]

valid_groups = ["Greedy Search", "Doubling Search", "Greedy Search with Early Stopping", "Doubling Search with Early Stopping"]


groups = args.groups[0]
groups = eval(groups)


for group in groups:
  if group not in valid_groups:
    print(valid_groups)
    print("Error: group", group, "not valid")
    exit(1)

if '[' in args.datasets:
  datasets = string_to_list(args.datasets)
else:
  print('invalid argument')
  exit(1)



if not args.graphs_only:
  for dataset_name in datasets:
      directory = "qps_recall_results_earlystopping/"
      os.makedirs(directory, exist_ok=True)
      outFile = directory + "/" + str(dataset_name) + ".txt"
      # clear output file
      os.system("echo \"\" > " + outFile)
      # clear output file
      os.system("echo \"\" > " + outFile)
      for group in groups:
        if group == "Greedy Search":
          search_mode = ""
        elif group == "Doubling Search":
          search_mode = "doublingSearch"
        elif group == "Greedy Search with Early Stopping":
          search_mode = "earlyStopping"
        elif group == "Doubling Search with Early Stopping":
          search_mode = "doublingSearchEarlyStopping"
        print("Running test for dataset:", dataset_name, "group:", group)
        # run the test
        with open(outFile, 'a') as f:
          f.write("\n")
          f.write(group + ":" + "\n")
          f.write("\n")
        runtest(dataset_name, outFile, search_mode)




'''
Results file is understood to include the QPS/Recall curves for *four* different groups
and *three* different algorithms
Groups are: all, zeros, onetwos, threepluses
Algorithms are: beam search, beam search + range search, beam search + range search + early stopping
Also includes *two* different recall choices: cumulative and pointwise
Returns list of recalls and list of QPS
'''
def readResultsFile(dataset_name, search_strategy):
  file_path = "qps_recall_results_earlystopping/" + str(dataset_name) + ".txt"
  qps_list = []
  recall_list = []
  inside_section = False  # Flag to track when we're in the correct section

  with open(file_path, 'r', encoding="utf-8") as file:
    for line in file:
      # Check if we've entered the right section
      if line.strip() == f"{search_strategy}:":
        inside_section = True
        continue  # Move to the next line

      # Stop extraction when we reach another search strategy
      if inside_section:
        for group in valid_groups:
          if line.strip() == f"{group}:":
            inside_section = False
            break 

      # Extract values once inside the section
      if inside_section and "For Beam:" in line:
        parts = line.split(", ")
        # print(parts)
        if len(parts) >= 3:  # Ensure we have enough components
          recall_value = float(parts[2].split("=")[1].strip())
          qps_value = float(parts[3].split("=")[1].strip())

          recall_list.append(recall_value)
          qps_list.append(qps_value)
    print("QPS for ", search_strategy,  qps_list)
    print("Average Precision", search_strategy, recall_list)

  return qps_list, recall_list

result_data = {}

for dataset_name in datasets:
    result_data[dataset_name] = {}
    for group in groups:
        qps, recall = readResultsFile(dataset_name, group)
        result_data[dataset_name][group + ", QPS"] = qps
        result_data[dataset_name][group] = recall

# print(result_data)

linestyles = {"Greedy Search": "solid",
              "Doubling Search": "dashdot",
              "Greedy Search with Early Stopping": "dashed",
              "Doubling Search with Early Stopping": "dotted",
            }

def plot_qps_recall_graph(result_data, graph_name, paper_ver=False):
  os.makedirs("graphs/qps_recall_earlystopping", exist_ok=True)

  outputFile = 'graphs/qps_recall_earlystopping/' + graph_name.replace('.', '') + '.pdf'
  mpl.rcParams.update({'font.size': 35})

  print(outputFile)
  xmin = 1.0
  xmax = 0


  fig, axs = plt.subplots()
  # fig = plt.figure()
  
  rects = {}
  
  for dataset_name, dataset_data in result_data.items():
    alginfo = dsinfo[dataset_name]
    
    for group in groups:
      algorithm = group
      recall_type = "Average Precision"
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
    legend = plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y), ncol=nc, framealpha=0.0)
    export_legend(legend, 'graphs/' + graph_name + '_legend.pdf')
  plt.close('all')


plot_qps_recall_graph(result_data, args.graph_name, args.paper_version)


