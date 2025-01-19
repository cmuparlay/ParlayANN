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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import statistics as st
from dataset_info import mk, dsinfo, data_options
from utils import runtest



parser = argparse.ArgumentParser()
parser.add_argument("-dataset", help="dataset")
parser.add_argument("-recalls", help="recall list")
parser.add_argument("-g","--graphs_only", help="graphs only",action="store_true")
parser.add_argument("-graph_name", help="graphs name")  

args = parser.parse_args()
print("dataset " + args.dataset)
print("recalls:", args.recalls)

already_ran = set()

def string_to_list(s):
  s = s.strip().strip('[').strip(']').split(',')
  return [ss.strip() for ss in s]


if '[' in args.recalls:
  recalls = string_to_list(args.recalls)
else:
  print('invalid argument')
  exit(1)



if not args.graphs_only:
    directory = "qps_recall_results/"
    os.makedirs(directory, exist_ok=True)
    outFile = directory + "/" + str(args.dataset) + ".txt"
    # clear output file
    os.system("echo \"\" > " + outFile)
    runtest(args.dataset, outFile)

'''
Results file is understood to include the QPS/Recall curves for *four* different groups
and *three* different algorithms
Groups are: all, zeros, onetwos, threepluses
Algorithms are: beam search, beam search + range search, beam search + range search + early stopping
Also includes *two* different recall choices: cumulative and pointwise
Also includes list of pairs of timings for each run
Returns list of recalls and list of QPS
'''
def readResultsFile(dataset_name, recall_line, timings_line, beams_line):
    result_filename = "qps_recall_results/" + str(dataset_name) + ".txt"
    file = open(result_filename, 'r')
    for line in file.readlines():
      if line.find(recall_line) != -1:
          recall = line.split(': ')[1]
          recall = eval(recall)
      if line.find(timings_line) != -1:
          times = line.split(': ')[1]
          times = eval(times)
      if line.find(beams_line) != -1:
          beams = line.split(': ')[1]
          beams = eval(beams)
    return times, recall, beams

result_data = {}

algorithms=["Beam Search", "Greedy Search", "Doubling Search", "Early Stopping"]

for algorithm in algorithms:
    result_data[algorithm] = {}
    recall_line = "All, " + algorithm + ", Cumulative Recall"
    timings_line = "All, " + algorithm + ", Timings"
    beams_line = "All, " + algorithm + ", Beams"
    timings, recall, beams = readResultsFile(args.dataset, recall_line, timings_line, beams_line)
    result_data[algorithm]["recall"] = recall
    result_data[algorithm]["timings"] = timings
    result_data[algorithm]["beams"] = beams

print(result_data)


def export_legend(legend, filename="legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

# takes in a list of recalls and returns the index of the one closest to the target
def find_index_closest_to(recall_list, target):
  closest_value = min(recall_list, key=lambda x: abs(x - target))
  return recall_list.index(closest_value)

alginfo = {
  "Beam Search" : {"color": "C0"},
  "Greedy Search" : {"color": "C1"},
  "Doubling Search" : {"color": "C2"},
  "Early Stopping": {"color": "C3"},
}

def plot_time_bar_chart(result_data, graph_name, paper_ver=False):
  directory = "graphs/bar_charts"
  os.makedirs(directory, exist_ok=True)

  outputFile = directory + "/" + graph_name.replace('.', '') + '.pdf'
  mpl.rcParams.update({'font.size': 25})

  print(outputFile)
  max_height=0

  fig, ax = plt.subplots()

  width = .2
  multiplier = 0

  x = np.arange(len(recalls))

  # create a set of bars with one color for each algorithm
  for algorithm in algorithms:
    beams = []
    beam_heights = []
    total_heights = []
    data = result_data[algorithm]
    for recall in recalls:
      index = find_index_closest_to(data["recall"], float(recall))
      if abs(data["recall"][index]-float(recall)) > 1:
        print("No recall values close enough to", recall, "continuing")
        continue
      beams.append(data["beams"][index])
      total_time = float(data["timings"][index][0]) + float(data["timings"][index][1])
      if total_time > max_height:
        max_height = total_time
      total_heights.append(float(data["timings"][index][1]))
      beam_heights.append(float(data["timings"][index][0]))
    offset = width * multiplier
    labels = [str(x) for x in beams]
    rects_low = ax.bar(x+offset, beam_heights, width, color=alginfo["Beam Search"]["color"])
    rects_high = ax.bar(x+offset, total_heights, width, bottom=beam_heights, label=algorithm, color=(alginfo[algorithm])["color"])
    ax.bar_label(rects_high, labels=labels, padding=2)
    multiplier += 1

  ax.set_ylabel('Time (s)')
  ax.set_xticks(x + width, recalls)


  legend_x = 1
  legend_y = 0.5 
  plt.grid()
  plt.tight_layout()
  ax.set_ylim(0, max_height+1.2)
  # fig.set_figheight(max_height+1)
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
    export_legend(legend, directory + "/" + graph_name + '_legend.pdf')
  plt.close('all')


plot_time_bar_chart(result_data, args.graph_name)


