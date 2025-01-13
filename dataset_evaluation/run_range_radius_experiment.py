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
parser.add_argument("-radius", help="radius list of lists")
parser.add_argument("-dataset", help="dataset list")
parser.add_argument("-g","--graphs_only", help="graphs only",action="store_true")
parser.add_argument("-p","--paper_version", help="paper version",action="store_true")
parser.add_argument("-graph_name", help="graphs name")

args = parser.parse_args()
print("datasets " + args.dataset)
print("radiuses: " + args.radius)


already_ran = set()

def string_to_list(s):
  s = s.strip().strip('[').strip(']').split(',')
  return [ss.strip() for ss in s]

def to_list(s):
  if type(s) == list:
    return s
  return [s]

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
    
def runtest(dataset_name, radius, outfile) :

    op = "./range_radius"
    op += " -base_path " + data_options[dataset_name]["base"] 
    op += " -query_path " + data_options[dataset_name]["query"] 
    op += " -data_type " + data_options[dataset_name]["data_type"] 
    op += " -dist_func " + data_options[dataset_name]["dist_fn"] 
    op += " -r " + str(radius)
    runstring(op, outfile)





if '[' in args.radius:
  radiuses = eval(args.radius)
else:
  print('invalid argument')
  exit(1)



if '[' in args.dataset:
  datasets = string_to_list(args.dataset)
else:
  print('invalid argument')
  exit(1)



if not args.graphs_only:
  for dataset_name, radius_list in zip(datasets, radiuses):
    for radius in radius_list:
        directory = "radius_results/"+str(dataset_name)
        os.makedirs(directory, exist_ok=True)
        outFile = directory + "/" + str(radius) + ".txt"
        # clear output file
        os.system("echo \"\" > " + outFile)
        runtest(dataset_name, radius, outFile)

def readResultsFile(dataset_name, radius):
    result_filename = "radius_results/" + str(dataset_name) + "/" + str(radius) + ".txt"
    file = open(result_filename, 'r')
    for line in file.readlines():
        if line.find('Percent covered:') != -1:
            pct = float(line.split(': ')[1])
            return pct

result_data = {}

for dataset_name, radius_list in zip(datasets, radiuses):
    result_data[dataset_name] = {"radiuses":[],"percents":[]}
    for radius in radius_list:
        pct = readResultsFile(dataset_name, radius)
        result_data[dataset_name]["radiuses"].append(float(radius))
        result_data[dataset_name]["percents"].append(pct)


# perform a normalization so that results with very different radii can be plotted on the same graph
def normalize_radius_list(radius_list, pct_list):
    rad_list = sorted(radius_list)
    percent_list = [x for _, x in sorted(zip(radius_list, pct_list))]
    normalized_rad = []
    highest_rad = rad_list[-1]
    additive_factor=0
    if highest_rad < 0:
      additive_factor = 2
    print(rad_list)
    for r in rad_list:
        normalized_rad.append(r/(abs(rad_list[-1]))+additive_factor)
    print(additive_factor)
    print(normalized_rad)
    return normalized_rad, percent_list

for dataset_name in datasets:
    radiuses, percents = normalize_radius_list(result_data[dataset_name]["radiuses"], result_data[dataset_name]["percents"])
    result_data[dataset_name]["radiuses"] = radiuses
    result_data[dataset_name]["percents"] = percents



def export_legend(legend, filename="legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def plot_radius_graph(result_data, graph_name, paper_ver=False):
  os.makedirs("graphs", exist_ok=True)

  outputFile = 'graphs/' + graph_name.replace('.', '') + '.pdf'
  mpl.rcParams.update({'font.size': 25})


  ymax = 0


  fig, axs = plt.subplots()
  # fig = plt.figure()
  opacity = 0.8
  rects = {}
  
  for dataset_name, dataset_data in result_data.items():
    alginfo = dsinfo[dataset_name]
    radius_data = dataset_data["radiuses"]
    pct_data = dataset_data["percents"]
    print(radius_data)
    print(pct_data)
    rects[dataset_name] = axs.plot(pct_data, radius_data,
      alpha=opacity,
      color=alginfo["color"],
      linewidth=3.0,
      linestyle="-",
      marker=alginfo["marker"],
      markersize=14,
      label=dataset_name)


  axs.set_xscale('log')
  # axs.set_yscale('log')
  axs.set(xlabel='Percent Captured', ylabel='Normalized Radius')
  legend_x = 1
  legend_y = 0.5 
  plt.grid()
  if not paper_ver:
    plt.title(graph_name)
    plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y))
  plt.savefig(outputFile, bbox_inches='tight')

  if paper_ver:
    nc = 8
    if len(result_data) == 9:
      nc = 5
    legend = plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y), ncol=nc, framealpha=0.0)
    export_legend(legend, 'graphs/' + graph_name + '_legend.pdf')
  plt.close('all')


plot_radius_graph(result_data, args.graph_name, args.paper_version)


