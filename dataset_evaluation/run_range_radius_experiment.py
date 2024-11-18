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



# parameters:
#  - datastructures: neighbors_bench only for now
#  - threads, update_percent, number of nearest neighbors


bigann = "/ssd1/data/bigann"
msturing= "/ssd1/data/MSTuringANNS"
deep = "/ssd1/data/deep1b"
gist="/ssd1/data/gist"
ssnpp="/ssd1/data/FB_ssnpp"
wikipedia="/ssd1/data/wikipedia_cohere"
msmarco="/ssd1/data/msmarco_websearch"
text2image="/ssd1/data/text2image1B"
openai="/ssd1/data/OpenAIArXiv"
data_options = {
  "bigann-1M" : {"base": bigann+"/base.1B.u8bin.crop_nb_1000000", 
                "query": bigann+"/query.public.10K.u8bin", 
                "data_type" : "uint8", 
                "dist_fn": "Euclidian"},
  "msturing-1M" : {"base": msturing+"/base1b.fbin.crop_nb_1000000", 
                "query": msturing+"/query10K.fbin", 
                "data_type" : "float", 
                "dist_fn": "Euclidian"},
  "deep-1M" : {"base": deep+"/base.1B.fbin.crop_nb_1000000", 
                "query": deep+"/query.public.10K.fbin", 
                "data_type" : "float",
                "dist_fn": "Euclidian"},
  "gist-1M" : {"base": gist+"/gist_base.fbin", 
                "query": gist+"/gist_query.fbin", 
                "data_type" : "float", 
                "dist_fn": "Euclidian"},
  "ssnpp-1M" : {"base": ssnpp+"/FB_ssnpp_database.u8bin.crop_nb_1000000", 
                "query": ssnpp+"/FB_ssnpp_public_queries.u8bin", 
                "data_type" : "uint8", 
                "dist_fn": "Euclidian"},
  "wikipedia-1M" : {"base":wikipedia+"/wikipedia_base.bin.crop_nb_1000000", 
                "query": wikipedia+"/wikipedia_query.bin", 
                "data_type" : "float", 
                "dist_fn": "mips"},
  "msmarco-1M" : {"base": msmarco+"/vectors.bin.crop_nb_1000000", 
                "query": msmarco+"/query.bin", 
                "data_type" : "float", 
                "dist_fn": "mips"},
  "text2image-1M" : {"base": text2image+"/base.1B.fbin.crop_nb_1000000", 
                "query": text2image+"/query.public.100K.fbin", 
                "data_type" : "float", 
                "dist_fn": "mips"},
  "openai-1M" : {"base": openai+"/openai_base_1M.bin", 
                "query": openai+"/openai_query_10K.bin", 
                "data_type" : "float", 
                "dist_fn": "Euclidian"},
}
parser = argparse.ArgumentParser()
parser.add_argument("-radius", help="radius list of lists")
parser.add_argument("-dataset", help="dataset list")
parser.add_argument("-g","--graphs_only", help="graphs only",action="store_true")
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
    for r in rad_list:
        normalized_rad.append(r/(rad_list[-1]))
    return normalized_rad, percent_list

for dataset_name in datasets:
    radiuses, percents = normalize_radius_list(result_data[dataset_name]["radiuses"], result_data[dataset_name]["percents"])
    result_data[dataset_name]["radiuses"] = radiuses
    result_data[dataset_name]["percents"] = percents

mk = ['o', 'v', '^', '1', 's', '+', 'x', 'D', '|', '>', '<',]

dsinfo = {
  "bigann-1M" : {"marker": mk[0], 
                "color": "C0"},
  "msturing-1M" : {"marker": mk[1], 
                "color": "C1"},
  "deep-1M" : {"marker": mk[2], 
                "color": "C2"},
  "gist-1M" : {"marker": mk[3], 
                "color": "C3"},
  "ssnpp-1M" : {"marker": mk[4], 
                "color": "C4"},
  "wikipedia-1M" : {"marker": mk[5], 
                "color": "C5"},
  "msmarco-1M" : {"marker": mk[6], 
                "color": "C6"},
  "text2image-1M" : {"marker": mk[7], 
                "color": "C7"},
  "openai-1M" : {"marker": mk[8], 
                "color": "C8"},
}

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
    if len(algs) == 2:
      ncol = 1
    elif len(algs) == 5:
      ncol = 3
    legend = plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y), ncol=nc, framealpha=0.0)
    export_legend(legend, 'graphs/' + graph_name + '_legend.pdf')
  plt.close('all')


plot_radius_graph(result_data, args.graph_name)


