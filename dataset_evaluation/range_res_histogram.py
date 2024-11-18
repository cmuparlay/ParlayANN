import numpy as np
import matplotlib.pyplot as plt

p="/ssd1/data/"
filenames = ['FB_ssnpp/ssnpp-1M', 'bigann/range_gt_1M_10000','MSTuringANNS/range_gt_1M_100K_.3', 'gist/range_gt_1M_.5', 'deep1b/range_gt_1M_.02','msmarco_websearch/range_gt_1M_-62', 'wikipedia_cohere/range_gt_1M_-10.5', 'OpenAIArXiv/openai_gt_1M_.2', 'text2image1B/range_gt_1M_-.6']
titles = ['SSNPP', 'BIGANN', 'MSTuring', 'GIST', 'DEEP', 'MSMARCO', 'Wikipedia', 'OpenAI', 'Text2Image']
result_dir="graphs/range_result_histograms/"

dsinfo = {
  "BIGANN" : {"mult": 100, 
                "color": "C0"},
  "MSTuring" : {"mult": 1000,  
                "color": "C1"},
  "DEEP" : {"mult": 100, 
                "color": "C2"},
  "GIST" : {"mult": 10,  
                "color": "C3"},
  "SSNPP" : {"mult": 1000,  
                "color": "C4"},
  "Wikipedia" : {"mult": 100, 
                "color": "C5"},
  "MSMARCO" : {"mult": 100, 
                "color": "C6"},
  "Text2Image" : {"mult": 1000, 
                "color": "C7"},
  "OpenAI" : {"mult": 100,  
                "color": "C8"},
}

def plot_distances_hist(filename, title):
    num_results = np.fromfile(p+filename, dtype=np.int32, count=1)[0]
    print("Num results: ", num_results)
    all_results = np.fromfile(p+filename, dtype=np.int32, count=num_results+2)
    result = all_results[2:]

    plt.xlabel("Number of Range Results")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.yscale("log")
    plt.hist(result, bins=int(len(result)/dsinfo[title]["mult"]), alpha=1.0, color=dsinfo[title]["color"])
    plt.savefig(result_dir + title + ".pdf")
    plt.close()

for file, title in zip(filenames, titles):
    plot_distances_hist(file, title)
