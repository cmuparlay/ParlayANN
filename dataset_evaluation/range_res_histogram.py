import numpy as np
import matplotlib.pyplot as plt
import math 

p="/ssd1/data/"
filenames = ['FB_ssnpp/ssnpp-1M', 'FB_ssnpp/ssnpp-10M', 'FB_ssnpp/ssnpp-100M', 'bigann/range_gt_1M_10000','bigann/range_gt_10M_10000','bigann/range_gt_100M_10000','MSTuringANNS/range_gt_1M_100K_.3', 'gist/range_gt_1M_10K_.5', 'deep1b/range_gt_1M_.02','msmarco_websearch/range_gt_1M_-62', 'wikipedia_cohere/range_gt_1M_-10.5', 'wikipedia_cohere/range_gt_10M_-10.5', 'OpenAIArXiv/openai_gt_1M_.2', 'text2image1B/range_gt_1M_-.6']
titles = ['SSNPP', 'SSNPP-10M', 'SSNPP-100M', 'BIGANN', 'BIGANN-10M', 'BIGANN-100M', 'MSTuring', 'GIST', 'DEEP', 'MSMARCO', 'Wikipedia', 'Wikipedia-10M', 'OpenAI', 'Text2Image']
result_dir="graphs/range_result_histograms/"

dsinfo = {
  "BIGANN" : {"mult": 100, 
                "color": "C0"},
  "BIGANN-10M" : {"mult": 100, 
                "color": "C0"},
  "BIGANN-100M" : {"mult": 100, 
                "color": "C0"},
  "MSTuring" : {"mult": 1000,  
                "color": "C1"},
  "DEEP" : {"mult": 100, 
                "color": "C2"},
  "GIST" : {"mult": 100,  
                "color": "C3"},
  "SSNPP" : {"mult": 1000,  
                "color": "C4"},
  "SSNPP-10M" : {"mult": 1000,  
                "color": "C4"},
  "SSNPP-100M" : {"mult": 1000,  
                "color": "C4"},
  "Wikipedia" : {"mult": 100, 
                "color": "C5"},
  "Wikipedia-10M" : {"mult": 100, 
                "color": "C5"},
  "MSMARCO" : {"mult": 100, 
                "color": "C6"},
  "Text2Image" : {"mult": 1000, 
                "color": "C7"},
  "OpenAI" : {"mult": 100,  
                "color": "C8"},
}

def range_res_stats(results):
    counts = np.zeros(8)
    for result in results:
        if result == 0:
            counts[0] += 1
        elif result <= 10:
            counts[1] += 1
        else:
            power = math.ceil(math.log10(result))
            assert(power <= 8)
            counts[power] += 1

    print(counts)


def plot_distances_hist(filename, title):
    num_results = np.fromfile(p+filename, dtype=np.int32, count=1)[0]
    print("Dataset:", title)
    print("Num results: ", num_results)
    all_results = np.fromfile(p+filename, dtype=np.int32, count=num_results+2)
    result = all_results[2:]
    range_res_stats(result)

    plt.rcParams.update({'font.size': 18})

    plt.xlabel("Number of Range Results")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.hist(result, bins=int(len(result)/dsinfo[title]["mult"]), alpha=1.0, color=dsinfo[title]["color"])
    plt.tight_layout()
    plt.savefig(result_dir + title + ".pdf")
    plt.close()

for file, title in zip(filenames, titles):
    plot_distances_hist(file, title)
