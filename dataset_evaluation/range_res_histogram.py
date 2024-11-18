import numpy as np
import matplotlib.pyplot as plt

p="/ssd1/data/"
filenames = ['FB_ssnpp/ssnpp-1M', 'bigann/range_gt_1M_10000','MSTuringANNS/range_gt_1M_100K_.3', 'gist/range_gt_1M_.5', 'deep1b/range_gt_1M_.02','msmarco_websearch/range_gt_1M_-62', 'wikipedia_cohere/range_gt_1M_-10.5', 'OpenAIArXiv/openai_gt_1M_.2']
titles = ['SSNPP', 'BIGANN', 'MSTuring', 'GIST', 'DEEP', 'MSMARCO', 'Wikipedia', 'OpenAI']
result_dir="graphs/range_result_histograms/"

def plot_distances_hist(filename, title):
    num_results = np.fromfile(p+filename, dtype=np.int32, count=1)[0]
    print("Num results: ", num_results)
    all_results = np.fromfile(p+filename, dtype=np.int32, count=num_results+2)
    result = all_results[2:]

    plt.xlabel("Number of Range Results")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.yscale("log")
    plt.hist(result, bins=int(len(result)/100), alpha=1.0)
    plt.savefig(result_dir + title + ".pdf")
    plt.close()

for file, title in zip(filenames, titles):
    plot_distances_hist(file, title)
