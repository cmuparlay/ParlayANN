import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import runtest, string_to_list

# Algorithm config: (search_mode, early_stop, base_color, top_color, label)
ALGORITHM_CONFIG = {
    "Beam Search": ("beam", False, "tab:blue", None, "Beam Search"),
    "Greedy Search": ("greedy", False, "tab:blue", "tab:orange", "Greedy Search"),
    "Greedy + ES": ("greedy", True, "tab:gray", "tab:orange", "Early Stopping Beam Search"),
    "Doubling Search": ("doubling", False, "tab:blue", "tab:green", "Doubling Beam Search"),
    "Doubling + ES": ("doubling", True, "tab:gray", "tab:green", "Doubling + Early Stop"),
}

# === Parse beam logs ===
def parse_beam_logs(result_file):
    beam_data = []
    with open(result_file, 'r') as f:
        for line in f:
            if "For Beam:" in line:
                parts = line.split(", ")
                beam_value = int(parts[0].split(":")[1].strip())
                recall_value = float(parts[2].split("=")[1].strip())
                timing_list = string_to_list(parts[7].split("=")[1].strip())
                t1=float(timing_list[0])
                t2=float(timing_list[1])
                beam_data.append((beam_value, recall_value, t1, t2))
            # match = pattern.search(line)
            # if match:
            #     beam = int(match.group(1))
            #     recall = float(match.group(2))
            #     beam_time = float(match.group(3))
            #     other_time = float(match.group(4))
            #     beam_data.append((beam, recall, beam_time, other_time))
    return beam_data

# === Find best beam for each recall threshold ===
def find_min_beams(beam_data, recall_cuts):
    selected = []
    for recall_cut in recall_cuts:
        candidates = [entry for entry in beam_data if entry[1] >= recall_cut and (entry[2] + entry[3]) > 0]
        if candidates:
            best = min(candidates, key=lambda x: x[0])
            selected.append((recall_cut, best[0], best[2], best[3]))
        else:
            selected.append((recall_cut, None, 0.0, 0.0))
    return selected

# === Plot grouped stacked bar chart ===
def plot_grouped_bar_chart(all_data, recall_cuts, output_file):
    algorithms = list(all_data.keys())
    num_algos = len(algorithms)
    x = np.arange(len(recall_cuts))
    width = 0.13

    fig, ax = plt.subplots(figsize=(10, 6))
    max_height = 0

    for i, algo in enumerate(algorithms):
        search_mode, early_stop, base_color, top_color, _ = ALGORITHM_CONFIG[algo]
        data = all_data[algo]
        beam_times = [bt for _, _, bt, _ in data]
        other_times = [ot for _, _, _, ot in data]
        beams = [b if b is not None else 0 for _, b, _, _ in data]

        offset = width * i - width * (num_algos - 1) / 2
        bar1 = ax.bar(x + offset, beam_times, width, edgecolor='black', color=base_color)
        bar2 = ax.bar(x + offset, other_times, width, bottom=beam_times, edgecolor='black', color=top_color)

        for j in range(len(bar1)):
            total = beam_times[j] + other_times[j]
            max_height = max(max_height, total)
            ax.text(bar2[j].get_x() + bar2[j].get_width() / 2,
                    total + 0.2, str(beams[j]), ha='center', fontsize=9)

    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Average Precision")
    ax.set_title(output_file)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.3f}" for r in recall_cuts])
    ax.set_yscale('symlog')
    ax.set_ylim(0, max_height * 2)
    ax.grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig(output_file + ".pdf")
    plt.close()

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", required=True, help="dataset")
    parser.add_argument("-recalls", required=True, help="recall list, e.g. [0.99, 0.995, 0.999]")
    parser.add_argument("-graph_name", required=True, help="graph output name")
    parser.add_argument("-g","--graphs_only", help="graphs only",action="store_true")

    args = parser.parse_args()
    recalls = [float(r.strip()) for r in args.recalls.strip("[]").split(",")]
    output_dir = "bar_chart_results"
    os.makedirs(output_dir, exist_ok=True)

    all_result_data = {}

    for algo in ALGORITHM_CONFIG:
        search_mode, early_stop, _, _, _ = ALGORITHM_CONFIG[algo]
        outfile = os.path.join(output_dir, f"{args.dataset}_{search_mode}_{'es' if early_stop else 'noes'}.txt")
        if not args.graphs_only:
            # clear output file
            os.system("echo \"\" > " + outfile)
            runtest(args.dataset, outfile, search_mode, early_stop)
        beam_data = parse_beam_logs(outfile)
        selected = find_min_beams(beam_data, recalls)
        all_result_data[algo] = selected

    plot_grouped_bar_chart(all_result_data, recalls, args.graph_name)
