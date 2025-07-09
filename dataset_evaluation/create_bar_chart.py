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
                t1 = float(timing_list[0])
                t2 = float(timing_list[1])
                beam_data.append((beam_value, recall_value, t1, t2))
    return beam_data

# === Find best beam for each recall threshold ===
def find_min_beams(beam_data, recall_cuts):
    selected = []
    for recall_cut in recall_cuts:
        candidates = [entry for entry in beam_data if entry[1] >= recall_cut and (entry[2] + entry[3]) > 0]
        if candidates:
            best = min(candidates, key=lambda x: x[0])
            selected.append((recall_cut, best[0], best[2], best[3]))
    return selected

# === Export legend to standalone file ===
def export_legend(legend, filename="legend.pdf"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

# === Plot grouped stacked bar chart ===
def plot_grouped_bar_chart(all_data, recall_cuts, output_file, export_legend_flag=True):
    algorithms = list(all_data.keys())
    num_algos = len(algorithms)
    x = np.arange(len(recall_cuts))
    width = 0.13

    fig, ax = plt.subplots(figsize=(10, 6))
    max_height = 0
    legend_handles = []
    legend_labels = []

    for i, algo in enumerate(algorithms):
        data = all_data[algo]
        filtered_indices = [j for j, entry in enumerate(data) if entry[1] is not None]
        if not filtered_indices:
            continue

        x_filtered = np.array([x[j] for j in filtered_indices])
        beam_times = [data[j][2] for j in filtered_indices]
        other_times = [data[j][3] for j in filtered_indices]
        beams = [data[j][1] for j in filtered_indices]

        search_mode, early_stop, base_color, top_color, label = ALGORITHM_CONFIG[algo]
        offset = width * i - width * (num_algos - 1) / 2

        bar1 = ax.bar(x_filtered + offset, beam_times, width, edgecolor='black', color=base_color)
        if top_color:
            bar2 = ax.bar(x_filtered + offset, other_times, width, bottom=beam_times, edgecolor='black', color=top_color)
        else:
            bar2 = ax.bar(x_filtered + offset, other_times, width, bottom=beam_times, edgecolor='black', color=base_color)

        for j in range(len(bar1)):
            total = beam_times[j] + other_times[j]
            max_height = max(max_height, total)
            ax.text(bar2[j].get_x() + bar2[j].get_width() / 2, total + 0.2, str(beams[j]), ha='center', fontsize=9)

        legend_handles.append(bar2)
        legend_labels.append(label)

    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Average Precision")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.3f}" for r in recall_cuts])
    ax.set_yscale('symlog')
    ax.set_ylim(0, max_height * 2)
    ax.grid(True, linestyle=":")

    plt.tight_layout()
    os.makedirs("graphs/bar_charts", exist_ok=True)
    plt.savefig("graphs/bar_charts/" + output_file + ".pdf", bbox_inches='tight')
    plt.close()

    # Export legend separately
    if export_legend_flag:
        from matplotlib.patches import Patch
        legend_handles = []
        legend_labels = []

        # Add bottom parts (beam or early stop)
        legend_handles.append(Patch(facecolor='tab:blue', edgecolor='black'))
        legend_labels.append('Beam Search')

        legend_handles.append(Patch(facecolor='tab:gray', edgecolor='black'))
        legend_labels.append('Early Stopping Beam Search')

        # Add top parts
        legend_handles.append(Patch(facecolor='tab:orange', edgecolor='black'))
        legend_labels.append('Greedy Search')

        legend_handles.append(Patch(facecolor='tab:green', edgecolor='black'))
        legend_labels.append('Doubling Beam Search')

        fig_legend = plt.figure(figsize=(8, 1.5))
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')
        legend = ax_legend.legend(legend_handles, legend_labels, loc='center', ncol=4, frameon=False, handlelength=1.5, fontsize=12)
        export_legend(legend, "graphs/bar_charts/" + output_file + "_legend.pdf")
        plt.close(fig_legend)

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", required=True, help="dataset")
    parser.add_argument("-recalls", required=True, help="recall list, e.g. [0.99, 0.995, 0.999]")
    parser.add_argument("-graph_name", required=True, help="graph output name")
    parser.add_argument("-g", "--graphs_only", help="graphs only", action="store_true")

    args = parser.parse_args()
    recalls = [float(r.strip()) for r in args.recalls.strip("[]").split(",")]
    output_dir = "bar_chart_results"
    os.makedirs(output_dir, exist_ok=True)

    all_result_data = {}

    for algo in ALGORITHM_CONFIG:
        search_mode, early_stop, _, _, _ = ALGORITHM_CONFIG[algo]
        outfile = os.path.join(output_dir, f"{args.dataset}_{search_mode}_{'es' if early_stop else 'noes'}.txt")
        if not args.graphs_only:
            os.system("echo \"\" > " + outfile)
            runtest(args.dataset, outfile, search_mode, early_stop)
        beam_data = parse_beam_logs(outfile)
        selected = find_min_beams(beam_data, recalls)
        all_result_data[algo] = selected

    plot_grouped_bar_chart(all_result_data, recalls, args.graph_name)
