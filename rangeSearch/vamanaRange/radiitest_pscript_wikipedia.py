import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

if len(sys.argv) != 3:
    print("Usage: python radiitest_pscript.py <csv_file> <recall_value>")
    sys.exit(1)

csv_file = sys.argv[1]
recall_val = sys.argv[2]

total_queries = 5000

label_map = {
    "earlyStopping": "Greedy Search with Early Stopping",
    "doublingSearchEarlyStopping": "Doubling Search with Early Stopping",
    "greedy": "Greedy Search with Early Stopping",
    "doubling": "Doubling Search with Early Stopping"
}

df = pd.read_csv(csv_file)
df["MatchesPerQuery"] = df["NumMatches"] / total_queries

plt.figure(figsize=(10, 6))

for algo, group in df.groupby("Algorithm"):
    readable_label = label_map.get(algo, algo)
    group = group.sort_values("MatchesPerQuery")
    plt.plot(group["MatchesPerQuery"], group["QPS"], marker="o", label=readable_label)
    for _, row in group.iterrows():
        plt.text(row["MatchesPerQuery"], row["QPS"], f"{row['Beam']}", fontsize=15, ha='right')

plt.xlabel("Matches per Query", fontsize=30)
plt.ylabel("QPS", fontsize=30)
plt.xscale("symlog", linthresh=0.2)
plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))

# Save and export legend separately
output_file = f"qps_vs_matches_symlog_wikipedia_{recall_val}.pdf"
plt.grid(True)
plt.tight_layout()
plt.savefig(output_file, bbox_inches='tight')

# === Export legend separately ===
def export_legend(legend, filename="legend.pdf"):
    fig = legend.figure
    fig.patch.set_visible(False)
    ax = fig.axes[0]
    ax.axis('off')
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox, transparent=True)

legend = legend = plt.legend(fontsize=20, frameon=False)
export_legend(legend, filename=f"qps_vs_matches_symlog_wikipedia_{recall_val}_legend.pdf")

plt.close()
