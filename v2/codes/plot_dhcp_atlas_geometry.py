"""Plot dHCP neonatal atlas geometry sensitivity results."""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "outputs" / "dhcp_atlas_geometry_summary.csv")
FIG = ROOT.parent / "manuscript-tex-v2" / "fig11_dhcp_atlas.pdf"
mpl.rcParams.update({"font.family": ["Arial", "DejaVu Sans", "Times New Roman"],
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.7,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.edgecolor": "#444444", "pdf.fonttype": 42, "ps.fonttype": 42})
metrics = [("effective_rank", "effective rank"),
           ("median_nearest_normalized_distance", "median nearest\ndistance"),
           ("p10_nearest_normalized_distance", "10th percentile nearest\ndistance")]
colors = {"age-adaptive": "#D55F6F", "adult-helmet": "#6B6773"}
fig, axes = plt.subplots(1, 3, figsize=(18 / 2.54, 5.7 / 2.54), sharex=True)
x = np.arange(3); width = 0.34
for ax, (key, ylabel) in zip(axes, metrics):
    for offset, layout in [(-width / 2, "age-adaptive"), (width / 2, "adult-helmet")]:
        q = df[df.layout == layout].sort_values("K")
        ax.bar(x + offset, q[key].to_numpy(), width=width, color=colors[layout],
               edgecolor="none", label="AA" if layout == "age-adaptive" else "AH")
    ax.set_xticks(x, ["25", "41", "61"]); ax.set_xlabel("channels"); ax.set_ylabel(ylabel); ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.7); ax.set_axisbelow(True)
axes[0].legend(frameon=True, facecolor="white", edgecolor="#666666", framealpha=0.55, loc="upper left", borderpad=0.25)
for i, ax in enumerate(axes):
    ax.text(-0.15, 1.06, f"({chr(97+i)})", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold", color="#202020")
fig.subplots_adjust(wspace=0.52, bottom=0.25, left=0.08, right=0.995)
fig.savefig(FIG, bbox_inches="tight", dpi=600); plt.close(fig)
print(f"Wrote {FIG}")
