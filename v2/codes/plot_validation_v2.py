"""Generate the additional vector validation figures from v2 CSV outputs."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ecd_robustness_v2"
FIG = ROOT.parent / "manuscript-tex-v2"
AA = "age-adaptive"
AH = "adult-helmet"
COLORS = {AA: "#D55F6F", AH: "#6B6773"}
MARKERS = {AA: "o", AH: "s"}
AGES = [12, 48, 120, 216]
K = [25, 41, 61]


def set_style():
    mpl.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans", "Times New Roman"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.15,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.65,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel(ax, label):
    ax.text(-0.14, 1.06, f"({label})", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, fontweight="bold",
            color="#202020", clip_on=False)


def figure_ecd():
    df = pd.read_csv(OUT / "ecd_summary.csv")
    df = df[(df.array == "D4") & (df.noise_model == "iid") & (df.mismatch == "none")]
    fig, axs = plt.subplots(2, 2, figsize=(18 / 2.54, 13.2 / 2.54), sharex=True, sharey=True)
    for ax, age, lab in zip(axs.flat, AGES, "abcd"):
        sub = df[df.age_months == age]
        x = np.arange(len(K))
        width = 0.34
        for offset, layout in [(-width / 2, AA), (width / 2, AH)]:
            q = sub[sub.layout == layout].sort_values("K")
            med = q.median_error_mm.to_numpy()
            p90 = q.p90_error_mm.to_numpy()
            ax.bar(x + offset, med, width=width, color=COLORS[layout],
                   edgecolor="none",
                   yerr=np.vstack([np.zeros_like(med), p90 - med]),
                   error_kw={"ecolor": "#333333", "elinewidth": 0.8, "capsize": 0.0},
                   label="AA" if layout == AA else "AH")
            for xi, yi in zip(x + offset, p90):
                ax.plot([xi - 0.06, xi + 0.06], [yi, yi], color="#333333", linewidth=0.8)
        ax.set_xticks(x, K)
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.set_xlabel("channels")
        ax.set_ylabel("ECD localization error (mm)")
        ax.set_ylim(0, 110)
        ax.grid(True)
        ax.set_axisbelow(True)
        panel(ax, lab)
    handles = [
        Line2D([0], [0], color=COLORS[AA], marker=MARKERS[AA], lw=1.1, label="AA"),
        Line2D([0], [0], color=COLORS[AH], marker=MARKERS[AH], lw=1.1, ls="--", label="AH"),
    ]
    axs[0, 0].legend(handles=handles, frameon=True, facecolor="white", edgecolor="#666666",
                     framealpha=0.55, loc="upper right", borderpad=0.25, handlelength=1.3)
    fig.subplots_adjust(hspace=0.42, wspace=0.36)
    fig.savefig(FIG / "fig09_ecd.pdf")
    plt.close(fig)


def figure_robustness():
    base = pd.read_csv(OUT / "ecd_summary.csv")
    base = base[(base.array == "D4") & (base.K == 41) & (base.noise_model == "iid") & (base.mismatch == "none")]
    rob = pd.read_csv(OUT / "robustness_summary.csv")
    corr = rob[(rob.experiment == "correlated_noise") & (rob.K == 41)]
    mismatch = rob[(rob.experiment == "registration_mismatch") & (rob.K == 41) & (rob.mismatch == "combined_3mm_3deg")]

    fig, axs = plt.subplots(1, 2, figsize=(18 / 2.54, 7.8 / 2.54))
    age_color = {12: "#005A8D", 120: "#178B5B", 216: "#8A4FA3"}
    for ax, rows, lab in [(axs[0], corr, "a"), (axs[1], mismatch, "b")]:
        for age in [12, 120, 216]:
            for layout in [AA, AH]:
                q = rows[(rows.age_months == age) & (rows.layout == layout)]
                b = base[(base.age_months == age) & (base.layout == layout)]
                if q.empty or b.empty:
                    continue
                ax.scatter(b.median_error_mm.iloc[0], q.median_error_mm.iloc[0],
                           color=age_color[age], marker=MARKERS[layout], s=25,
                           edgecolors="white", linewidths=0.4, zorder=3)
        ax.plot([0, 70], [0, 70], color="#555555", lw=0.8, ls=":")
        ax.set_xlim(0, 70)
        ax.set_ylim(0, 70)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("baseline median ECD error (mm)")
        ax.set_ylabel("perturbed median ECD error (mm)")
        ax.grid(True)
        panel(ax, lab)

    age_handles = [Line2D([0], [0], marker="o", lw=0, color=age_color[a], label=f"{a} mo")
                   for a in [12, 120, 216]]
    layout_handles = [Line2D([0], [0], marker=MARKERS[l], lw=0, color="#444444",
                             markerfacecolor="#777777", label="AA" if l == AA else "AH")
                      for l in [AA, AH]]
    axs[0].legend(handles=age_handles + layout_handles, ncol=2, frameon=True,
                  facecolor="white", edgecolor="#666666", framealpha=0.55,
                  loc="upper left", borderpad=0.25, handletextpad=0.3, columnspacing=0.6)
    fig.subplots_adjust(wspace=0.40)
    fig.savefig(FIG / "fig10_robustness.pdf")
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    figure_ecd()
    figure_robustness()
    print(f"Wrote vector figures to {FIG}")
