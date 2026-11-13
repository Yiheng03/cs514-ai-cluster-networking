"""Make the report figures from results/ and data/."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, DATA, FIG = ROOT / "results", ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Each component keeps the same color in every figure.
COL = {"TP": "#2a78d6", "DP": "#eb6834", "Computation": "#1baf7a",
       "EP all-to-all": "#eda100", "Expert DP": "#e87ba4", "Bubble": "#6250d6"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
BAR = dict(edgecolor="white", linewidth=1.0)


def stacked(ax, x, parts, width, labels=True):
    bottom = np.zeros(len(x))
    for name, vals in parts:
        ax.bar(x, vals, width, bottom=bottom, color=COL[name],
               label=name if labels else None, **BAR)
        bottom += np.asarray(vals)
    return bottom


def fig10():
    paper = pd.read_csv(DATA / "fig10_digitized.csv")
    paper = paper[paper.series == "simai"]
    ours = pd.read_csv(RES / "fig10_ours.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
    for ax, gpu in zip(axes, ["H100", "H20"]):
        p = paper[paper.gpu == gpu].sort_values("bw_gbit")
        o = ours[ours.gpu == gpu].sort_values("bw_gbit")
        x = np.arange(3)
        stacked(ax, x - 0.2, [("TP", p.tp), ("DP", p.dp), ("Computation", p.comp)],
                0.36, labels=(gpu == "H100"))
        tot = stacked(ax, x + 0.2, [("TP", o.tp), ("DP", o.dp),
                                    ("Computation", o.comp_incl_bubble)], 0.36, labels=False)
        for xi, t, pt in zip(x, tot, p.total):
            ax.text(xi - 0.2, pt, f"{pt:.2f}", ha="center", va="bottom", fontsize=6.5, color=MUTED)
            ax.text(xi + 0.2, t, f"{t:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(np.r_[x - 0.2, x + 0.2])
        ax.set_xticklabels(["paper"] * 3 + ["ours"] * 3, fontsize=6.5)
        sec = ax.secondary_xaxis(-0.13)
        sec.set_xticks(x)
        sec.set_xticklabels(["100 Gbit/s", "200 Gbit/s", "400 Gbit/s"])
        sec.spines["bottom"].set_visible(False)
        sec.tick_params(length=0)
        ax.set_title(f"{gpu}", fontsize=9, color=INK)
        ax.set_ylabel("Iteration time (s)")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.savefig(FIG / "fig10_replication.pdf")


def fig11b():
    cmp_ = pd.read_csv(RES / "fig11b_compare.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
    hosts = [8, 16, 32, 64]
    for ax, gpu in zip(axes, ["H100", "H20"]):
        d = cmp_[cmp_.gpu == gpu]
        for i, tp in enumerate([8, 16, 32]):
            s = d[d.tp == tp].set_index("gpus_per_host").reindex(hosts)
            x = np.arange(4) + (i - 1) * 0.26
            ax.bar(x, s.norm_time, 0.24, color=SERIES[i], label=f"TP {tp} (ours)", **BAR)
            ax.scatter(x, s.norm_time_paper, marker="_", s=90, color=INK,
                       linewidths=1.6, zorder=3, label="paper" if i == 0 else None)
        ax.set_xticks(range(4))
        ax.set_xticklabels([str(h) for h in hosts])
        ax.set_xlabel("GPUs per host")
        ax.set_title(gpu, fontsize=9)
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("Normalized iteration time")
    axes[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.08), fontsize=7)
    fig.savefig(FIG / "fig11b_replication.pdf")


def moe_gain():
    A = pd.read_csv(RES / "moe_bandwidth_ep.csv")
    A = A[(A["skew"] == 1) & (A["comp_scale"] == 1)]
    g3 = pd.read_csv(RES / "fig10_ours.csv").pivot(index="gpu", columns="bw_gbit", values="total")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    for ax, gpu in zip(axes, ["H100", "H20"]):
        for i, world in enumerate([1024, 256]):
            p = A[(A.gpu == gpu) & (A.world == world)].pivot(index="ep", columns="bw_gbit", values="t_total")
            gain = 1 - p[400] / p[200]
            ax.plot(p.index, gain * 100, marker="o", ms=4, lw=1.6, color=SERIES[i],
                    label=f"DeepSeek, {world} GPUs ({1024 * 4096 // world} tok/GPU)")
        ref = (1 - g3.loc[gpu, 400] / g3.loc[gpu, 200]) * 100
        ax.axhline(ref, color=MUTED, ls="--", lw=1)
        ax.text(64, ref * 0.97, f"GPT-3 175B: {ref:.1f}%", ha="right", va="top", fontsize=7, color=MUTED)
        ax.axvspan(6, 11, color="#f0efec", zorder=0)
        ax.text(8, 1, "EP in one\n8-GPU host", ha="center",
                fontsize=6.5, color=MUTED, va="bottom")
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64])
        ax.set_xticklabels(["8", "16", "32", "64"])
        ax.set_xlabel("EP size (8-GPU hosts)")
        ax.set_title(f"{gpu}: time saved by 200 → 400 Gbit/s", fontsize=9)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("Iteration-time reduction (%)")
    axes[0].legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.1), ncol=2)
    fig.savefig(FIG / "moe_bandwidth_gain.pdf")


def moe_breakdown():
    A = pd.read_csv(RES / "moe_bandwidth_ep.csv")
    A = A[(A["skew"] == 1) & (A["comp_scale"] == 1) & (A.gpu == "H100") & (A.bw_gbit == 200)]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    for ax, world in zip(axes, [1024, 256]):
        d = A[A.world == world].sort_values("ep")
        x = np.arange(len(d))
        stacked(ax, x, [("Computation", d.t_comp), ("EP all-to-all", d.t_ep),
                        ("DP", d.t_dp), ("Expert DP", d.t_dp_ep)], 0.55,
                labels=(world == 1024))
        ax.set_xticks(x)
        ax.set_xticklabels([f"EP {e}" for e in d.ep])
        ax.set_title(f"H100, 200 Gbit/s, {world} GPUs", fontsize=9)
    axes[0].set_ylabel("Iteration time (s)")
    axes[0].legend(fontsize=7, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.1))
    fig.savefig(FIG / "moe_breakdown.pdf")


def moe_hostsize():
    B = pd.read_csv(RES / "moe_hostsize_ep.csv")
    g = pd.read_csv(RES / "fig11b_ours.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    for ax, (gpu, bw) in zip(axes, [("H100", 200), ("H20", 200)]):
        d = B[(B.gpu == gpu) & (B.bw_gbit == bw)]
        for i, ep in enumerate([8, 16, 32, 64]):
            s = d[d.ep == ep].sort_values("gpus_per_host")
            ax.plot(s.gpus_per_host, s.t_total, marker="o", ms=4, lw=1.6,
                    color=SERIES[i], label=f"EP {ep}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32])
        ax.set_xticklabels(["8", "16", "32"])
        ax.set_xlabel("GPUs per host")
        ax.set_title(f"DeepSeek, {gpu}, {bw} Gbit/s", fontsize=9)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("Iteration time (s)")
    axes[0].legend(fontsize=7, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.1))
    fig.savefig(FIG / "moe_hostsize.pdf")


def moe_skew():
    A = pd.read_csv(RES / "moe_bandwidth_ep.csv")
    A = A[(A.world == 1024) & (A["comp_scale"] == 1) & (A.bw_gbit == 200)]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1), sharey=True)
    for ax, gpu in zip(axes, ["H100", "H20"]):
        p = A[A.gpu == gpu].pivot(index="ep", columns="skew", values="t_total")
        x = np.arange(len(p))
        for i, s in enumerate([1.5, 2.0]):
            ax.bar(x + (i - 0.5) * 0.3, (p[s] / p[1.0] - 1) * 100, 0.28,
                   color=SERIES[i], label=f"max/mean load {s}", **BAR)
        ax.set_xticks(x)
        ax.set_xticklabels([f"EP {e}" for e in p.index])
        ax.set_title(f"{gpu}, 200 Gbit/s, 1,024 GPUs", fontsize=9)
    axes[0].set_ylabel("Slowdown vs. even load (%)")
    axes[0].legend(fontsize=7)
    fig.savefig(FIG / "moe_skew.pdf")


if __name__ == "__main__":
    for f in (fig10, fig11b, moe_gain, moe_breakdown, moe_hostsize, moe_skew):
        f()
        print("ok", f.__name__)
