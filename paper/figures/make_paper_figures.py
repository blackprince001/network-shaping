# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "pandas>=2.0", "numpy>=1.24"]
# ///
"""Compact, IEEE-column-ready figures for the paper.

Reads CSVs from data/results/ and writes paper/figures/*.pdf
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "data", "results")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# IEEE single-column ≈ 3.5 in; double-column ≈ 7.16 in. Use small fonts.
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C_BASE = {
    "baseline_30mbps": "#4C72B0",
    "baseline_50mbps": "#2CA02C",
    "baseline_70mbps": "#C44E52",
    "ai_agent": "#DD8452",
}
LBL = {
    "baseline_30mbps": "30 Mbps",
    "baseline_50mbps": "50 Mbps",
    "baseline_70mbps": "70 Mbps",
    "ai_agent": "RL Agent",
}


def fmt_int(x, _):
    if abs(x) >= 1e6: return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3: return f"{x/1e3:.0f}k"
    return f"{x:.0f}"


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Static baselines — failure modes side by side (single column-pair)
def fig_static_baselines():
    rates = [30, 50, 70, 80]
    dfs = {r: pd.read_csv(os.path.join(RES, f"baseline_{r}mbps.csv")) for r in rates}

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0))
    cmap = {30: "#4C72B0", 50: "#2CA02C", 70: "#C44E52", 80: "#7B5BA6"}

    ax = axes[0]
    for r, df in dfs.items():
        ax.plot(df["timestamp"], df["throughput_mbps"],
                color=cmap[r], label=f"{r} Mbps cap", alpha=0.9)
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_xlabel("Time (s)")
    ax.set_title("(a) Throughput")
    ax.legend(loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.2)

    ax = axes[1]
    for r, df in dfs.items():
        ax.plot(df["timestamp"], df["queue_occupancy_bytes"] / 1e6,
                color=cmap[r], label=f"{r}", alpha=0.9)
    ax.set_ylabel("Queue (MB)")
    ax.set_xlabel("Time (s)")
    ax.set_title("(b) Queue occupancy")

    ax = axes[2]
    for r, df in dfs.items():
        ax.plot(df["timestamp"], df["drop_count"].cumsum(),
                color=cmap[r], label=f"{r}", alpha=0.9)
    ax.set_ylabel("Cumulative drops")
    ax.set_xlabel("Time (s)")
    ax.set_title("(c) Packet drops")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_int))

    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig_static_baselines.pdf"))
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: AI agent vs baselines — time series (wide variant)
def fig_ab_timeseries(variant: str, fname: str):
    df = pd.read_csv(os.path.join(RES, f"ab_ai_{variant}.csv_detail.csv"))
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 4.4), sharex=True)

    metrics = [
        ("throughput_mbps", "Throughput (Mbps)"),
        ("queue_bytes", "Queue (MB)"),
        ("drops", "Drops/step"),
    ]
    for ax, (col, ylabel) in zip(axes, metrics):
        for s in df["scenario"].unique():
            sub = df[df["scenario"] == s].sort_values("step")
            y = sub[col].values
            if col == "queue_bytes":
                y = y / 1e6
            ax.plot(sub["step"], y, color=C_BASE.get(s, "#999"),
                    label=LBL.get(s, s), alpha=0.9)
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("Simulation step (s)")
    axes[0].legend(loc="upper right", ncol=4, columnspacing=0.8,
                   handlelength=1.2, frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Summary bars (Wide vs Narrow side-by-side)
def fig_summary_compare():
    wide = pd.read_csv(os.path.join(RES, "ab_ai_wide.csv_summary.csv"))
    narrow = pd.read_csv(os.path.join(RES, "ab_ai_narrow.csv_summary.csv"))

    # Build a long frame with variant label
    wide = wide.copy(); wide["variant"] = "Wide"
    narrow = narrow.copy(); narrow["variant"] = "Narrow"
    df = pd.concat([wide, narrow], ignore_index=True)
    order = ["baseline_30mbps", "baseline_50mbps", "baseline_70mbps", "ai_agent"]

    metrics = [
        ("avg_throughput", "Avg throughput (Mbps)", 1.0),
        ("avg_queue", "Avg queue (MB)", 1e-6),
        ("total_drops", "Total drops", 1.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3))
    width = 0.38
    x = np.arange(len(order))
    for ax, (col, ylabel, scale) in zip(axes, metrics):
        for i, variant in enumerate(["Wide", "Narrow"]):
            sub = df[df["variant"] == variant].set_index("name").reindex(order)
            offset = (i - 0.5) * width
            colors = [C_BASE[n] for n in order]
            # darken AI bar to distinguish variant via hatch
            bars = ax.bar(x + offset, sub[col].values * scale,
                          width=width, color=colors,
                          edgecolor="black", linewidth=0.4,
                          hatch="" if variant == "Wide" else "//",
                          label=variant)
        ax.set_xticks(x)
        ax.set_xticklabels([LBL[n] for n in order], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        if col == "total_drops":
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_int))

    # custom legend for variants
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label="Wide [1,100]"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Narrow [40,90]"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper left", frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig_summary_compare.pdf"))
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: Stress test — bars at 100% and 200% intensity (narrow agent)
def fig_stress_summary():
    files = {
        "1×": "ab_ai_narrow.csv_summary.csv",
        "2×": "ab_ai_narrow-intense100.csv_summary.csv",
        "3×": "ab_ai_narrow-intense200.csv_summary.csv",
    }
    order = ["baseline_30mbps", "baseline_50mbps", "baseline_70mbps", "ai_agent"]
    frames = []
    for k, f in files.items():
        d = pd.read_csv(os.path.join(RES, f))
        d["intensity"] = k
        frames.append(d)
    df = pd.concat(frames)

    metrics = [
        ("avg_throughput", "Avg throughput (Mbps)", 1.0),
        ("total_drops", "Total drops", 1.0),
        ("avg_reward", "Avg reward", 1.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3))
    intensities = list(files.keys())
    width = 0.26
    x = np.arange(len(order))

    for ax, (col, ylabel, scale) in zip(axes, metrics):
        for i, intensity in enumerate(intensities):
            sub = df[df["intensity"] == intensity].set_index("name").reindex(order)
            offset = (i - 1) * width
            colors = [C_BASE[n] for n in order]
            ax.bar(x + offset, sub[col].values * scale, width=width,
                   color=colors, edgecolor="black", linewidth=0.4,
                   alpha=[1.0, 0.7, 0.45][i])
        ax.set_xticks(x)
        ax.set_xticklabels([LBL[n] for n in order], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        if col == "total_drops":
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_int))

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="grey", alpha=1.0, label="1× demand"),
        Patch(facecolor="grey", alpha=0.7, label="2× demand"),
        Patch(facecolor="grey", alpha=0.45, label="3× demand"),
    ]
    axes[1].legend(handles=legend_handles, loc="upper right", frameon=False, ncol=3,
                   columnspacing=0.6, handlelength=1.0)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig_stress_summary.pdf"))
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: Cloudflare diurnal demand profile (used to motivate non-stationarity)
def fig_cloudflare_profile():
    import json
    p = os.path.join(ROOT, "data", "traffic", "cloudflare_hourly.json")
    if not os.path.exists(p):
        return
    with open(p) as fh:
        data = json.load(fh)
    series = data["result"]["main"]["values"]
    y = np.array([float(v) for v in series], dtype=float)
    # normalize to 0..100% of peak
    y = 100.0 * y / max(y.max(), 1.0)
    t = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(7.16, 1.9))
    ax.plot(t, y, color="#2A4D8F", linewidth=0.9)
    ax.fill_between(t, y, color="#2A4D8F", alpha=0.15)
    ax.set_xlabel("Hour (≈ 22 days of hourly samples)")
    ax.set_ylabel("Demand (% of peak)")
    ax.set_xlim(0, len(y))
    ax.set_ylim(0, 105)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig_cloudflare_profile.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_static_baselines()
    fig_ab_timeseries("wide", "fig_ab_wide_timeseries.pdf")
    fig_ab_timeseries("narrow", "fig_ab_narrow_timeseries.pdf")
    fig_summary_compare()
    fig_stress_summary()
    fig_cloudflare_profile()
    print("OK", os.listdir(OUT))
