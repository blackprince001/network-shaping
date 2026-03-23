# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "seaborn>=0.13",
#   "matplotlib>=3.8",
#   "pandas>=2.0",
# ]
# ///
"""
Network Shaping Results — Seaborn Plotting Script
==================================================
Handles three CSV types discovered in the data directory:

  1. ab_ai_*_detail.csv   — per-step time-series for 4 scenarios
  2. ab_ai_*_summary.csv  — per-scenario aggregate bar charts
  3. baseline_*mbps.csv   — per-timestamp time-series (single rate)

Usage
-----
  python plot_network_results.py [--data-dir PATH] [--out-dir PATH]

Defaults:
  --data-dir  ./                (current directory, same as the tree shown)
  --out-dir   ./plots/

All plots are saved as high-resolution PNGs inside --out-dir.
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

# ── Aesthetics ──────────────────────────────────────────────────────────────
PALETTE = {
    "baseline_30mbps": "#4C72B0",
    "baseline_50mbps": "#55A868",
    "baseline_70mbps": "#C44E52",
    "ai_agent":        "#DD8452",
    # fallback for baseline_*mbps single-rate files
    "default":         "#8172B2",
}
SCENARIO_LABELS = {
    "baseline_30mbps": "Baseline 30 Mbps",
    "baseline_50mbps": "Baseline 50 Mbps",
    "baseline_70mbps": "Baseline 70 Mbps",
    "ai_agent":        "AI Agent",
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Helpers ──────────────────────────────────────────────────────────────────

def friendly_title(filename: str) -> str:
    """Turn a raw filename stem into a readable plot title."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    # handle double extensions like ab_ai_narrow.csv_detail → ab_ai_narrow_detail
    stem = stem.replace(".csv", "")
    stem = stem.replace("_csv_", "_")
    # decode tokens
    stem = re.sub(r"ab_ai_", "AI vs Baseline — ", stem)
    stem = re.sub(r"baseline_(\d+mbps)", r"Baseline \1", stem, flags=re.I)
    stem = stem.replace("_", " ").title()
    stem = stem.replace("Mbps", " Mbps").replace("  ", " ")
    return stem.strip()


def save(fig: plt.Figure, out_dir: str, stem: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  saved → {path}")


def color_for(scenario: str) -> str:
    return PALETTE.get(scenario, PALETTE["default"])


def label_for(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario)


# ── Plot: AB detail (time-series per scenario) ───────────────────────────────

DETAIL_METRICS = [
    ("throughput_mbps",  "Throughput (Mbps)",       False),
    ("demand_mbps",      "Demand (Mbps)",            False),
    ("queue_bytes",      "Queue Occupancy (bytes)",  True),
    ("drops",            "Packet Drops",             False),
    ("reward",           "Reward",                   False),
]

def plot_ab_detail(csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)
    scenarios = df["scenario"].unique()
    title_base = friendly_title(csv_path)
    stem = os.path.splitext(os.path.basename(csv_path))[0].replace(".csv", "").replace("_csv_", "_")

    metrics = [(col, label, log) for col, label, log in DETAIL_METRICS if col in df.columns]
    n = len(metrics)
    ncols = 2
    nrows = (n + 1) // ncols

    # Landscape: wider than tall
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), sharex=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.suptitle(title_base, fontsize=15, fontweight="bold", y=1.02)

    for i, (ax, (col, ylabel, use_log)) in enumerate(zip(axes_flat, metrics)):
        for scenario in scenarios:
            sub = df[df["scenario"] == scenario].sort_values("step")
            ax.plot(
                sub["step"], sub[col],
                label=label_for(scenario),
                color=color_for(scenario),
                linewidth=1.8,
                alpha=0.88,
            )
        ax.set_ylabel(ylabel)
        if use_log and df[col].max() > 0:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"
        ))
        # only show legend once or in a less intrusive way if needed, but for comparison it's good in each
        ax.legend(loc="upper right", framealpha=0.6, fontsize=9)
        
        # Add x-label to bottom row
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Simulation Step")

    # hide unused subplots
    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    fig.tight_layout()
    save(fig, out_dir, stem + "_timeseries")


# ── Plot: AB summary (bar charts) ────────────────────────────────────────────

SUMMARY_METRICS = [
    ("avg_throughput",  "Avg Throughput (Mbps)"),
    ("avg_demand",      "Avg Demand (Mbps)"),
    ("avg_queue",       "Avg Queue Occupancy (bytes)"),
    ("total_drops",     "Total Packet Drops"),
    ("avg_reward",      "Avg Reward"),
]

def plot_ab_summary(csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)
    title_base = friendly_title(csv_path)
    stem = os.path.splitext(os.path.basename(csv_path))[0].replace(".csv", "").replace("_csv_", "_")

    metrics = [(col, label) for col, label in SUMMARY_METRICS if col in df.columns]
    n = len(metrics)
    ncols = 3 # More landscape
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.suptitle(title_base, fontsize=15, fontweight="bold", y=1.02)

    df["color"] = df["name"].map(lambda s: color_for(s))
    df["label"] = df["name"].map(lambda s: label_for(s))

    for ax, (col, ylabel) in zip(axes_flat, metrics):
        bars = ax.bar(
            df["label"], df[col],
            color=df["color"].tolist(),
            edgecolor="white",
            linewidth=0.8,
            width=0.6,
        )
        # value labels on top of bars
        for bar, val in zip(bars, df[col]):
            height = bar.get_height()
            if abs(height) >= 1e6:
                txt = f"{height/1e6:.2f}M"
            elif abs(height) >= 1e3:
                txt = f"{height/1e3:.1f}k"
            else:
                txt = f"{height:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (abs(height) * 0.02 if height >= 0 else height * 0.02),
                txt,
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=9, color="#333333",
            )
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=12)
        ax.tick_params(axis="x", rotation=15)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"
        ))

    # hide unused subplots
    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    fig.tight_layout()
    save(fig, out_dir, stem + "_summary")


# ── Plot: Baseline single-rate time-series (Combined Support) ─────────────────

BASELINE_METRICS = [
    ("throughput_mbps",       "Throughput (Mbps)"),
    ("queue_occupancy_bytes", "Queue Occupancy (bytes)"),
    ("delay_ms",              "Delay (ms)"),
    ("drop_count",            "Drop Count"),
    ("reward",                "Reward"),
    ("rate_limit_mbps",       "Rate Limit (Mbps)"),
]

def plot_baselines_combined(csv_paths: list[str], out_dir: str):
    """Plots multiple baseline files together in one landscape comparison figure."""
    if not csv_paths: return
    
    # Load all DFs
    dfs = []
    for p in csv_paths:
        tmp = pd.read_csv(p)
        # Extract rate for labeling
        rate_label = re.search(r"(\d+mbps)", os.path.basename(p), re.I)
        rate_str = rate_label.group(1).lower() if rate_label else os.path.basename(p)
        tmp["scenario"] = f"baseline_{rate_str}"
        dfs.append(tmp)
    
    df_all = pd.concat(dfs, ignore_index=True)
    scenarios = df_all["scenario"].unique()
    
    # Determine metrics
    metrics = [(col, label) for col, label in BASELINE_METRICS if col in df_all.columns]
    n = len(metrics)
    ncols = 2
    nrows = (n + 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), sharex=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    
    title = "Baseline Comparison — " + " & ".join([s.replace("baseline_", "").upper() for s in scenarios])
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    
    for i, (ax, (col, ylabel)) in enumerate(zip(axes_flat, metrics)):
        time_col = "timestamp" if "timestamp" in df_all.columns else df_all.columns[0]
        for s in scenarios:
            sub = df_all[df_all["scenario"] == s].sort_values(time_col)
            color = color_for(s)
            # Area Chart: Plot line and fill below
            ax.plot(sub[time_col], sub[col], label=label_for(s), color=color, linewidth=2.0, alpha=0.9)
            ax.fill_between(sub[time_col], sub[col], color=color, alpha=0.15)
        
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"
        ))
        ax.legend(loc="upper right", framealpha=0.6, fontsize=9)
        
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Timestamp (s)")

    # hide unused subplots
    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    fig.tight_layout()
    stem = "combined_baselines_" + "_".join([s.replace("baseline_", "") for s in scenarios])
    save(fig, out_dir, stem)


# ── Discovery & dispatch ──────────────────────────────────────────────────────

def discover_and_plot(data_dir: str, out_dir: str):
    detail_glob   = glob.glob(os.path.join(data_dir, "*detail*.csv"))
    summary_glob  = glob.glob(os.path.join(data_dir, "*summary*.csv"))
    baseline_glob = sorted(glob.glob(os.path.join(data_dir, "baseline_*.csv")))

    # Group baselines for combined plots (at least 2 at a time)
    # Filter out detail/summary from baseline_glob
    baseline_files = [f for f in baseline_glob if "detail" not in f and "summary" not in f]
    baseline_groups = [baseline_files[i:i + 2] for i in range(0, len(baseline_files), 2)]

    print(f"\n{'─'*55}")
    print(f"  Data dir : {data_dir}")
    print(f"  Out dir  : {out_dir}")
    print(f"  Detail   : {len(detail_glob)} files")
    print(f"  Summary  : {len(summary_glob)} files")
    print(f"  Baselines: {len(baseline_files)} files (grouped into {len(baseline_groups)} plots)")
    print(f"{'─'*55}\n")

    for f in sorted(detail_glob):
        print(f"[detail]  {os.path.basename(f)}")
        plot_ab_detail(f, out_dir)

    for f in sorted(summary_glob):
        print(f"[summary] {os.path.basename(f)}")
        plot_ab_summary(f, out_dir)

    for group in baseline_groups:
        print(f"[baseline] Combining: {', '.join([os.path.basename(f) for f in group])}")
        plot_baselines_combined(group, out_dir)

    print(f"\n✅  All plots saved to: {os.path.abspath(out_dir)}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot network shaping results.")
    parser.add_argument("--data-dir", default=".",         help="Directory containing CSV files")
    parser.add_argument("--out-dir",  default="./plots/",  help="Output directory for PNG plots")
    args = parser.parse_args()
    discover_and_plot(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
