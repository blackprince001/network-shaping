#!/usr/bin/env python3
"""
Plot evaluation results.

Usage:
    uv run python plot.py --detail data/results/evaluation_detail.csv --summary data/results/evaluation_summary.csv
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_comparison(detail_df, summary_df, output_dir):
    """Generate comparison plots from per-step evaluation data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = detail_df["scenario"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Network Shaping: AI Agent vs Baselines", fontsize=14, fontweight="bold")

    # 1. Throughput over time (with demand reference)
    ax = axes[0, 0]
    for sc in scenarios:
        d = detail_df[detail_df["scenario"] == sc]
        ax.plot(d["step"], d["throughput_mbps"], label=sc, linewidth=1.5)
    if "demand_mbps" in detail_df.columns:
        d0 = detail_df[detail_df["scenario"] == scenarios[0]]
        ax.plot(d0["step"], d0["demand_mbps"], "k--", label="demand", linewidth=1, alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Throughput vs Demand")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Queue occupancy over time
    ax = axes[0, 1]
    for sc in scenarios:
        d = detail_df[detail_df["scenario"] == sc]
        ax.plot(d["step"], d["queue_bytes"], label=sc, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Queue (bytes)")
    ax.set_title("Queue Occupancy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Drops over time
    ax = axes[1, 0]
    for sc in scenarios:
        d = detail_df[detail_df["scenario"] == sc]
        ax.plot(d["step"], d["drops"], label=sc, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Drops")
    ax.set_title("Total Drops")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Reward over time
    ax = axes[1, 1]
    for sc in scenarios:
        d = detail_df[detail_df["scenario"] == sc]
        ax.plot(d["step"], d["reward"], label=sc, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward per Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "comparison_timeseries.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # Bar chart summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Summary Comparison", fontsize=14, fontweight="bold")

    names = summary_df["name"].tolist()

    ax = axes[0]
    ax.bar(names, summary_df["avg_throughput"], color=["#4CAF50" if "ai" in n else "#2196F3" for n in names])
    ax.set_ylabel("Avg Throughput (Mbps)")
    ax.set_title("Throughput")
    ax.tick_params(axis='x', rotation=45)

    ax = axes[1]
    ax.bar(names, summary_df["avg_queue"], color=["#4CAF50" if "ai" in n else "#2196F3" for n in names])
    ax.set_ylabel("Avg Queue (bytes)")
    ax.set_title("Queue (lower = better)")
    ax.tick_params(axis='x', rotation=45)

    ax = axes[2]
    ax.bar(names, summary_df["total_reward"], color=["#4CAF50" if "ai" in n else "#2196F3" for n in names])
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward (higher = better)")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    path = output_dir / "summary_bars.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # Rate action plot (only meaningful for AI agent)
    if "ai_agent" in scenarios:
        fig, ax = plt.subplots(figsize=(10, 4))
        ai = detail_df[detail_df["scenario"] == "ai_agent"]
        ax.plot(ai["step"], ai["rate_mbps"], "g-", linewidth=2, label="AI Agent Rate")
        ax.axhline(y=30, color="b", linestyle="--", alpha=0.5, label="Baseline 30Mbps")
        ax.axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="Baseline 50Mbps")
        ax.axhline(y=70, color="r", linestyle="--", alpha=0.5, label="Baseline 70Mbps")
        ax.set_xlabel("Step")
        ax.set_ylabel("TBF Rate (Mbps)")
        ax.set_title("AI Agent Rate Decisions")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = output_dir / "agent_actions.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument("--detail", type=str, required=True, help="Detail CSV from evaluate.py")
    parser.add_argument("--summary", type=str, required=True, help="Summary CSV from evaluate.py")
    parser.add_argument("--output", type=str, default="data/results/plots")
    args = parser.parse_args()

    detail = pd.read_csv(args.detail)
    summary = pd.read_csv(args.summary)
    plot_comparison(detail, summary, args.output)


if __name__ == "__main__":
    main()
