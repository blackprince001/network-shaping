"""
src/visualization/plotting.py

Plot evaluation results. Used by the `plot` subcommand in main.py.
Moved from the old plot.py with no behaviour changes.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_comparison(detail_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate comparison time-series, summary bars, and (optionally) agent action plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = detail_df["scenario"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Network Shaping: AI Agent vs Baselines", fontsize=14, fontweight="bold")

    # 1. Throughput vs demand
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

    # 2. Queue occupancy
    ax = axes[0, 1]
    for sc in scenarios:
        d = detail_df[detail_df["scenario"] == sc]
        ax.plot(d["step"], d["queue_bytes"], label=sc, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Queue (bytes)")
    ax.set_title("Queue Occupancy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Drops
    ax = axes[1, 0]
    for sc in scenarios:
        d = detail_df[detail_df["scenario"] == sc]
        ax.plot(d["step"], d["drops"], label=sc, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Drops")
    ax.set_title("Total Drops")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Reward
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

    # Summary bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Summary Comparison", fontsize=14, fontweight="bold")
    names = summary_df["name"].tolist()
    colors = ["#4CAF50" if "ai" in n else "#2196F3" for n in names]

    axes[0].bar(names, summary_df["avg_throughput"], color=colors)
    axes[0].set_ylabel("Avg Throughput (Mbps)")
    axes[0].set_title("Throughput")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(names, summary_df["avg_queue"], color=colors)
    axes[1].set_ylabel("Avg Queue (bytes)")
    axes[1].set_title("Queue (lower = better)")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(names, summary_df["total_reward"], color=colors)
    axes[2].set_ylabel("Total Reward")
    axes[2].set_title("Reward (higher = better)")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = output_dir / "summary_bars.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # Agent action plot (AI only)
    if "ai_agent" in scenarios:
        fig, ax = plt.subplots(figsize=(10, 4))
        ai = detail_df[detail_df["scenario"] == "ai_agent"]
        ax.plot(ai["step"], ai["rate_mbps"], "g-", linewidth=2, label="AI Agent Rate")
        ax.axhline(y=30, color="b", linestyle="--", alpha=0.5, label="Baseline 30 Mbps")
        ax.axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="Baseline 50 Mbps")
        ax.axhline(y=70, color="r", linestyle="--", alpha=0.5, label="Baseline 70 Mbps")
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
