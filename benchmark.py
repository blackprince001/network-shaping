#!/usr/bin/env python3
"""
Benchmark: AI models vs static baselines across demand profiles (ns-3).

Usage:
    # Cloudflare demand (default, scale=1.0)
    uv run python benchmark.py --steps 50 \
        --model v1:models/v1_wide/ppo_curriculum_final.zip \
        --model v2:models/v2_narrow/ppo_curriculum_final.zip

    # Cloudflare demand at higher scale
    uv run python benchmark.py --steps 50 --cloudflare-scale 1.5 \
        --model v1:models/v1_wide/ppo_curriculum_final.zip

    # Models only (no baselines)
    uv run python benchmark.py --steps 50 --no-baselines \
        --model v1:models/v1_wide/ppo_curriculum_final.zip
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.ppo_agent import PPOAgent
from src.environments.ns3_env import DEFAULT_NS3_BINARY, Ns3Env

BASELINE_RATES = [30.0, 50.0, 70.0]

REWARD_CONFIG = {
    "alpha": 1.0,
    "beta": 0.5,
    "gamma": 0.8,
    "delta": 0.0,
    "max_rate_mbps": 100.0,
    "max_drops_norm": 100000.0,
}


def run_scenario(env, action_fn, total_steps, label):
    """Run N steps collecting per-step metrics."""
    records = []
    obs, _ = env.reset()
    for step in range(total_steps):
        action = action_fn(obs, step)
        obs, reward, term, trunc, info = env.step(action)
        records.append({
            "scenario": label,
            "step": step,
            "queue_bytes": info["queue_bytes"],
            "throughput_mbps": info["throughput_mbps"],
            "drops": info["drops"],
            "rate_mbps": float(action[0]),
            "reward": reward,
        })
        if (step + 1) % 10 == 0:
            print(f"  {step + 1}/{total_steps}", end=" ", flush=True)
        if term or trunc:
            obs, _ = env.reset()
    print()
    return pd.DataFrame(records)


def make_env_config(demand_mode, steps, cloudflare_scale=None):
    """Build env config for a demand mode."""
    cfg = {
        **REWARD_CONFIG,
        "max_episode_steps": steps + 10,
        "stochastic_demand": demand_mode == "stochastic",
    }
    if demand_mode == "cloudflare":
        cfg["traffic_file"] = "data/traffic/cloudflare_hourly.json"
    if cloudflare_scale is not None:
        cfg["cloudflare_scale"] = cloudflare_scale
    return cfg


def run_benchmark(args):
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    demand_modes = ["cloudflare"]

    all_details = []
    all_summaries = []

    for mode in demand_modes:
        print(f"\n{'=' * 50}")
        print(f"Demand profile: {mode}")
        print(f"{'=' * 50}")

        env_config = make_env_config(mode, args.steps, args.cloudflare_scale)
        env_config["max_episode_steps"] = args.steps + 10

        # Static baselines
        if not args.no_baselines:
            for rate in BASELINE_RATES:
                label = f"{mode}/baseline_{rate:.0f}mbps"
                print(f"Running {label}...", flush=True)
                env = Ns3Env(ns3_binary=args.ns3_binary, config=env_config, mock=False)
                fn = lambda obs, step, r=rate: np.array([r])
                df = run_scenario(env, fn, args.steps, label)
                env.close()
                all_details.append(df)
                all_summaries.append({
                    "demand": mode,
                    "name": f"baseline_{rate:.0f}mbps",
                    "avg_throughput": df["throughput_mbps"].mean(),
                    "avg_queue": df["queue_bytes"].mean(),
                    "total_drops": df["drops"].iloc[-1] if len(df) > 0 else 0,
                    "avg_reward": df["reward"].mean(),
                    "total_reward": df["reward"].sum(),
                })

        # AI models
        for model_spec in args.models:
            parts = model_spec.split(":", 1)
            if len(parts) == 2:
                model_name, model_path = parts
            else:
                model_name = Path(parts[0]).stem
                model_path = parts[0]

            label = f"{mode}/{model_name}"
            print(f"Running {label}...", flush=True)
            try:
                env = Ns3Env(ns3_binary=args.ns3_binary, config=env_config, mock=False)
                agent = PPOAgent(env, config={}, model_path=model_path)
                fn = lambda obs, step: agent.predict(obs)
                df = run_scenario(env, fn, args.steps, label)
                env.close()
                all_details.append(df)
                all_summaries.append({
                    "demand": mode,
                    "name": model_name,
                    "avg_throughput": df["throughput_mbps"].mean(),
                    "avg_queue": df["queue_bytes"].mean(),
                    "total_drops": df["drops"].iloc[-1] if len(df) > 0 else 0,
                    "avg_reward": df["reward"].mean(),
                    "total_reward": df["reward"].sum(),
                })
            except Exception as e:
                print(f"  {model_name} failed: {e}")

    # Save
    if not all_details:
        print("No scenarios completed.")
        return

    detail_df = pd.concat(all_details, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)

    detail_path = str(args.output) + "_detail.csv"
    summary_path = str(args.output) + "_summary.csv"
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Print table per demand mode
    for mode in demand_modes:
        subset = summary_df[summary_df["demand"] == mode]
        if subset.empty:
            continue
        print(f"\n--- {mode} ---")
        print(f"{'Name':<20} {'Thru(Mbps)':>10} {'Queue':>10} {'Drops':>8} {'Reward':>8}")
        print("-" * 60)
        for _, s in subset.iterrows():
            print(f"{s['name']:<20} {s['avg_throughput']:>10.1f} {s['avg_queue']:>10.0f} "
                  f"{s['total_drops']:>8.0f} {s['total_reward']:>8.1f}")

    print(f"\nSaved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"\nTo plot: uv run python plot.py --detail {detail_path} --summary {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark AI models vs baselines")
    parser.add_argument("--model", dest="models", action="append", default=[],
                        help="Model spec: name:path (e.g., v1:models/v1/ppo.zip)")
    parser.add_argument("--steps", type=int, default=50, help="Steps per scenario")
    parser.add_argument("--output", type=str, default="data/results/benchmark")
    parser.add_argument("--ns3-binary", type=str, default=DEFAULT_NS3_BINARY)
    parser.add_argument("--no-baselines", action="store_true", help="Skip static baselines")
    parser.add_argument("--cloudflare-scale", type=float, default=None,
                        help="Also run Cloudflare demand at this scale (e.g., 1.2)")
    args = parser.parse_args()

    if not args.models:
        print("No models specified. Use --model name:path")
        print("Example: --model v1:models/v1_wide/ppo_curriculum_final.zip")
        sys.exit(1)

    run_benchmark(args)


if __name__ == "__main__":
    main()
