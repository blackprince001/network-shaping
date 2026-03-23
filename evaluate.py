#!/usr/bin/env python3
"""
A/B Evaluation: AI agent vs static baselines (parallel execution).

Usage:
    # Default Cloudflare traffic
    uv run python evaluate.py --steps 100

    # Stress test — demand swings 30-110 Mbps, forces drops
    uv run python evaluate.py --stress --steps 100

    # Scaled Cloudflare — push demand 1.5x higher
    uv run python evaluate.py --traffic-scale 1.5 --steps 100

    # With AI agent
    uv run python evaluate.py --model models/ppo_curriculum_final.zip --stress --steps 100
"""

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.ppo_agent import PPOAgent
from src.environments.ns3_env import DEFAULT_NS3_BINARY, Ns3Env

_print_lock = threading.Lock()


def generate_stress_traffic(steps: int) -> list[float]:
    """
    Generate demand that forces the rate limiter to work.

    Pattern per step (cycled):
      - Ramp up 30→100 over 8 steps (pressure builds)
      - Hold at 100-110 for 6 steps (drops guaranteed at low rates)
      - Spike to 120 for 2 steps (maximum stress)
      - Crash down to 30 for 4 steps (agent should lower rate fast)
      - Gentle wave 40-80 for 10 steps (recovery)

    This forces real decisions:
      - Baseline 30: constant drops during high phases
      - Baseline 70: wastes capacity during low phases, still drops at spikes
      - Baseline 50: mediocre everywhere
      - AI agent: should adapt up during pressure, down during calm
    """
    pattern = []

    # Phase 1: Ramp up
    for i in range(8):
        pattern.append(30.0 + (70.0 * i / 7))

    # Phase 2: Hold high
    for i in range(6):
        pattern.append(100.0 + 10.0 * np.sin(i * np.pi / 3))

    # Phase 3: Spike
    pattern.extend([120.0, 120.0])

    # Phase 4: Crash
    pattern.extend([30.0, 30.0, 35.0, 40.0])

    # Phase 5: Gentle wave
    for i in range(10):
        pattern.append(60.0 + 20.0 * np.sin(i * 2 * np.pi / 10))

    # Repeat to fill requested steps
    result = []
    for i in range(steps):
        result.append(pattern[i % len(pattern)])
    return result


def run_scenario(env, action_fn, total_steps, label):
    """Run N steps collecting per-step metrics. Returns DataFrame."""
    import threading
    records = []
    obs, _ = env.reset()
    for step in range(total_steps):
        action = action_fn(obs, step)
        obs, reward, term, trunc, info = env.step(action)
        records.append(
            {
                "scenario": label,
                "step": step,
                "demand_mbps": info["demand_mbps"],
                "queue_bytes": info["queue_bytes"],
                "throughput_mbps": info["throughput_mbps"],
                "drops": info["drops"],
                "rate_mbps": float(action[0]),
                "reward": reward,
            }
        )
        # Log every step with thread-safe output
        with _print_lock:
            print(
                f"  [{label}] step {step+1}/{total_steps}  "
                f"thru={info['throughput_mbps']:.1f}Mbps  "
                f"queue={info['queue_bytes']}  "
                f"drops={info['drops']}  "
                f"reward={reward:.3f}",
                flush=True,
            )
        if term or trunc:
            obs, _ = env.reset()
    return pd.DataFrame(records)


def run_baseline(rate, steps, ns3_binary, env_config):
    """Run a single baseline scenario."""
    label = f"baseline_{rate:.0f}mbps"
    print(f"  Starting {label}...", flush=True)
    env = Ns3Env(ns3_binary=ns3_binary, config=env_config, mock=False)
    fn = lambda obs, step, r=rate: np.array([r])
    df = run_scenario(env, fn, steps, label)
    env.close()
    print(f"  {label} done", flush=True)
    return label, df


def run_agent(model_path, steps, ns3_binary, env_config):
    """Run the AI agent scenario."""
    label = "ai_agent"
    print(f"  Starting {label}...", flush=True)
    env = Ns3Env(ns3_binary=ns3_binary, config=env_config, mock=False)
    agent = PPOAgent(env, config={}, model_path=model_path)
    fn = lambda obs, step: agent.predict(obs)
    df = run_scenario(env, fn, steps, label)
    env.close()
    print(f"  {label} done", flush=True)
    return label, df


def main():
    parser = argparse.ArgumentParser(description="A/B Evaluation (parallel)")
    parser.add_argument("--model", type=str, help="Path to trained PPO model")
    parser.add_argument("--steps", type=int, default=100, help="Steps per scenario")
    parser.add_argument("--output", type=str, default="data/results/evaluation")
    parser.add_argument("--ns3-binary", type=str, default=DEFAULT_NS3_BINARY)
    parser.add_argument("--stress", action="store_true",
                        help="Use stress traffic (demand swings 30-120 Mbps, forces drops)")
    parser.add_argument("--traffic-scale", type=float, default=1.0,
                        help="Multiply Cloudflare demand by this factor (e.g. 1.5)")
    args = parser.parse_args()

    if not os.path.exists(args.ns3_binary):
        print(f"Error: ns-3 binary not found at {args.ns3_binary}")
        print("Build it first: cd ns-3.46.1 && ./ns3 build")
        return

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build demand profile
    if args.stress:
        demand_profile = generate_stress_traffic(args.steps + 20)
        traffic_label = "stress (30-120 Mbps swings)"
    else:
        demand_profile = None  # use default Cloudflare
        traffic_label = f"cloudflare (scale={args.traffic_scale})x"

    env_config = {
        "alpha": 1.0,
        "beta": 0.5,
        "gamma": 0.8,
        "delta": 0.0,
        "max_rate_mbps": 100.0,
        "max_drops_norm": 100000.0,
        "max_episode_steps": args.steps + 10,
        "cloudflare_scale": args.traffic_scale,
    }
    if demand_profile is not None:
        env_config["demand_profile"] = demand_profile

    print(f"Traffic: {traffic_label}")
    print(f"Steps: {args.steps}")

    # Collect all tasks
    futures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        # Submit baselines
        for rate in [30.0, 50.0, 70.0]:
            fut = pool.submit(run_baseline, rate, args.steps, args.ns3_binary, env_config)
            futures[fut] = f"baseline_{rate:.0f}mbps"

        # Submit AI agent
        if args.model and os.path.exists(args.model):
            fut = pool.submit(run_agent, args.model, args.steps, args.ns3_binary, env_config)
            futures[fut] = "ai_agent"
        elif args.model:
            print(f"Model not found: {args.model}")

        print(f"Running {len(futures)} scenarios in parallel...", flush=True)

        # Collect results as they complete
        results = {}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                label, df = fut.result()
                results[label] = df
            except Exception as e:
                print(f"  {name} failed: {e}", flush=True)

    # Build summaries in submission order
    all_step_data = []
    summaries = []
    for name in [f"baseline_{r:.0f}mbps" for r in [30.0, 50.0, 70.0]] + (["ai_agent"] if args.model and os.path.exists(args.model) else []):
        if name not in results:
            continue
        df = results[name]
        all_step_data.append(df)
        summaries.append(
            {
                "name": name,
                "avg_throughput": df["throughput_mbps"].mean(),
                "avg_demand": df["demand_mbps"].mean(),
                "avg_queue": df["queue_bytes"].mean(),
                "max_queue": df["queue_bytes"].max(),
                "total_drops": df["drops"].iloc[-1] if len(df) > 0 else 0,
                "avg_reward": df["reward"].mean(),
                "total_reward": df["reward"].sum(),
            }
        )

    if not all_step_data:
        print("No scenarios completed.")
        return

    # Save
    detail_df = pd.concat(all_step_data, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    detail_path = str(args.output) + "_detail.csv"
    summary_path = str(args.output) + "_summary.csv"
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Print table
    print(f"\n{'Name':<20} {'Demand':>8} {'Thru':>8} {'Queue':>10} {'MaxQ':>10} {'Drops':>8} {'Reward':>8}")
    print("-" * 78)
    for s in summaries:
        print(
            f"{s['name']:<20} {s['avg_demand']:>7.1f}  {s['avg_throughput']:>7.1f}  "
            f"{s['avg_queue']:>9.0f}  {s['max_queue']:>9.0f}  "
            f"{s['total_drops']:>7.0f}  {s['total_reward']:>7.1f}"
        )

    print(f"\nSaved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(
        f"\nTo plot: uv run python plot.py --detail {detail_path} --summary {summary_path}"
    )


if __name__ == "__main__":
    main()
