#!/usr/bin/env python3
"""
CLI entrypoint for the RL Network Shaping project.

Usage:
    # Train through all curriculum levels (uses mock — real ns-3 too slow for 50K steps)
    uv run python main.py train --config configs/curriculum/ --output models/

    # Benchmark: verify toll booth behavior with real ns-3
    uv run python main.py benchmark

    # Run AI inference with real ns-3
    uv run python main.py infer --model models/ppo_curriculum_final.zip \\
        --output data/results/ai_metrics.csv

    # Run static baseline (bypasses AI entirely)
    uv run python main.py infer --baseline 50 \\
        --output data/results/baseline_metrics.csv

    # Run AI inference with real traffic profile
    uv run python main.py infer --model models/ppo_curriculum_final.zip \\
        --traffic data/traffic/cloudflare_hourly.json \\
        --output data/results/ai_realworld.csv

    # A/B evaluation: AI vs baselines
    uv run python evaluate.py --model models/ppo_curriculum_final.zip --steps 20

    # Plot results
    uv run python plot.py --detail data/results/evaluation_detail.csv \\
        --summary data/results/evaluation_summary.csv
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.ppo_agent import PPOAgent
from src.environments.ns3_env import DEFAULT_NS3_BINARY, Ns3Env
from src.environments.metrics import MetricsCollector
from src.traffic.cloudflare_loader import load_cloudflare_traffic


def parse_args():
    parser = argparse.ArgumentParser(description="RL Network Shaping CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train PPO across curriculum levels"
    )
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to curriculum directory containing YAML files",
    )
    train_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save model checkpoints",
    )
    train_parser.add_argument(
        "--ns3-binary",
        type=str,
        default=DEFAULT_NS3_BINARY,
        help="Path to compiled ns-3 binary",
    )

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference (AI or baseline)")
    infer_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained PPO model (.zip)",
    )
    infer_parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="Static baseline rate in Gbps (skips PPO entirely)",
    )
    infer_parser.add_argument(
        "--ns3-binary",
        type=str,
        default=DEFAULT_NS3_BINARY,
        help="Path to compiled ns-3 binary",
    )
    infer_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV path for metrics",
    )
    infer_parser.add_argument(
        "--traffic",
        type=str,
        default=None,
        help="Path to Cloudflare Radar JSON for real traffic demand",
    )
    infer_parser.add_argument(
        "--steps",
        type=int,
        default=3600,
        help="Number of inference steps to run",
    )
    infer_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a curriculum YAML for environment config",
    )

    # Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark", help="Test toll booth behavior across rate/demand scenarios"
    )
    bench_parser.add_argument(
        "--ns3-binary",
        type=str,
        default=DEFAULT_NS3_BINARY,
        help="Path to compiled ns-3 binary",
    )
    bench_parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Steps per scenario",
    )

    return parser.parse_args()


def train(args):
    config_dir = Path(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(config_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"No YAML files found in {config_dir}")
        return

    print(f"Found {len(yaml_files)} curriculum levels:")
    for f in yaml_files:
        print(f"  - {f.name}")

    model = None
    model_checkpoint = output_dir / "checkpoint_latest.zip"
    prev_checkpoint = output_dir / "checkpoint_prev.zip"

    for level_path in yaml_files:
        print(f"\n{'=' * 60}")
        print(f"Training on {level_path.name}")
        print(f"{'=' * 60}")

        cfg = yaml.safe_load(level_path.read_text())
        print(f"Level {cfg['level']}: {cfg['name']}")
        print(f"Topology: {cfg['topology']}")
        print(f"Traffic: {cfg['traffic']}")
        print(f"Timesteps: {cfg['timesteps']}")

        env_cfg = {
            **cfg.get("reward", {}),
            **cfg.get("agent", {}),
            "max_episode_steps": cfg.get("max_episode_steps", 3600),
            "max_rate_mbps": cfg.get("max_rate_mbps", 100_000),
            "min_rate_mbps": cfg.get("min_rate_mbps", 1.0),
            "max_action_rate_mbps": cfg.get("max_action_rate_mbps", 100.0),
            "max_queue_bytes": cfg.get("max_queue_bytes", 5_242_880),
            "max_drops_norm": cfg.get("max_drops_norm", 100.0),
        }
        if "traffic_file" in cfg:
            env_cfg["traffic_file"] = cfg["traffic_file"]

        # Training always uses mock — real ns-3 respawns per step (seconds each)
        # which makes 50K+ timesteps impractical. Mock is ~1000x faster.
        env = Ns3Env(
            ns3_binary=args.ns3_binary,
            config=env_cfg,
            mock=True,
        )

        if model is not None:
            model.set_env(env)
        else:
            model = PPOAgent(env, config=cfg.get("agent", {}))

        model.train(total_timesteps=cfg["timesteps"])

        if prev_checkpoint.exists():
            prev_checkpoint.unlink()
        if model_checkpoint.exists():
            import shutil

            shutil.copy(model_checkpoint, prev_checkpoint)
        model.save(str(model_checkpoint))
        print(f"Saved checkpoint to {model_checkpoint}")

        env.close()

    final_path = output_dir / "ppo_curriculum_final.zip"
    model.save(str(final_path))
    print(f"\nTraining complete! Final model saved to {final_path}")


def infer(args):
    if args.model and args.baseline:
        print("Error: specify either --model or --baseline, not both")
        return
    if not args.model and args.baseline is None:
        print("Error: must specify either --model or --baseline")
        return

    if args.baseline is not None:
        print(f"Running baseline inference at {args.baseline} Gbps")
        mode = "baseline"
        baseline_rate = args.baseline
    else:
        print(f"Running AI inference with model: {args.model}")
        mode = "ai"

    env_cfg = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        env_cfg = {
            **cfg.get("reward", {}),
            **cfg.get("agent", {}),
            "max_episode_steps": cfg.get("max_episode_steps", 3600),
            "max_rate_mbps": cfg.get("max_rate_mbps", 100_000),
            "min_rate_mbps": cfg.get("min_rate_mbps", 1.0),
            "max_action_rate_mbps": cfg.get("max_action_rate_mbps", 100.0),
            "max_queue_bytes": cfg.get("max_queue_bytes", 5_242_880),
            "max_drops_norm": cfg.get("max_drops_norm", 100.0),
        }

    if not os.path.exists(args.ns3_binary):
        print(f"Error: ns-3 binary not found at {args.ns3_binary}")
        print("Build it first: cd ns-3.46.1 && ./ns3 build")
        return

    print(f"Using ns-3 at {args.ns3_binary}")
    env = Ns3Env(
        ns3_binary=args.ns3_binary,
        config=env_cfg,
        mock=False,
    )

    if args.traffic:
        print(f"Loading traffic profile from {args.traffic}")
        demands = load_cloudflare_traffic(args.traffic)
    else:
        demands = None

    metrics = MetricsCollector()
    obs, _ = env.reset()

    model = None
    baseline_rate = None
    if args.baseline is not None:
        baseline_rate = args.baseline
    if mode == "ai":
        model = PPOAgent(env=env, model_path=args.model)

    for step in range(args.steps):
        if mode == "baseline":
            action = env.action_space.sample()
            action[0] = baseline_rate
        else:
            action = model.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        metrics.record(
            timestamp=float(step),
            throughput_mbps=info["throughput_mbps"],
            delay_ms=info.get("delay_ms", 0.0),
            drop_count=info["drops"],
            queue_occupancy_bytes=info["queue_bytes"],
            reward=reward,
            rate_limit_mbps=info["rate_mbps"],
        )

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    metrics.export_csv(args.output)
    print(f"\nInference complete! Metrics exported to {args.output}")
    print(f"Total steps: {args.steps}")
    print(f"Total reward: {sum(r['reward'] for r in metrics.records):.2f}")


def benchmark(args):
    """Run toll booth scenarios to verify rate limiting behavior."""
    if not os.path.exists(args.ns3_binary):
        print(f"Error: ns-3 binary not found at {args.ns3_binary}")
        print("Build it first: cd ns-3.46.1 && ./ns3 build")
        return

    from src.utils.pipe_bridge import Ns3Process

    proc = Ns3Process(args.ns3_binary)

    scenarios = [
        ("demand<rate (no limit)",  100, 500000,  80, "queue=0, thru~80"),
        ("demand>rate (limit)",      50, 500000, 100, "queue>0, thru~50"),
        ("demand>>rate (heavy)",     30, 500000, 100, "queue>>0, thru~30"),
        ("demand=rate (boundary)",   60, 500000,  60, "queue=0, thru~60"),
    ]

    print(f"\n{'Scenario':<28} {'Rate':>6} {'Demand':>7} | {'Thru':>8} {'Queue':>10} {'Drops':>6} | Expected")
    print("-" * 100)

    for label, rate, burst, demand, expected in scenarios:
        thru_sum = 0
        q_last = 0
        drops_last = 0
        for _ in range(args.steps):
            q, t, d = proc.step(rate_mbps=rate, burst_bytes=burst, demand_mbps=demand)
            thru_sum += t
            q_last = q
            drops_last = d

        avg_thru = thru_sum / args.steps

        status = "OK" if (
            (demand <= rate and q_last < 1_000_000) or
            (demand > rate and avg_thru < rate * 1.3)
        ) else "FAIL"

        print(f"  {label:<26} {rate:>5}  {demand:>6}  | "
              f"{avg_thru:>7.1f}  {q_last:>9,}  {drops_last:>5}  | {expected}  [{status}]")

    proc.stop()
    print("\nBenchmark complete.")


def main():
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    elif args.command == "benchmark":
        benchmark(args)
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
