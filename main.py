#!/usr/bin/env python3
"""
CLI entrypoint for the RL Network Shaping project.

All operations go through this single entrypoint. The ns-3 binary path can
be set via the NS3_BINARY environment variable to avoid repeating it.

Subcommands
-----------
  train     Train PPO through curriculum levels (always uses mock simulator)
  infer     Run AI model or static baseline, export per-step CSV
  evaluate  A/B test: AI models vs static baselines in parallel
  plot      Generate comparison plots from evaluate/infer output
  sanity    Quick toll-booth verification using real ns-3
  test      Run all unit tests in src/

Examples
--------
  # Train (mock, fast)
  uv run python main.py train --config configs/curriculum/ --output models/

  # Inference — AI model
  uv run python main.py infer --model models/ppo_curriculum_final.zip \\
      --output data/results/ai_metrics.csv

  # Inference — static baseline at 50 Mbps
  uv run python main.py infer --baseline 50 \\
      --output data/results/baseline_50mbps.csv

  # A/B evaluation
  uv run python main.py evaluate --stress --steps 100 \\
      --model ai_v1:models/ppo_curriculum_final.zip \\
      --output data/results/eval/

  # Plot results
  uv run python main.py plot \\
      --detail data/results/eval/detail.csv \\
      --summary data/results/eval/summary.csv

  # Verify simulator is working
  uv run python main.py sanity

  # Run unit tests
  uv run python main.py test
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.agents.ppo_agent import PPOAgent
from src.environments.ns3_env import DEFAULT_NS3_BINARY, Ns3Env
from src.environments.metrics import MetricsCollector
from src.traffic.cloudflare_loader import load_cloudflare_traffic

# Respect NS3_BINARY env var; fall back to compiled default
_DEFAULT_NS3 = os.environ.get("NS3_BINARY", DEFAULT_NS3_BINARY)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _ns3_arg(p: argparse.ArgumentParser) -> None:
    """Add the shared --ns3-binary argument to a subparser."""
    p.add_argument(
        "--ns3-binary",
        type=str,
        default=_DEFAULT_NS3,
        metavar="PATH",
        help=(
            "Path to compiled ns-3 binary "
            "(default: NS3_BINARY env var or built-in default)"
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="RL Network Shaping — unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train PPO through curriculum levels")
    p_train.add_argument(
        "--config", required=True, metavar="DIR",
        help="Directory containing curriculum YAML files",
    )
    p_train.add_argument(
        "--output", required=True, metavar="DIR",
        help="Directory to save model checkpoints",
    )
    p_train.add_argument(
        "--mock", action=argparse.BooleanOptionalAction, default=True,
        help="Use mock simulator (default: --mock). Pass --no-mock for real ns-3 (very slow)",
    )
    _ns3_arg(p_train)

    # ── infer ────────────────────────────────────────────────────────────────
    p_infer = sub.add_parser("infer", help="Run AI model or static baseline")
    grp = p_infer.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model", metavar="PATH", help="Trained PPO model (.zip)")
    grp.add_argument(
        "--baseline", type=float, metavar="MBPS",
        help="Static baseline rate in Mbps (e.g. 50)",
    )
    p_infer.add_argument(
        "--output", required=True, metavar="CSV",
        help="Output CSV path for per-step metrics",
    )
    p_infer.add_argument(
        "--steps", type=int, default=100,
        help="Number of simulation steps (default: 100)",
    )
    p_infer.add_argument(
        "--traffic", metavar="JSON",
        help="Cloudflare Radar JSON for real traffic demand",
    )
    p_infer.add_argument(
        "--config", metavar="YAML",
        help="Curriculum YAML to override environment settings",
    )
    _ns3_arg(p_infer)

    # ── evaluate ─────────────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "evaluate",
        help="A/B test: AI models vs static baselines (parallel)",
    )
    p_eval.add_argument(
        "--model", dest="models", action="append", default=[], metavar="NAME:PATH",
        help="AI model — repeat for multiple (e.g. --model v1:models/ppo.zip)",
    )
    p_eval.add_argument(
        "--steps", type=int, default=100,
        help="Steps per scenario (default: 100)",
    )
    p_eval.add_argument(
        "--output", default="data/results/evaluation", metavar="DIR",
        help="Output directory; saves detail.csv and summary.csv (default: data/results/evaluation)",
    )
    p_eval.add_argument(
        "--stress", action="store_true",
        help="Use stress traffic (demand swings 30–120 Mbps, forces drops)",
    )
    p_eval.add_argument(
        "--traffic-scale", type=float, default=1.0, metavar="X",
        help="Multiply Cloudflare demand by X (default: 1.0)",
    )
    p_eval.add_argument(
        "--baselines", nargs="+", type=float, default=[30.0, 50.0, 70.0],
        metavar="MBPS",
        help="Static baseline rates in Mbps (default: 30 50 70)",
    )
    p_eval.add_argument(
        "--no-baselines", action="store_true",
        help="Skip static baselines (run AI models only)",
    )
    p_eval.add_argument(
        "--workers", type=int, default=4,
        help="Maximum parallel workers (default: 4)",
    )
    _ns3_arg(p_eval)

    # ── plot ─────────────────────────────────────────────────────────────────
    p_plot = sub.add_parser("plot", help="Generate comparison plots from CSV output")
    p_plot.add_argument(
        "--detail", required=True, metavar="CSV",
        help="Per-step detail CSV (from evaluate or infer)",
    )
    p_plot.add_argument(
        "--summary", required=True, metavar="CSV",
        help="Summary CSV (from evaluate)",
    )
    p_plot.add_argument(
        "--output", default="data/results/plots", metavar="DIR",
        help="Directory to save plots (default: data/results/plots)",
    )

    # ── sanity ───────────────────────────────────────────────────────────────
    p_sanity = sub.add_parser(
        "sanity",
        help="Quick toll-booth verification (demand < rate, demand > rate, etc.)",
    )
    p_sanity.add_argument(
        "--steps", type=int, default=5,
        help="Steps per scenario (default: 5)",
    )
    _ns3_arg(p_sanity)

    # ── test ─────────────────────────────────────────────────────────────────
    sub.add_parser("test", help="Run all unit tests in src/")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_train(args):
    config_dir = Path(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(config_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"Error: no YAML files found in {config_dir}")
        sys.exit(1)

    print(f"Found {len(yaml_files)} curriculum level(s):")
    for f in yaml_files:
        print(f"  - {f.name}")

    if not args.mock:
        print("\nWarning: training with real ns-3 spawns a new binary per step (~1 s each).")
        print(f"         At 50 K+ steps this can take hours. Consider --mock.\n")

    model = None
    checkpoint = output_dir / "checkpoint_latest.zip"
    prev_checkpoint = output_dir / "checkpoint_prev.zip"

    for level_path in yaml_files:
        print(f"\n{'=' * 60}")
        print(f"Training on {level_path.name}")
        print(f"{'=' * 60}")

        cfg = yaml.safe_load(level_path.read_text())
        print(f"Level {cfg['level']}: {cfg['name']}")
        print(f"Topology : {cfg['topology']}")
        print(f"Traffic  : {cfg['traffic']}")
        print(f"Timesteps: {cfg['timesteps']}")
        print(f"Simulator: {'mock' if args.mock else 'real ns-3'}")

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

        env = Ns3Env(ns3_binary=args.ns3_binary, config=env_cfg, mock=args.mock)

        if model is not None:
            model.set_env(env)
        else:
            model = PPOAgent(env, config=cfg.get("agent", {}))

        model.train(total_timesteps=cfg["timesteps"])

        if prev_checkpoint.exists():
            prev_checkpoint.unlink()
        if checkpoint.exists():
            import shutil
            shutil.copy(checkpoint, prev_checkpoint)
        model.save(str(checkpoint))
        print(f"Saved checkpoint to {checkpoint}")

        env.close()

    final = output_dir / "ppo_curriculum_final.zip"
    model.save(str(final))
    print(f"\nTraining complete! Final model saved to {final}")


def cmd_infer(args):
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
        sys.exit(1)

    mode = "baseline" if args.baseline is not None else "ai"
    if mode == "ai":
        print(f"Running AI inference  : {args.model}")
    else:
        print(f"Running baseline      : {args.baseline} Mbps")
    print(f"Steps                 : {args.steps}")
    print(f"ns-3 binary           : {args.ns3_binary}")

    env = Ns3Env(ns3_binary=args.ns3_binary, config=env_cfg, mock=False)

    if args.traffic:
        print(f"Loading traffic profile: {args.traffic}")
        env_cfg["demand_profile"] = list(load_cloudflare_traffic(args.traffic))

    metrics = MetricsCollector()
    obs, _ = env.reset()

    ppo_model = None
    if mode == "ai":
        ppo_model = PPOAgent(env=env, model_path=args.model)

    for step in range(args.steps):
        if mode == "baseline":
            action = env.action_space.sample()
            action[0] = args.baseline
        else:
            action = ppo_model.predict(obs)

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

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.export_csv(str(out_path))
    total_reward = sum(r["reward"] for r in metrics.records)
    print(f"\nInference complete!")
    print(f"Metrics   : {out_path}")
    print(f"Steps     : {args.steps}")
    print(f"Total reward: {total_reward:.2f}")


def cmd_evaluate(args):
    from src.evaluation.ab_test import run_evaluation

    if not os.path.exists(args.ns3_binary):
        print(f"Error: ns-3 binary not found at {args.ns3_binary}")
        print("Build it first: cd ns-3.46.1 && ./ns3 build")
        sys.exit(1)

    # Parse model specs: "name:path" or bare "path"
    models: list[tuple[str, str]] = []
    for spec in args.models:
        parts = spec.split(":", 1)
        if len(parts) == 2:
            models.append((parts[0], parts[1]))
        else:
            models.append((Path(parts[0]).stem, parts[0]))

    baselines = [] if args.no_baselines else args.baselines
    traffic_label = "stress (30–120 Mbps)" if args.stress else f"cloudflare ×{args.traffic_scale}"
    print(f"Traffic  : {traffic_label}")
    print(f"Steps    : {args.steps}")
    print(f"Baselines: {baselines}")
    print(f"Models   : {[n for n, _ in models] or '(none)'}")

    detail_df, summary_df = run_evaluation(
        steps=args.steps,
        ns3_binary=args.ns3_binary,
        models=models,
        baselines=baselines,
        stress=args.stress,
        cloudflare_scale=args.traffic_scale,
        max_workers=args.workers,
    )

    if detail_df.empty:
        print("No scenarios completed.")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "detail.csv"
    summary_path = out_dir / "summary.csv"
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Print summary table
    print(f"\n{'Name':<20} {'Demand':>8} {'Thru':>8} {'Queue':>10} {'MaxQ':>10} {'Drops':>8} {'Reward':>8}")
    print("-" * 78)
    for _, s in summary_df.iterrows():
        print(
            f"{s['name']:<20} {s['avg_demand']:>7.1f}  {s['avg_throughput']:>7.1f}  "
            f"{s['avg_queue']:>9.0f}  {s['max_queue']:>9.0f}  "
            f"{s['total_drops']:>7.0f}  {s['total_reward']:>7.1f}"
        )

    print(f"\nSaved detail  : {detail_path}")
    print(f"Saved summary : {summary_path}")
    print(f"\nPlot: uv run python main.py plot --detail {detail_path} --summary {summary_path}")


def cmd_plot(args):
    import pandas as pd
    from src.visualization.plotting import plot_comparison

    detail = pd.read_csv(args.detail)
    summary = pd.read_csv(args.summary)
    plot_comparison(detail, summary, args.output)
    print(f"\nPlots saved to {args.output}/")


def cmd_sanity(args):
    """Quick toll-booth verification using real ns-3."""
    if not os.path.exists(args.ns3_binary):
        print(f"Error: ns-3 binary not found at {args.ns3_binary}")
        print("Build it first: cd ns-3.46.1 && ./ns3 build")
        sys.exit(1)

    from src.utils.pipe_bridge import Ns3Process

    proc = Ns3Process(args.ns3_binary)

    scenarios = [
        ("demand < rate (no limit)",   100, 500_000,  80, "queue=0, thru~80"),
        ("demand = rate (boundary)",    60, 500_000,  60, "queue≈0, thru~60"),
        ("demand > rate (limit)",       50, 500_000, 100, "queue>0, thru~50"),
        ("demand >> rate (heavy)",      30, 500_000, 100, "queue>>0, thru~30"),
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
        ok = (demand <= rate and q_last < 1_000_000) or (demand > rate and avg_thru < rate * 1.3)
        status = "OK" if ok else "FAIL"

        print(
            f"  {label:<26} {rate:>5}  {demand:>6}  | "
            f"{avg_thru:>7.1f}  {q_last:>9,}  {drops_last:>5}  | {expected}  [{status}]"
        )

    proc.stop()
    print("\nSanity check complete.")


def cmd_test(_args):
    from src.utils.tester import run_all_tests
    ok = run_all_tests(PROJECT_ROOT)
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cmds = {
        "train":    cmd_train,
        "infer":    cmd_infer,
        "evaluate": cmd_evaluate,
        "plot":     cmd_plot,
        "sanity":   cmd_sanity,
        "test":     cmd_test,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
