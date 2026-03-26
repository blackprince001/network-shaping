"""
src/evaluation/ab_test.py

A/B evaluation: run AI models and static baselines against each other,
returning structured DataFrames. Used by the `evaluate` subcommand in main.py.

Consolidates logic from the old evaluate.py and benchmark.py.
"""

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.ns3_env import DEFAULT_NS3_BINARY, Ns3Env
from agents.ppo_agent import PPOAgent

_print_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Traffic generators
# ---------------------------------------------------------------------------

def generate_stress_traffic(steps: int) -> list[float]:
    """
    Generate demand that forces the rate limiter to make real decisions.

    Pattern (cycled):
      Phase 1 — Ramp up 30→100 over 8 steps (pressure builds)
      Phase 2 — Hold 100-110 for 6 steps (drops guaranteed at low rates)
      Phase 3 — Spike to 120 for 2 steps (maximum stress)
      Phase 4 — Crash to 30 for 4 steps (agent should drop rate fast)
      Phase 5 — Gentle wave 40-80 for 10 steps (recovery)
    """
    pattern: list[float] = []
    for i in range(8):
        pattern.append(30.0 + 70.0 * i / 7)
    for i in range(6):
        pattern.append(100.0 + 10.0 * np.sin(i * np.pi / 3))
    pattern.extend([120.0, 120.0])
    pattern.extend([30.0, 30.0, 35.0, 40.0])
    for i in range(10):
        pattern.append(60.0 + 20.0 * np.sin(i * 2 * np.pi / 10))

    return [pattern[i % len(pattern)] for i in range(steps)]


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(env: Ns3Env, action_fn, total_steps: int, label: str) -> pd.DataFrame:
    """Run *total_steps* steps and return a per-step DataFrame."""
    records = []
    obs, _ = env.reset()
    for step in range(total_steps):
        action = action_fn(obs, step)
        obs, reward, terminated, truncated, info = env.step(action)
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
        with _print_lock:
            print(
                f"  [{label}] step {step + 1}/{total_steps}  "
                f"thru={info['throughput_mbps']:.1f}Mbps  "
                f"queue={info['queue_bytes']}  "
                f"drops={info['drops']}  "
                f"reward={reward:.3f}",
                flush=True,
            )
        if terminated or truncated:
            obs, _ = env.reset()

    return pd.DataFrame(records)


def _run_baseline(rate: float, steps: int, ns3_binary: str, env_config: dict, mock: bool):
    label = f"baseline_{rate:.0f}mbps"
    print(f"  Starting {label}...", flush=True)
    env = Ns3Env(ns3_binary=ns3_binary, config=env_config, mock=mock)
    fn = lambda obs, step, r=rate: np.array([r])
    df = run_scenario(env, fn, steps, label)
    env.close()
    print(f"  {label} done", flush=True)
    return label, df


def _run_agent(model_path: str, model_name: str, steps: int, ns3_binary: str, env_config: dict, mock: bool):
    label = model_name
    print(f"  Starting {label}...", flush=True)
    env = Ns3Env(ns3_binary=ns3_binary, config=env_config, mock=mock)
    agent = PPOAgent(env, config={}, model_path=model_path)
    fn = lambda obs, step: agent.predict(obs)
    df = run_scenario(env, fn, steps, label)
    env.close()
    print(f"  {label} done", flush=True)
    return label, df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_BASELINES = [30.0, 50.0, 70.0]


def run_evaluation(
    steps: int,
    ns3_binary: str = DEFAULT_NS3_BINARY,
    models: Optional[list[tuple[str, str]]] = None,
    baselines: Optional[list[float]] = None,
    stress: bool = False,
    cloudflare_scale: float = 1.0,
    mock: bool = False,
    max_workers: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all scenarios in parallel and return (detail_df, summary_df).

    Parameters
    ----------
    models    : list of (name, path) tuples
    baselines : list of fixed rates in Mbps; defaults to [30, 50, 70]
    stress    : use stress traffic instead of Cloudflare
    """
    if baselines is None:
        baselines = DEFAULT_BASELINES
    if models is None:
        models = []

    env_config: dict = {
        "alpha": 1.0,
        "beta": 0.5,
        "gamma": 0.8,
        "delta": 0.0,
        "max_rate_mbps": 100.0,
        "max_drops_norm": 100_000.0,
        "max_episode_steps": steps + 10,
        "cloudflare_scale": cloudflare_scale,
    }
    if stress:
        env_config["demand_profile"] = generate_stress_traffic(steps + 20)

    futures: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for rate in baselines:
            fut = pool.submit(_run_baseline, rate, steps, ns3_binary, env_config, mock)
            futures[fut] = f"baseline_{rate:.0f}mbps"

        for name, path in models:
            if not os.path.exists(path):
                print(f"  Warning: model not found at {path}, skipping.", flush=True)
                continue
            fut = pool.submit(_run_agent, path, name, steps, ns3_binary, env_config, mock)
            futures[fut] = name

        print(f"Running {len(futures)} scenario(s) in parallel…", flush=True)

        results: dict[str, pd.DataFrame] = {}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                label, df = fut.result()
                results[label] = df
            except Exception as exc:
                print(f"  {name} failed: {exc}", flush=True)

    # Build ordered summaries
    ordered_names = [f"baseline_{r:.0f}mbps" for r in baselines] + [n for n, _ in models]
    all_rows: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for name in ordered_names:
        if name not in results:
            continue
        df = results[name]
        all_rows.append(df)
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

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True), pd.DataFrame(summaries)
