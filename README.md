# RL-Based Network Traffic Shaping

This project implements a Reinforcement Learning (RL) agent that dynamically manages network traffic shaping using the **ns-3 network simulator**. By training a PPO (Proximal Policy Optimization) agent, the system optimizes Token Bucket Filter (TBF) rates in real-time to balance throughput, queue depth, and packet loss.

## Problem Statement

Traditional traffic shaping relies on static rate limiters (TBF) with fixed bandwidth caps. These fail to adapt to fluctuating real-world traffic patterns, leading to:
- **Packet drops** during demand spikes.
- **Wasted bandwidth** during off-peak hours.
- **Latency spikes** due to bufferbloat.

**Goal:** Build an adaptive RL agent that outperforms static baselines on all key networking metrics.

## 🛠 Architecture

The system uses a Python-based RL environment (Gymnasium) that communicates with an ns-3 C++ simulation via stdin/stdout pipes.

```text
┌─────────────┐      stdin/stdout       ┌──────────────────┐
│  PPO Agent  │ ◄──────────────────────► │  ns-3 Simulator  │
│  (Python)   │   STEP,<rate>,<burst>,   │  (C++)           │
│             │   <demand>               │                  │
│  Gymnasium  │   <queue>,<throughput>,  │  TBF Queue Disc  │
│  Environment│   <drops>                │  on bottleneck   │
└─────────────┘                          └──────────────────┘
```

### Key Components
- **`main.py`**: Unified CLI for all operations (train, infer, evaluate, plot, sanity, test).
- **`src/environments/ns3_env.py`**: Gymnasium wrapper for the ns-3 simulator.
- **`src/evaluation/ab_test.py`**: A/B testing — AI models vs static baselines.
- **`src/visualization/plotting.py`**: Comparison plots from evaluation results.
- **`ns-3.36/contrib/network-shaping/`**: C++ simulation module.
- **`src/traffic/cloudflare_loader.py`**: Loader for Cloudflare Radar traffic profiles.

## Quick Start

### Prerequisites
- Python 3.13+
- `uv` package manager
- ns-3.46 (pre-built, path in `NS3_BINARY` env var or `docs/commands.md`)

### Training
```bash
# Train through curriculum levels (mock simulator — fast)
uv run python main.py train --config configs/curriculum/ --output models/

# Train with real ns-3 (very slow — ~1 s/step)
uv run python main.py train --config configs/curriculum/ --output models/ --no-mock
```

### Inference
```bash
# AI agent
uv run python main.py infer --model models/ppo_curriculum_final.zip \
    --output data/results/ai_metrics.csv --steps 100

# Static baseline at 50 Mbps
uv run python main.py infer --baseline 50 \
    --output data/results/baseline_50mbps.csv --steps 100
```

### A/B Evaluation
```bash
# Stress test — baselines only
uv run python main.py evaluate --stress --steps 100 --output data/results/eval/

# With AI agent
uv run python main.py evaluate --stress --steps 100 \
    --model ai_v1:models/ppo_curriculum_final.zip \
    --output data/results/eval/
```

### Plot Results
```bash
uv run python main.py plot \
    --detail data/results/eval/detail.csv \
    --summary data/results/eval/summary.csv
```

### Other Commands
```bash
# Verify simulator toll-booth behavior
uv run python main.py sanity

# Run unit tests
uv run python main.py test
```

## Further Documentation

- [CLI Reference](docs/commands.md): All subcommands and flags.
- [Technical Specification](docs/spec.md): Detailed logic and module specs.
- [Debug Report](docs/debug_report.md): Historical debugging notes.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **PPO Agent** | Handles continuous action spaces (rate control) stably. |
| **stdin/stdout Bridge** | Simple, low-overhead Python ↔ C++ communication. |
| **Curriculum Training** | Progressively builds agent capability from basic to real-world. |
| **Mock Simulator** | Enables rapid iteration without 30-min ns-3 compilation overhead. |
| **Unified CLI** | Single `main.py` entrypoint — no scattered scripts to remember. |
