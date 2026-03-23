# RL-Based Network Traffic Shaping

This project implements a Reinforcement Learning (RL) agent that dynamically manages network traffic shaping using the **ns-3 network simulator**. By training a PPO (Proximal Policy Optimization) agent, the system optimizes Token Bucket Filter (TBF) rates in real-time to balance throughput, queue depth, and packet loss.

---

## 🚀 Problem Statement

Traditional traffic shaping relies on static rate limiters (TBF) with fixed bandwidth caps. These fail to adapt to fluctuating real-world traffic patterns, leading to:
- **Packet drops** during demand spikes.
- **Wasted bandwidth** during off-peak hours.
- **Latency spikes** due to bufferbloat.

**Our Goal:** Build an adaptive RL agent that outperforms static baselines on all key networking metrics.

## 🛠 Architecture

The system uses a Python-based RL environment (Gymnasium) that communicates with an ns-3 C++ simulation via stdin/stdout pipes.

```text
┌─────────────┐      stdin/stdout       ┌──────────────────┐
│  PPO Agent  │ ◄──────────────────────► │  ns-3 Simulator  │
│  (Python)   │   STEP,<rate>,<burst>,   │  (C++)           │
│             │   <demand>               │                  │
│  Gymnasium  │   <queue>,<throughput>,   │  TBF Queue Disc  │
│  Environment│   <drops>                │  on bottleneck   │
└─────────────┘                          └──────────────────┘
```

### Key Components
- **`main.py`**: CLI for training and inference.
- **`src/environments/ns3_env.py`**: Gymnasium wrapper for the ns-3 simulator.
- **`ns-3.36/contrib/network-shaping/`**: C++ simulation module implementing the bottleneck topology.
- **`src/traffic/cloudflare_loader.py`**: Loader for real-world Cloudflare Radar traffic profiles.

## Quick Start

### Prerequisites
- Python 3.13+
- `uv` package manager
- ns-3.36 (pre-built)

### Training
```bash
# Train through curriculum levels (Mock mode for fast testing)
uv run main.py train --config configs/curriculum/ --output models/ --mock

# Train with real ns-3
uv run main.py train --config configs/curriculum/ --output models/ --ns3-binary path/to/ns3
```

### Evaluation
```bash
# Compare AI vs Static Baselines
uv run evaluate.py --model models/ppo_curriculum_final.zip --steps 50 --mock

# Generate visual reports
uv run plot.py --detail data/results/evaluation_detail.csv --summary data/results/evaluation_summary.csv
```

## Further Documentation

For more detailed information, see the `docs/` directory:
- [Technical Specification](docs/spec.md): Detailed logic and module specs.
- [CLI Reference](docs/commands.md): Comprehensive list of available commands.
- [Debug Report](docs/debug_report.md): Historical debugging and troubleshooting notes.

---

##  Design Decisions

| Decision | Rationale |
|----------|-----------|
| **PPO Agent** | Handles continuous action spaces (rate control) stably and efficiently. |
| **stdin/stdout Bridge** | Simple, low-overhead communication between Python and C++. |
| **Curriculum Training** | Progressively builds agent capability from basic to real-world scenarios. |
| **Mock Simulator** | Enables rapid iteration without the 30-min ns-3 compilation overhead. |
