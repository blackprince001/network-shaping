# Network Shaping RL — CLI Reference

All operations use a single entrypoint: `main.py`.

## Prerequisites

```bash
# Install Python dependencies
uv sync

# Build ns-3 (first time or after C++ changes)
cd /home/blackprince/Downloads/ns-3.46.1
./ns3 configure --enable-examples
./ns3 build

# Optional: set env var so you never need to pass --ns3-binary
export NS3_BINARY=/home/blackprince/Downloads/ns-3.46.1/build/scratch/ns3.46.1-network-shaping-simulation-optimized
```

---

## `train` — Train the PPO Agent

```bash
uv run python main.py train --config <DIR> --output <DIR> [--mock | --no-mock]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config DIR` | required | Directory of curriculum YAML files |
| `--output DIR` | required | Directory to save model checkpoints |
| `--mock` / `--no-mock` | `--mock` | Use mock or real ns-3 simulator |
| `--ns3-binary PATH` | `$NS3_BINARY` | Path to compiled ns-3 binary |

> Training always defaults to `--mock` because real ns-3 spawns a new process per step (~1 s each), making 50 K+ timesteps impractical. Use `--no-mock` only for final validation runs.

### Curriculum levels

| Level | Name | Timesteps |
|-------|------|-----------|
| 1 | Basic Bottleneck | 50,000 |
| 2 | Bursty Traffic | 80,000 |
| 3 | Chaotic Mixed | 80,000 |
| 4 | Real-World Cloudflare | 50,000 |
| **Total** | | **260,000** |

### Outputs
- `models/ppo_curriculum_final.zip` — Final trained model
- `models/checkpoint_latest.zip` — Latest per-level checkpoint

### Examples
```bash
# Fast (mock simulator)
uv run python main.py train --config configs/curriculum/ --output models/

# Real ns-3 (slow)
uv run python main.py train --config configs/curriculum/ --output models/ --no-mock
```

---

## `infer` — Run Inference

```bash
uv run python main.py infer (--model PATH | --baseline MBPS) --output CSV [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model PATH` | — | Path to trained PPO model (`.zip`) |
| `--baseline MBPS` | — | Static baseline rate in **Mbps** |
| `--output CSV` | required | Output CSV path for per-step metrics |
| `--steps N` | `100` | Number of simulation steps |
| `--traffic JSON` | — | Cloudflare Radar JSON for real demand |
| `--config YAML` | — | Curriculum YAML to override env settings |
| `--ns3-binary PATH` | `$NS3_BINARY` | Path to compiled ns-3 binary |

### Examples
```bash
# AI agent
uv run python main.py infer \
    --model models/ppo_curriculum_final.zip \
    --output data/results/ai_metrics.csv --steps 100

# Static baseline at 50 Mbps
uv run python main.py infer \
    --baseline 50 \
    --output data/results/baseline_50mbps.csv --steps 100

# AI with real-world Cloudflare demand
uv run python main.py infer \
    --model models/ppo_curriculum_final.zip \
    --traffic data/traffic/cloudflare_hourly.json \
    --output data/results/ai_realworld.csv --steps 168

# Multiple baselines
for rate in 30 40 50 60 70 80; do
  uv run python main.py infer --baseline $rate \
      --output "data/results/baseline_${rate}mbps.csv" --steps 100
done
```

---

## `evaluate` — A/B Evaluation (Parallel)

Compare AI models and static baselines simultaneously.

```bash
uv run python main.py evaluate [--model NAME:PATH ...] [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model NAME:PATH` | — | AI model, repeat for multiple |
| `--steps N` | `100` | Steps per scenario |
| `--output DIR` | `data/results/evaluation` | Output directory for CSVs |
| `--stress` | off | Use stress traffic (30–120 Mbps swings) |
| `--traffic-scale X` | `1.0` | Scale factor on Cloudflare demand |
| `--baselines MBPS...` | `30 50 70` | Static baseline rates |
| `--no-baselines` | off | Skip baselines (AI models only) |
| `--workers N` | `4` | Max parallel workers |
| `--ns3-binary PATH` | `$NS3_BINARY` | Path to compiled ns-3 binary |

### Stress traffic pattern

| Phase | Steps | Demand | Effect |
|-------|-------|--------|--------|
| Ramp up | 1–8 | 30→100 Mbps | Pressure builds |
| Hold high | 9–14 | 100–110 Mbps | Drops at low rates |
| Spike | 15–16 | 120 Mbps | Maximum stress |
| Crash | 17–20 | 30 Mbps | Agent should lower rate fast |
| Recovery | 21–30 | 40–80 Mbps | Gentle wave |

### Outputs (saved in `--output DIR`)
- `detail.csv` — Per-step data (scenario, step, demand, throughput, queue, drops, rate, reward)
- `summary.csv` — Aggregated comparison per scenario

### Examples
```bash
# Baselines only, stress traffic
uv run python main.py evaluate --stress --steps 100 --output data/results/eval/

# With AI model
uv run python main.py evaluate --stress --steps 100 \
    --model ai_v1:models/ppo_curriculum_final.zip \
    --output data/results/eval/

# Multiple models
uv run python main.py evaluate --steps 50 \
    --model v1:models/v1/ppo_curriculum_final.zip \
    --model v2:models/v2/ppo_curriculum_final.zip \
    --output data/results/compare/

# Scaled Cloudflare demand
uv run python main.py evaluate --traffic-scale 1.5 --steps 100 \
    --output data/results/cloudflare_scaled/
```

---

## `plot` — Generate Plots

```bash
uv run python main.py plot --detail CSV --summary CSV [--output DIR]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--detail CSV` | required | Per-step detail CSV |
| `--summary CSV` | required | Summary CSV |
| `--output DIR` | `data/results/plots` | Directory to save plots |

Plots saved:
- `comparison_timeseries.png` — Throughput / queue / drops / reward over time
- `summary_bars.png` — Bar chart comparison (throughput, queue, reward)
- `agent_actions.png` — AI rate decisions vs baseline lines *(only if AI agent present)*

### Example
```bash
uv run python main.py plot \
    --detail data/results/eval/detail.csv \
    --summary data/results/eval/summary.csv \
    --output data/results/eval/plots/
```

---

## `sanity` — Simulator Verification

Quick toll-booth sanity check using real ns-3.

```bash
uv run python main.py sanity [--steps N] [--ns3-binary PATH]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--steps N` | `5` | Steps per scenario |
| `--ns3-binary PATH` | `$NS3_BINARY` | ns-3 binary path |

Expected output:
```
Scenario                       Rate  Demand |     Thru      Queue  Drops | Expected
  demand < rate (no limit)      100      80  |    79.6          0      0  | [OK]
  demand = rate (boundary)       60      60  |    59.7          0      0  | [OK]
  demand > rate (limit)          50     100  |    52.3  4,999,104      0  | [OK]
  demand >> rate (heavy)         30     100  |    32.9  4,999,104      0  | [OK]
```

---

## `test` — Unit Tests

```bash
uv run python main.py test
```

Runs the `__main__` test blocks in:
- `src/utils/pipe_bridge.py`
- `src/environments/metrics.py`
- `src/environments/ns3_env.py`
- `src/agents/ppo_agent.py`

Exits with code `0` if all pass, `1` otherwise.

---

## Rebuilding ns-3 After C++ Changes

```bash
cd /home/blackprince/Downloads/ns-3.46.1

# Copy updated files
cp /home/blackprince/Documents/dev/work/network-shaping/ns-3.36/contrib/network-shaping/*.cc \
   /home/blackprince/Documents/dev/work/network-shaping/ns-3.36/contrib/network-shaping/*.h \
   contrib/network-shaping/

./ns3 build

# Verify
./ns3 run "network-shaping-simulation --rate=50Mbps --burst=500000 --source=100Mbps --duration=1"
# Expected: ~50 Mbps throughput, queue > 0, drops = 0
```

---

## How It Works

```
Python (RL agent)  →  spawns  →  ns-3 binary (one-shot)
                       ←  reads  ←  stdout: queue,throughput,drops
```

Each step:
1. Python reads the next demand value from the Cloudflare profile (Mbps)
2. Python spawns the ns-3 binary: `--rate=<mbps> --burst=<bytes> --source=<mbps> --duration=1`
3. ns-3 runs for 1 second, outputs `queue_bytes,throughput_mbps,drops` and exits
4. Python reads metrics, computes reward, decides next action

### Toll Booth Model

```
Source (100 Mbps) → [RateLimiterQueueDisc] → 100 Mbps link → Receiver
                          ↑ the bottleneck
                    agent sets this rate
```

- Source sends at demand Mbps; RateLimiterQueueDisc caps output at the agent's chosen rate
- If `agent_rate < demand`: excess queues up (5 MB max); if queue full → drops
- All values in **Mbps** and **bytes** — no Gbps anywhere in the pipeline
