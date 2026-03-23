# Network Shaping RL - Commands

## Prerequisites

```bash
# Install Python dependencies
uv sync

# Build ns-3 (first time or after C++ changes)
cd /home/blackprince/Downloads/ns-3.46.1
./ns3 configure --enable-examples
./ns3 build

# Verify binary exists
ls build/scratch/ns3.46.1-network-shaping-simulation-optimized
```

## Benchmark

Verify the toll booth behavior before training or evaluation.

```bash
cd /home/blackprince/Documents/dev/work/network-shaping
uv run python main.py benchmark
```

Expected output:
```
Scenario                       Rate  Demand |     Thru      Queue  Drops | Expected
  demand<rate (no limit)       100      80  |    79.6          0      0  | [OK]
  demand>rate (limit)           50     100  |    52.3  4,999,104      0  | [OK]
  demand>>rate (heavy)          30     100  |    32.9  4,999,104      0  | [OK]
  demand=rate (boundary)        60      60  |    59.7          0      0  | [OK]
```

Run with custom step count:
```bash
uv run python main.py benchmark --steps 10
```

## Training

Training always uses the mock simulator — real ns-3 spawns a binary per step (~1s each), making 50K+ timesteps impractical.

```bash
cd /home/blackprince/Documents/dev/work/network-shaping
uv run python main.py train --config configs/curriculum/ --output models/
```

### Curriculum levels

| Level | Name | Timesteps | Max Episode Steps | n_steps |
|-------|------|-----------|-------------------|---------|
| 1 | Basic Bottleneck | 50,000 | 50 | 512 |
| 2 | Bursty Traffic | 80,000 | 50 | 512 |
| 3 | Chaotic Mixed | 80,000 | 50 | 512 |
| 4 | Real-World Cloudflare | 50,000 | 50 | 256 |
| **Total** | | **260,000** | | |

Output:
- `models/ppo_curriculum_final.zip` — Final trained model
- `models/checkpoint_latest.zip` — Latest per-level checkpoint

## Inference

Inference uses real ns-3. The binary must be built first.

### AI agent

```bash
uv run python main.py infer --model models/ppo_curriculum_final.zip \
  --output data/results/ai_metrics.csv --steps 100
```

### AI with Cloudflare real-world traffic

```bash
uv run python main.py infer --model models/ppo_curriculum_final.zip \
  --traffic data/traffic/cloudflare_hourly.json \
  --output data/results/ai_realworld.csv --steps 168
```

### Static baseline (no AI, fixed rate)

```bash
uv run python main.py infer --baseline 50 \
  --output data/results/baseline_50mbps.csv --steps 100
```

### Multiple baselines

```bash
for rate in 30 40 50 60 70 80; do
  uv run python main.py infer --baseline $rate \
    --output "data/results/baseline_${rate}mbps.csv" --steps 100
done
```

### Custom ns-3 binary path

```bash
uv run python main.py infer --baseline 50 \
  --ns3-binary /path/to/ns3.46.1-network-shaping-simulation-optimized \
  --output data/results/baseline.csv --steps 100
```

## A/B Evaluation

Compare AI agent against static baselines in parallel.

### Stress test (recommended — forces real decisions)

Demand swings 30–120 Mbps in a pattern that guarantees drops and queue buildup:

| Phase | Steps | Demand | What happens |
|-------|-------|--------|-------------|
| Ramp up | 1–8 | 30→100 Mbps | Pressure builds |
| Hold high | 9–14 | 100–110 Mbps | Drops at low rates |
| Spike | 15–16 | 120 Mbps | Maximum stress |
| Crash | 17–20 | 30 Mbps | Agent should drop rate fast |
| Recovery | 21–30 | 40–80 Mbps | Gentle wave |

```bash
# Baselines only
uv run python evaluate.py --stress --steps 100

# With AI agent
uv run python evaluate.py --model models/ppo_curriculum_final.zip --stress --steps 100
```

### Scaled Cloudflare traffic

```bash
# Push demand 1.5x higher than raw Cloudflare data
uv run python evaluate.py --traffic-scale 1.5 --steps 100

# With AI agent
uv run python evaluate.py --model models/ppo_curriculum_final.zip --traffic-scale 1.5 --steps 100
```

### Default Cloudflare traffic

```bash
uv run python evaluate.py --steps 100
```

### Output

- `data/results/evaluation_detail.csv` — Per-step data (demand, throughput, queue, drops, rate, reward)
- `data/results/evaluation_summary.csv` — Summary comparison

## Plotting

```bash
uv run python plot.py \
  --detail data/results/evaluation_detail.csv \
  --summary data/results/evaluation_summary.csv
```

Plots saved to `data/results/plots/`:
- `comparison_timeseries.png` — Throughput (with demand line) / Queue / Drops / Reward over time
- `summary_bars.png` — Bar chart comparison across scenarios
- `agent_actions.png` — AI agent rate decisions (only if AI agent was evaluated)

## Unit Tests

```bash
# All tests
uv run python src/utils/pipe_bridge.py
uv run python src/environments/metrics.py
uv run python src/environments/ns3_env.py
uv run python src/agents/ppo_agent.py
```

## Rebuilding ns-3 After C++ Changes

```bash
cd /home/blackprince/Downloads/ns-3.46.1

# Copy updated contrib files from project
cp /home/blackprince/Documents/dev/work/network-shaping/ns-3.36/contrib/network-shaping/*.cc \
   /home/blackprince/Documents/dev/work/network-shaping/ns-3.36/contrib/network-shaping/*.h \
   contrib/network-shaping/

# Build
./ns3 build

# Verify
./ns3 run "network-shaping-simulation --rate=50Mbps --burst=500000 --source=100Mbps --duration=1"
# Expected: ~50 Mbps throughput, queue > 0, drops = 0
```

## How It Works

### Architecture

```
Python (RL agent)  →  spawns  →  ns-3 binary (one-shot)
                          ←  reads  ←  stdout: queue,throughput,drops
```

Each simulation step:
1. Python reads the next demand value from the Cloudflare profile (Mbps)
2. Python spawns the ns-3 binary with CLI args: `--rate=<mbps> --burst=<bytes> --source=<mbps> --duration=1`
3. ns-3 runs the simulation for 1 second with a fixed source at `source` rate
4. The `RateLimiterQueueDisc` (toll booth) caps output at `rate` Mbps
5. ns-3 outputs `queue_bytes,throughput_mbps,drops` to stdout and exits
6. Python reads the metrics, computes reward, decides next action

### Toll Booth Model

```
Source (100 Mbps) → [RateLimiterQueueDisc] → 100 Mbps link → Receiver
                         ↑ the bottleneck
                    agent sets this rate
```

- Source always sends at 100 Mbps (full link rate)
- RateLimiterQueueDisc caps output at the agent's rate (30–90 Mbps)
- Throughput at receiver = min(source_rate, agent_rate)
- If agent_rate < source_rate: excess queues up (5 MB max)
- If queue is full: packets are dropped
- The 100 Mbps link is just wire — irrelevant to shaping

### Units

All values in Mbps and bytes. No Gbps anywhere in the pipeline.

### Queue

5 MB token bucket queue. Burst is `rate * 0.1 * 1e6 / 8` bytes (scales with rate).
