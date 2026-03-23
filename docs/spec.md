# SPEC.md — RL Network Shaping Project

## Autonomous Reinforcement Learning Traffic Shaper with ns-3 Integration

---

## Project Goal

Build a production-grade, end-to-end reinforcement learning system that dynamically controls network traffic shaping in a simulated environment. The agent learns to outperform static, traditional Token Bucket rate limiters by intelligently adjusting bandwidth allocation in real time — minimizing packet drops, latency, and bufferbloat while maximizing throughput across a simulated internet backbone (Dumbbell topology).

The system must be scientifically evaluable: a single CLI invocation should produce exportable CSV metrics proving AI superiority over a naive fixed-rate baseline.

---

## Project Architecture

```
rl-network-shaping/
├── main.py                          # CLI entrypoint (train / infer / baseline)
├── pyproject.toml                   # uv-managed Python deps
│
├── configs/
│   └── curriculum/
│       ├── level1_basic.yaml        # 3x3 dumbbell, flat traffic
│       ├── level2_bursty.yaml       # 15x15 topology, VBR
│       ├── level3_chaotic.yaml      # 50x50, TCP elephant + UDP mice
│       └── level4_realworld.yaml    # Cloudflare Radar hourly traffic
│
├── src/
│   ├── environments/
│   │   ├── ns3_env.py               # Gymnasium Env — owns ns-3 lifecycle
│   │   └── metrics.py               # MetricsCollector — CSV export
│   │
│   ├── agents/
│   │   └── ppo_agent.py             # PPO wrapper (stable-baselines3)
│   │
│   ├── traffic/
│   │   └── cloudflare_loader.py     # Parses Cloudflare Radar JSON profiles
│   │
│   └── utils/
│       └── pipe_bridge.py           # Subprocess stdin/stdout line protocol helpers
│
├── ns-3.36/
│   └── contrib/
│       └── network-shaping/
│           ├── network-shaping-simulation.cc   # Main topology + scheduler
│           ├── bridge-env.cc / .h              # ZMQ server loop (C++)
│           └── traffic-generator.cc / .h       
│
├── models/
│   └── ppo_curriculum_final.zip     # Trained model checkpoint
│
└── data/
    ├── results/
    │   ├── baseline_metrics.csv
    │   └── ai_metrics.csv
    └── traffic/
        └── cloudflare_hourly.json
```

---

## Module Specifications

---

### 1. `main.py` — CLI Entrypoint

**Responsibilities:** Parse CLI args and dispatch to train, infer, or baseline mode.

**Commands:**

```bash
# Train through all curriculum levels
uv run python main.py train --config configs/curriculum/ --output models/

# Run AI inference and record metrics
uv run python main.py infer --model models/ppo_curriculum_final.zip --use-ns3 --output data/results/ai_metrics.csv

# Run static baseline (bypasses AI entirely)
uv run python main.py infer --baseline 50 --use-ns3 --output data/results/baseline_metrics.csv

# Optional: pass real traffic profile
uv run python main.py infer --model models/ppo_curriculum_final.zip --traffic data/traffic/cloudflare_hourly.json --output data/results/ai_realworld.csv
```

**Key logic:**

- In `train` mode: loop through YAML curriculum levels in order, preserving model weights between stages via `model.save()` / `model.load()`.
- In `infer --baseline N` mode: skip PPO entirely; feed constant `N` Gbps rate limit to the env every step.
- Both modes must write structured CSV via `MetricsCollector`.

---

### 2. `src/environments/ns3_env.py` — Gymnasium Environment

**Class:** `Ns3Env(gym.Env)`

**Observation space:** `Box(low=0, shape=(3,), dtype=np.float32)`

- `[queue_bytes_normalized, throughput_gbps_normalized, drop_count_normalized]`

**Action space:** `Box(low=1.0, high=100.0, shape=(1,), dtype=np.float32)`

- Represents `rate_limit_gbps` — any continuous value between 1 and 100 Gbps.
- `burst_mb` is **not** an action. It is auto-derived as a fixed multiple of the chosen rate:

  ```python
  BURST_MULTIPLIER = 0.1  # configurable in YAML
  burst_mb = rate_limit_gbps * BURST_MULTIPLIER
  # e.g. agent picks 60 Gbps → burst = 6 MB
  ```

- This keeps burst proportional to rate at all times, meaning the TBF always allows a burst window sized to ~100ms of the current rate limit — a standard rule of thumb for Token Bucket sizing.

**Key methods:**

`reset(seed=None)`

- Kill any zombie ns-3 process from prior episode.
- Spawn fresh ns-3 subprocess via `Ns3Process(binary, args)`.
- Wait for `"READY"` signal over stdout pipe.
- Return initial observation (zeroed state).

`step(action)`

- Extract `rate_limit_gbps = float(action[0])`, clipped to `[1.0, 100.0]`.
- Derive `burst_mb = rate_limit_gbps * self.cfg.get("burst_multiplier", 0.1)`.
- Call `self.ns3.step(rate_limit_gbps, burst_mb)` over stdin/stdout pipe.
- Receive `(queue_bytes, throughput_mbps, drops)`.
- Compute reward via `_compute_reward(..., prev_rate=self.current_rate, curr_rate=rate_limit_gbps)`.
- Update `self.current_rate = rate_limit_gbps`.
- Call `MetricsCollector.record()`, return `(obs, reward, terminated, truncated, info)`.

`_compute_reward(throughput_mbps, queue_bytes, drops)`

```python
from math import tanh

def _compute_reward(self, throughput_mbps, queue_bytes, drops):
    max_rate  = self.cfg.get("max_rate_mbps",  100)
    max_queue = self.cfg.get("max_queue_bytes", 5_242_880)
    max_drops = self.cfg.get("max_drops_norm",  100.0)

    alpha = self.cfg.get("alpha", 1.0)
    beta  = self.cfg.get("beta",  0.5)
    gamma = self.cfg.get("gamma", 0.8)

    # Normalize each term to [0, 1]
    t_norm = min(1.0, throughput_mbps / max_rate)
    q_norm = min(1.0, queue_bytes / max_queue)
    d_norm = tanh(drops / max_drops)

    # Normalize weights so they sum to 1.0, keeping relative magnitudes
    weight_sum = alpha + beta + gamma
    alpha_n = alpha / weight_sum
    beta_n  = beta  / weight_sum
    gamma_n = gamma / weight_sum

    # Reward is bounded to [-1.0, +1.0]
    return (alpha_n * t_norm) - (beta_n * q_norm) - (gamma_n * d_norm)
```

`close()`

- Call `self.ns3.stop()`, terminate subprocess cleanly.

**Notes:**

- Do NOT use `ns3gym` or Protobuf. Raw `pyzmq` only.
- ns-3 subprocess must be killed on `reset()` to prevent socket hangs between episodes.

---

### 3. `src/environments/metrics.py` — MetricsCollector

**Class:** `MetricsCollector`

**Purpose:** Record one row per simulation timestep (1.0 second of ns-3 clock time).

**Tracked columns:**

| Column | Type | Description |
|---|---|---|
| `timestamp` | float | Elapsed ns-3 simulation seconds |
| `throughput_gbps` | float | Measured bottleneck throughput |
| `delay_ms` | float | Congestion-induced queue delay |
| `drop_count` | int | Packets dropped at queue shaper |
| `queue_occupancy_bytes` | int | Raw bytes in queue buffer |
| `reward` | float | Computed RL reward scalar |
| `rate_limit_gbps` | float | Agent's chosen rate (or static baseline) |
| `rate_delta_gbps` | float | Step-to-step change in rate limit (`\|curr - prev\|`) |

**Methods:**

- `record(timestamp, throughput, delay, drops, queue, reward, rate_limit)` — appends row.
- `to_dataframe()` → `pd.DataFrame`
- `export_csv(path: str)` — writes CSV to disk.

---

### 4. `src/agents/ppo_agent.py` — PPO Agent Wrapper

**Dependencies:** `stable-baselines3`

**Responsibilities:**

- Instantiate `PPO("MlpPolicy", env, verbose=1, ...)`.
- Expose `train(total_timesteps)`, `save(path)`, `load(path)`, `predict(obs)`.
- Hyperparameters should be configurable from YAML (learning rate, gamma, clip range, n_steps).

**Curriculum training flow (in `main.py`):**

```python
for level_config in sorted(curriculum_dir.glob("*.yaml")):
    cfg = yaml.safe_load(level_config.read_text())
    env = Ns3Env(topology=cfg["topology"], traffic=cfg["traffic"])
    if model_exists:
        model.set_env(env)
    else:
        model = PPO("MlpPolicy", env)
    model.learn(total_timesteps=cfg["timesteps"])
    model.save("models/checkpoint_latest.zip")
```

---

### 5. `configs/curriculum/*.yaml` — Curriculum Level Definitions

**Schema:**

```yaml
level: 1
name: "Basic Bottleneck"
topology:
  n_senders: 3
  n_receivers: 3
  bottleneck_gbps: 100
  link_delay_ms: 5
traffic:
  type: "constant"       # "elephant_mice", "cloudflare"
  demand_gbps: 60
timesteps: 10000
agent:
  learning_rate: 3.0e-4
  gamma: 0.99
  clip_range: 0.2
  n_steps: 2048
  burst_multiplier: 0.1   # burst_mb = rate_limit_gbps * burst_multiplier
reward:
  alpha: 1.0              # throughput weight
  beta:  0.5              # queue occupancy weight
  gamma: 0.8              # drop count weight
  delta: 0.0              # smoothing penalty weight (0.0 = disabled for Level 1)
```

---

### 6. `ns-3.36/contrib/network-shaping/network-shaping-simulation.cc` — C++ Simulator

**Topology:** Dumbbell — N senders → Router A → 100 Gbps bottleneck → Router B → N receivers.

**Queue placement:** `TbfQueueDisc` installed exclusively on Router A's outbound bottleneck interface.

**TBF parameters updated dynamically:**

```cpp
Ptr<TbfQueueDisc> tbf = ...;
tbf->SetAttribute("Rate", DataRateValue(DataRate(rateMbps * 1e6)));
tbf->SetAttribute("Burst", UintegerValue(burstBytes));
```

**DO NOT use custom QueueShaper C++ class** — `TrafficControlHelper::Install()` enters infinite deadlock with non-standard QueueDiscs.

**Traffic generation:** Use  `ns3::PacketSink` or any better option to create realistic data. Call `app->SetAttribute("DataRate", ...)` per 1.0-second step to scale demand safely.

---

### 7. `ns-3.36/contrib/network-shaping/bridge-env.cc` — ZMQ Server (C++)

**Dependencies:** `libzmq`, `cppzmq`

**Protocol:**

- Binds a `ZMQ_REP` socket on port `5555`.
- Scheduled via `Simulator::Schedule(Seconds(1.0), &BridgeEnv::Step, this)`.

**Step loop:**

1. Freeze: block on `socket.recv(msg)`.
2. Parse: `"STEP,<rate_gbps>,<burst_mb>"` → extract floats.
3. Apply: update `TbfQueueDisc` attributes with new rate + burst.
4. Collect: read `queue_bytes`, `throughput_mbps`, `drop_count` from flow monitors.
5. Reply: `"<queue_bytes>,<throughput_mbps>,<drop_count>"`.
6. Reschedule: `Simulator::Schedule(Seconds(1.0), &BridgeEnv::Step, this)`.

Handle `"STOP"` message by calling `Simulator::Stop()`.

---

### 8. `src/traffic/cloudflare_loader.py` — Real Traffic Profile

**Input:** Cloudflare Radar JSON file with hourly demand percentages.

**Output:** `List[float]` of demand values in Gbps, indexed by simulation step.

**Usage:** In `Ns3Env`, if `--traffic` flag is provided, the loader feeds demand values to `TrafficGenerator` via ZMQ alongside the rate limit command.

---

## Communication Protocol (Python ↔ C++)

Communication uses **stdio pipes** — Python spawns ns-3 as a subprocess and talks to it via `stdin`/`stdout`. No sockets, no ports, no bind/connect race conditions. This makes parallel training instances trivially safe since each subprocess has its own isolated pipe.

```
Python → ns-3 stdin:   "STEP,<rate_gbps>,<burst_mb>\n"
ns-3 → Python stdout:  "<queue_bytes>,<throughput_mbps>,<drops>\n"

Python → ns-3 stdin:   "STOP\n"
ns-3 → Python stdout:  "BYE\n"
```

**Python side (`pipe_bridge.py`):**

```python
import subprocess

class Ns3Process:
    def __init__(self, binary_path, args):
        self.proc = subprocess.Popen(
            [binary_path] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1  # line-buffered
        )
        # Wait for ns-3 to signal it's ready
        ready = self.proc.stdout.readline().strip()
        assert ready == "READY", f"Unexpected startup message: {ready}"

    def step(self, rate_gbps: float, burst_mb: float) -> tuple[int, float, int]:
        self.proc.stdin.write(f"STEP,{rate_gbps:.4f},{burst_mb:.4f}\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline().strip()
        queue_bytes, throughput_mbps, drops = reply.split(",")
        return int(queue_bytes), float(throughput_mbps), int(drops)

    def stop(self):
        self.proc.stdin.write("STOP\n")
        self.proc.stdin.flush()
        self.proc.stdout.readline()  # consume "BYE"
        self.proc.terminate()
        self.proc.wait()
```

**C++ side (in `bridge-env.cc`):**

```cpp
// On startup, signal Python we're ready
std::cout << "READY" << std::endl;

// Each scheduled step
void BridgeEnv::Step() {
    std::string msg;
    std::getline(std::cin, msg);

    if (msg == "STOP") {
        std::cout << "BYE" << std::endl;
        Simulator::Stop();
        return;
    }

    // Parse "STEP,<rate>,<burst>"
    // ... apply to TbfQueueDisc ...
    // ... collect metrics ...

    std::cout << queue_bytes << "," << throughput_mbps << "," << drops << std::endl;
    Simulator::Schedule(Seconds(1.0), &BridgeEnv::Step, this);
}
```

**Notes:**

- Always one write per readline — strict alternating protocol, never pipeline.
- `bufsize=1` (line-buffered) and `std::endl` (not `"\n"`) are critical — both sides must flush after every message.
- No port numbers means you can run 8 parallel training envs with zero configuration.
- Debugging is trivial: just `tee` stdout to a file or add print statements.

---

## Reward Function

Computed entirely in Python (`Ns3Env._compute_reward`). Never in C++.

```python
from math import tanh

def _compute_reward(self, throughput_mbps, queue_bytes, drops, prev_rate, curr_rate):
    max_rate   = self.cfg.get("max_rate_mbps",   100)
    max_queue  = self.cfg.get("max_queue_bytes",  5_242_880)
    max_drops  = self.cfg.get("max_drops_norm",   100.0)
    max_delta  = self.cfg.get("max_rate_delta",   99.0)   # max possible jump = 100 - 1

    alpha = self.cfg.get("alpha", 1.0)
    beta  = self.cfg.get("beta",  0.5)
    gamma = self.cfg.get("gamma", 0.8)
    delta = self.cfg.get("delta", 0.2)   # smoothing weight — set 0.0 to disable

    # Normalize inputs to [0, 1]
    t_norm = min(1.0, throughput_mbps / max_rate)
    q_norm = min(1.0, queue_bytes / max_queue)
    d_norm = tanh(drops / max_drops)
    s_norm = abs(curr_rate - prev_rate) / max_delta   # smoothness penalty, in [0, 1]

    # Normalize weights so they sum to 1.0 — keeps reward bounded to [-1.0, +1.0]
    weight_sum = alpha + beta + gamma + delta
    alpha_n = alpha / weight_sum
    beta_n  = beta  / weight_sum
    gamma_n = gamma / weight_sum
    delta_n = delta / weight_sum

    return (alpha_n * t_norm) - (beta_n * q_norm) - (gamma_n * d_norm) - (delta_n * s_norm)
```

**Why normalize the weights?**

- Without normalization, changing any weight inflates the reward scale, destabilizing PPO's value function estimates.
- With normalized weights, reward is always in `[-1.0, +1.0]` regardless of raw values in YAML.
- The agent only cares about the *ratio* between weights — normalization preserves that exactly.

**The smoothing term (`delta`):**

- Penalizes large step-to-step rate jumps: a jump from 90 → 5 Gbps scores `s_norm = 0.86`, a nudge from 50 → 52 scores `s_norm = 0.02`.
- `prev_rate` is tracked as instance state in `Ns3Env` and updated each step.
- Set `delta: 0.0` in YAML to disable it entirely if you want to observe unsmoothed behavior.
- Recommended curriculum progression: start with `delta: 0.0` in Level 1 so the agent first learns *what* the right rate is, then introduce `delta: 0.2` from Level 2 onward to encourage smooth, stable control.

**Tuning guidance:**

| Weight | Effect when increased |
|---|---|
| `alpha` | Agent prioritizes throughput — pushes rate limit higher |
| `beta` | Agent is more conservative about queue depth |
| `gamma` | Agent is more aggressive about avoiding drops |
| `delta` | Agent makes smoother, more gradual rate adjustments |

All four exposed in YAML per curriculum level.

---

## A/B Evaluation Protocol

To scientifically prove AI superiority:

```bash
# 1. Generate baseline CSV
uv run python main.py infer --baseline 50 --use-ns3 --output data/results/baseline_50g.csv

# 2. Generate AI CSV
uv run python main.py infer --model models/ppo_curriculum_final.zip --use-ns3 --output data/results/ai_ppo.csv
```

Then in Python/Jupyter:

```python
import pandas as pd
import matplotlib.pyplot as plt

baseline = pd.read_csv("data/results/baseline_50g.csv")
ai = pd.read_csv("data/results/ai_ppo.csv")

# Plot 1: Throughput vs Rate Limit (wasted bandwidth)
# Plot 2: Drop Count over time (bufferbloat)
# Plot 3: Latency comparison
# Plot 4: Reward trajectory
```

Three key conclusions to draw:

- **Wasted Bandwidth:** Baseline locks at 50 Gbps; AI tracks demand dynamically.
- **Bufferbloat Prevention:** AI drops fewer packets by acting before bottleneck congestion.
- **Circadian Rhythm Tracking:** AI reacts to rush-hour spikes in Cloudflare traffic data.

---

## Dependencies

**Python (`pyproject.toml`):**

```toml
[project]
dependencies = [
  "gymnasium>=0.29",
  "stable-baselines3>=2.0",
  "numpy>=1.24",
  "pandas>=2.0",
  "pyyaml>=6.0",
  "matplotlib>=3.7",
  "seaborn>=0.12",
]
```

No ZMQ dependency — communication is via Python's built-in `subprocess` module.

**C++ (ns-3 build):**

- Standard ns-3.36 build — no extra C++ dependencies needed.
- `bridge-env.cc` uses only `<iostream>` and `<string>` for the pipe protocol.
- ns-3.36 with `./ns3 configure --enable-examples --enable-tests`

**Runtime:**

- `uv` for Python environment management
- ns-3.36 pre-built with `./ns3 configure --enable-examples --enable-tests`

---

## Coding Agent Task Order

The agent should complete tasks in this sequence to avoid blocked dependencies:

1. **Scaffold project structure** — create all directories and empty files.
2. **Implement `MetricsCollector`** — pure Python, no external deps, testable immediately.
3. **Implement `pipe_bridge.py`** — `Ns3Process` class wrapping `subprocess.Popen` with stdin/stdout line protocol.
4. **Implement `Ns3Env`** — depends on ZMQ bridge; mock ns-3 process for unit tests.
5. **Write YAML curriculum configs** — all 4 levels.
6. **Implement `PPO agent wrapper`** — thin wrapper around stable-baselines3.
7. **Implement `main.py` CLI** — wires everything together.
8. **Write C++ `bridge-env.cc`** — ZMQ REP server, TBF update logic.
9. **Write C++ `network-shaping-simulation.cc`** — full topology + scheduler.
10. **Write C++ `traffic-generator.cc`** — demand scaler.
11. **Integration test** — run single curriculum level end-to-end.
12. **Implement `cloudflare_loader.py`** — real traffic profile parser.
13. **Run full curriculum training** — 50,000 steps total across all levels.
14. **Run A/B evaluation** — generate both CSVs and comparison plots.

---

## Known Pitfalls (Lessons Learned)

| Problem | Wrong Approach | Correct Solution |
|---|---|---|
| C++ freeze on startup | Custom `QueueShaper` class | Use native `ns3::TbfQueueDisc` |
| Silent serialization errors | `ns3gym` + Protobuf | Plain stdin/stdout line protocol |
| Port conflicts on parallel training | ZMQ sockets with fixed port | stdio pipes — each subprocess is isolated |
| Socket hangs between episodes | Not killing ns-3 on `reset()` | Terminate subprocess at start of every `reset()` |
| Pipe deadlock | Not flushing after writes | `bufsize=1` in Python + `std::endl` in C++ |
| Reward scale instability | Raw alpha/beta/gamma weights | Normalize weights to sum to 1.0 before applying |
| Rate oscillation during training | Absolute action with no smoothing incentive | Add `delta` smoothing penalty; start at `0.0` in Level 1, introduce `0.2` from Level 2 |
