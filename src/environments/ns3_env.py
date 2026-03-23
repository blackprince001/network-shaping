import json
import os
import sys
from math import tanh
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.metrics import MetricsCollector
from utils.pipe_bridge import MockNs3Process, Ns3Process

DEFAULT_NS3_BINARY = "/home/blackprince/Downloads/ns-3.46.1/build/scratch/ns3.46.1-network-shaping-simulation-optimized"
DEFAULT_CF_DATA = "/home/blackprince/Documents/dev/work/network-shaping/data/traffic/cloudflare_hourly.json"


def load_demand_profile(json_path: str, scale_mbps: float = 100.0) -> list[float]:
    """Load Cloudflare demand pattern and scale to Mbps."""
    if not os.path.exists(json_path):
        # Fallback: sinusoidal demand
        return [
            scale_mbps * (0.5 + 0.4 * np.sin(i * 2 * np.pi / 24)) for i in range(168)
        ]

    with open(json_path) as f:
        data = json.load(f)
    result = data["result"]
    values = []
    if "main" in result:
        values.extend(float(v) for v in result["main"]["values"])
    if "previous" in result:
        values.extend(float(v) for v in result["previous"]["values"])
    if not values:
        values = [float(v) for v in result.get("values", [])]
    return [v * scale_mbps for v in values]


class Ns3Env(gym.Env):
    """
    ns-3 env with Cloudflare demand profile.

    Action: [rate_mbps] — agent's TBF rate [1, 100] Mbps
    Obs: [queue_norm, throughput_norm, drops_norm, demand_norm]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        ns3_binary: str = DEFAULT_NS3_BINARY,
        config: Optional[Dict[str, Any]] = None,
        mock: bool = False,
    ):
        super().__init__()

        self.cfg = config or {}
        self.mock = mock
        self.ns3_binary = ns3_binary

        self.min_rate_mbps = self.cfg.get("min_rate_mbps", 1.0)
        self.max_action_rate = self.cfg.get("max_action_rate_mbps", 100.0)

        self.action_space = gym.spaces.Box(
            low=np.array([self.min_rate_mbps], dtype=np.float32),
            high=np.array([self.max_action_rate], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation includes demand_norm
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Load demand profile
        if "demand_profile" in self.cfg:
            # Pre-built profile (e.g. stress test from evaluate.py)
            self.demand_profile = self.cfg["demand_profile"]
        else:
            cf_path = self.cfg.get("traffic_file", DEFAULT_CF_DATA)
            base_scale = self.cfg.get("cloudflare_scale", 1.0) * 100.0
            self.demand_profile = load_demand_profile(cf_path, scale_mbps=base_scale)

        self.ns3 = None
        self.current_rate = (self.min_rate_mbps + self.max_action_rate) / 2.0
        self.prev_rate = self.current_rate
        self.burst_multiplier = self.cfg.get("burst_multiplier", 0.1)
        self.max_rate_mbps = self.cfg.get("max_rate_mbps", 100)
        self.max_queue_bytes = self.cfg.get("max_queue_bytes", 5_242_880)
        self.max_drops_norm = self.cfg.get("max_drops_norm", 100000.0)
        self.max_delta = self.max_action_rate - self.min_rate_mbps

        self._max_episode_steps = self.cfg.get("max_episode_steps", 168)
        self._elapsed_steps = 0
        self.metrics = MetricsCollector()
        self.throughput_mbps = 0.0
        self.queue_bytes = 0
        self.drops = 0
        self.current_demand_mbps = 50.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.ns3 is not None:
            try:
                self.ns3.kill()
            except Exception:
                pass
            self.ns3 = None

        if self.mock:
            self.ns3 = MockNs3Process()
            self.ns3.use_external_demand = not self.cfg.get("stochastic_demand", False)
        else:
            self.ns3 = Ns3Process(self.ns3_binary)

        self._elapsed_steps = 0
        self.current_rate = (self.min_rate_mbps + self.max_action_rate) / 2.0
        self.prev_rate = self.current_rate
        self.throughput_mbps = 0.0
        self.queue_bytes = 0
        self.drops = 0
        self.current_demand_mbps = self.demand_profile[0]
        self.metrics = MetricsCollector()

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        rate_mbps = float(np.clip(action[0], self.min_rate_mbps, self.max_action_rate))
        burst_bytes = int(rate_mbps * self.burst_multiplier * 1e6 / 8.0)

        # Get demand from Cloudflare profile
        demand_idx = self._elapsed_steps % len(self.demand_profile)
        self.current_demand_mbps = self.demand_profile[demand_idx]

        # Run ns-3 step
        if self.ns3 is not None:
            self.queue_bytes, self.throughput_mbps, self.drops = self.ns3.step(
                rate_mbps, burst_bytes, self.current_demand_mbps
            )

        reward = self._compute_reward(
            self.throughput_mbps,
            self.queue_bytes,
            self.drops,
            prev_rate=self.prev_rate,
            curr_rate=rate_mbps,
        )

        self.metrics.record(
            timestamp=float(self._elapsed_steps),
            throughput_mbps=self.throughput_mbps,
            delay_ms=0.0,
            drop_count=self.drops,
            queue_occupancy_bytes=self.queue_bytes,
            reward=reward,
            rate_limit_mbps=rate_mbps,
        )

        self.prev_rate = self.current_rate
        self.current_rate = rate_mbps
        self._elapsed_steps += 1

        obs = self._get_obs()
        terminated = False
        truncated = self._elapsed_steps >= self._max_episode_steps

        info = {
            "queue_bytes": self.queue_bytes,
            "throughput_mbps": self.throughput_mbps,
            "drops": self.drops,
            "rate_mbps": rate_mbps,
            "demand_mbps": self.current_demand_mbps,
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, throughput, queue, drops, prev_rate, curr_rate):
        alpha = self.cfg.get("alpha", 1.0)
        beta = self.cfg.get("beta", 0.5)
        gamma = self.cfg.get("gamma", 0.8)
        delta = self.cfg.get("delta", 0.0)

        t_norm = min(1.0, throughput / self.max_rate_mbps)
        q_norm = min(1.0, queue / self.max_queue_bytes)
        d_norm = tanh(drops / self.max_drops_norm)
        s_norm = abs(curr_rate - prev_rate) / self.max_delta

        w = alpha + beta + gamma + delta
        if w == 0:
            w = 1.0

        reward = (
            (alpha / w * t_norm)
            - (beta / w * q_norm)
            - (gamma / w * d_norm)
            - (delta / w * s_norm)
        )
        return float(np.clip(reward, -1.0, 1.0))

    def _get_obs(self):
        t_norm = min(1.0, self.throughput_mbps / self.max_rate_mbps)
        q_norm = min(1.0, self.queue_bytes / self.max_queue_bytes)
        d_norm = min(1.0, self.drops / self.max_drops_norm)
        demand_norm = min(1.0, self.current_demand_mbps / self.max_rate_mbps)
        return np.array([q_norm, t_norm, d_norm, demand_norm], dtype=np.float32)

    def close(self):
        if self.ns3 is not None:
            try:
                self.ns3.stop()
            except Exception:
                self.ns3.kill()
            self.ns3 = None

    @property
    def max_episode_steps(self):
        return self._max_episode_steps


if __name__ == "__main__":
    print("Testing Ns3Env with demand profile...")

    env = Ns3Env(mock=True, config={"max_episode_steps": 10})
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape} (should be 4 with demand)")
    assert obs.shape == (4,), f"Expected (4,), got {obs.shape}"

    for i in range(5):
        action = env.action_space.sample()
        obs, r, t, tr, info = env.step(action)
        print(
            f"  Step {i + 1}: rate={action[0]:.1f} demand={info['demand_mbps']:.1f} "
            f"t={info['throughput_mbps']:.1f} q={info['queue_bytes']} d={info['drops']} r={r:.3f}"
        )

    env.close()
    print("Test passed!")
