import subprocess
from typing import Tuple


class Ns3Process:
    """
    Spawns ns-3 binary per step. Each invocation is independent.
    One-shot: runs simulation, outputs metrics, exits.

    Usage:
        proc = Ns3Process(binary_path)
        queue, throughput, drops = proc.step(rate_mbps=50, burst_bytes=5000000, demand_mbps=80)
        proc.stop()
    """

    def __init__(self, binary_path: str, args: list[str] | None = None):
        self.binary_path = binary_path
        self.extra_args = args or []

    def step(self, rate_mbps: float, burst_bytes: int,
             demand_mbps: float = 50.0) -> Tuple[int, float, int]:
        cmd = [
            self.binary_path,
            f"--rate={rate_mbps:.1f}Mbps",
            f"--burst={burst_bytes}",
            f"--source={demand_mbps:.1f}Mbps",
            "--duration=1.0",
        ] + self.extra_args

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"ns-3 exited with code {result.returncode}:\n{result.stderr}"
            )

        line = result.stdout.strip()
        queue_bytes, throughput_mbps, drops = line.split(",")
        return int(queue_bytes), float(throughput_mbps), int(drops)

    def stop(self):
        pass  # Nothing to clean up — each step is a separate process

    def kill(self):
        pass


class MockNs3Process:
    """
    Mock ns-3 process for unit testing without a compiled binary.
    Two demand modes:
      - stochastic: Ornstein-Uhlenbeck process (default for mock evaluation)
      - external: uses demand_mbps from environment (for Cloudflare traffic)
    """

    def __init__(self, binary_path: str = "", args: list[str] | None = None):
        del binary_path, args
        self.queue_bytes = 0
        self.throughput_mbps = 50.0
        self.drops = 0
        self.step_count = 0
        self.demand_mbps = 50.0
        self.rate_mbps = 50.0
        self._ou_demand = 50.0
        self.use_external_demand = False

    def _internal_demand(self) -> float:
        """Ornstein-Uhlenbeck demand: mean-reverting stochastic process with burst spikes."""
        import random
        theta = 0.10   # mean reversion speed
        mu = 70.0      # long-term mean (Mbps)
        sigma = 20.0   # volatility
        dt = 1.0

        noise = random.gauss(0, 1)
        self._ou_demand += theta * (mu - self._ou_demand) * dt + sigma * noise * (dt ** 0.5)
        self._ou_demand = max(10.0, min(130.0, self._ou_demand))

        # Burst: 20% chance of spike
        if random.random() < 0.20:
            self._ou_demand = min(130.0, self._ou_demand + random.uniform(20, 50))

        return self._ou_demand

    def step(self, rate_mbps: float, burst_bytes: int = 5000000,
             demand_mbps: float = 50.0) -> Tuple[int, float, int]:
        self.rate_mbps = rate_mbps
        self.step_count += 1

        if self.use_external_demand:
            self.demand_mbps = demand_mbps
        else:
            self.demand_mbps = self._internal_demand()

        # Throughput: source sends at demand, but bottleneck caps at rate
        self.throughput_mbps = min(self.demand_mbps, self.rate_mbps)

        # Queue: excess demand accumulates, drain when rate > demand
        excess = self.demand_mbps - self.rate_mbps
        if excess > 0:
            self.queue_bytes = int(min(5_000_000, self.queue_bytes + excess * 1000))
        else:
            drain = min(self.queue_bytes, abs(excess) * 1000)
            self.queue_bytes = int(self.queue_bytes - drain)

        # Drops: when demand severely exceeds rate
        if self.demand_mbps > self.rate_mbps * 1.5:
            self.drops += int((self.demand_mbps - self.rate_mbps) * 10)

        return self.queue_bytes, self.throughput_mbps, self.drops

    def stop(self):
        self.queue_bytes = 0
        self.throughput_mbps = 0.0
        self.drops = 0
        self.step_count = 0

    def kill(self):
        pass


if __name__ == "__main__":
    print("Testing MockNs3Process with internal demand...")
    mock = MockNs3Process()

    # Low rate (30) vs demand (~50-90): should have queue buildup and drops
    q, t, d = mock.step(rate_mbps=30, demand_mbps=0)
    print(f"Step 1: rate=30 demand~{mock.demand_mbps:.0f}: q={q} t={t:.1f} d={d}")
    assert t <= 30.0, f"Expected throughput<=30, got {t}"

    # High rate (90) vs demand: should drain most of queue
    for _ in range(5):
        q, t, d = mock.step(rate_mbps=90, demand_mbps=0)
    print(f"Step 6: rate=90 demand~{mock.demand_mbps:.0f}: q={q} t={t:.1f} d={d}")
    # Queue may not fully drain due to stochastic demand spikes

    mock.stop()
    print("MockNs3Process tests passed!")
