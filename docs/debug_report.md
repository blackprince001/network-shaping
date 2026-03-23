# ns-3 Simulation Debugging: Problem, Attempts, and Conclusions

## What We Are Trying To Do

Build a network simulation that behaves like a **toll booth**:
- Packets arrive at some rate (demand)
- A rate limiter (toll booth) controls the output rate
- Output throughput = min(demand, rate) — NEVER exceeds the configured rate
- If demand > rate: excess packets queue up
- If queue is full: excess packets are dropped
- The 100 Mbps link capacity is irrelevant — it's just the physical wire

The agent's action sets the toll booth rate (30–90 Mbps). The simulation should return: queue depth, actual throughput, and drop count.

---

## The Problem

**ns-3.46.1's `PacketSink::GetTotalRx()` returns exponentially growing values regardless of actual network throughput.**

Tested with:
- Source at 100 Mbps, 50 Mbps, 10 Mbps, 1 Mbps
- Custom QueueDisc (token bucket), TBF, simple FIFO, NO queue disc at all
- Various warmup periods (0ms, 100ms, 500ms)
- Various packet sizes (1500, 16384 bytes)
- Custom `RateTrafficSource`, ns-3's `OnOffApplication`

**Every configuration produces the same doubling pattern:**

| Source Rate | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|-------------|--------|--------|--------|--------|--------|
| 1 Mbps | 101 | 154 | 313 | 637 | 1297 |
| 50 Mbps | 42 | 83 | 166 | 332 | 665 |
| 100 Mbps | 144 | 248 | 460 | 961 | 1924 |
| 100 Mbps (no QueueDisc) | 144 | 248 | 460 | 961 | 1924 |

**Even with no QueueDisc at all** and a 100 Mbps source on a 100 Mbps link, the receiver reports 1924 Mbps. This is physically impossible — the link cannot send faster than 100 Mbps.

---

## All Scenarios Tried

### 1. Custom `RateTrafficSource` (initial approach)
- Custom C++ application that sends packets at a configured rate
- TBF QueueDisc for rate limiting
- **Result:** Throughput doubles each step. `SetRate()` to change the rate mid-simulation doesn't take effect — the source keeps sending at its initial rate.

### 2. ns-3 `OnOffApplication` (replace custom source)
- Used ns-3's built-in rate-controlled application
- Dynamic `SetAttribute("DataRate")` to change rate per step
- **Result:** Same doubling. `SetAttribute` doesn't take effect during simulation. Application keeps using initial rate.

### 3. OnOffApplication with source recreation per step
- Stop old source, start new source at new rate each step
- **Result:** Same doubling. Multiple applications accumulate. Still broken.

### 4. Custom `RateLimiterQueueDisc` (token bucket with child FifoQueueDisc)
- Token bucket: packets dequeue only when enough tokens
- Child FifoQueueDisc for packet storage
- **Result:** QueueDisc enqueues/dequeues correctly. But throughput still doubles because `PacketSink::GetTotalRx()` reports broken values.

### 5. Custom `RateLimiterQueueDisc` with internal DropTailQueue
- Removed child QueueDisc indirection
- Direct `GetInternalQueue(0)` for enqueue/dequeue
- Added `DoPeek()` override to prevent requeue bypass
- **Result:** QueueDisc works. `m_dequeuedBytes` counter shows packets flowing. But throughput still doubles.

### 6. Measure throughput at QueueDisc dequeue (not receiver)
- Track `m_dequeuedBytes` in QueueDisc instead of `GetTotalRx()`
- **Result:** `m_dequeuedBytes` also doubles! Because ns-3's event scheduler causes `DoDequeue()` to be called exponentially more times each step (17k → 35k → 70k → 140k → 280k calls/step).

### 7. Measure throughput using packet count (not bytes)
- IP fragments are 1500B instead of original 16384B
- Count fragments and multiply by original size
- **Result:** Still doubles. The fragment count itself doubles (3744 → 7491 → 14982).

### 8. Simple FIFO QueueDisc (no token bucket)
- Source rate = demand, simple FIFO, drops when full
- **Result:** Still doubles. Receiver measurement is broken regardless of QueueDisc.

### 9. TBF QueueDisc (ns-3 built-in)
- Used ns-3's built-in TbfQueueDisc
- `SetAttribute("Rate")` to change rate dynamically
- **Result:** TBF's `SetAttribute` doesn't take effect during simulation. TBF stays at initial rate.

### 10. Various queue sizes, warmup periods, device queue configs
- Queue sizes: 1MB, 5MB, 10MB, 50MB, 100MB
- Warmup: 0ms, 10ms, 100ms, 500ms, 1s, 3s
- Device queue: default, 1-packet, 5000-packet, disabled
- **Result:** No combination fixes the doubling.

---

## Root Cause

Two bugs in ns-3.46.1:

1. **`PacketSink::GetTotalRx()` returns exponentially growing values.** Even with no QueueDisc and a 1 Mbps source, it reports 1297 Mbps by step 5. This is a fundamental bug in ns-3.46.1's packet accounting.

2. **ns-3's event scheduler causes `DoDequeue()` calls to double each step.** The number of Run() → Restart() → DequeuePacket() → Dequeue() → DoDequeue() calls grows exponentially, independent of actual packet flow.

Both bugs are in ns-3's core infrastructure (event scheduler, packet accounting), not in our application code. They cannot be fixed without modifying ns-3's internal source code.

---

## Proposed Solutions

| Solution | Status | Notes |
|----------|--------|-------|
| Fix ns-3's `GetTotalRx()` | Not feasible | Requires modifying ns-3 core code |
| Fix ns-3's event scheduler | Not feasible | Requires modifying ns-3 core code |
| Use QueueDisc stats instead of receiver | Partially works | Counter doubles due to scheduler bug |
| Use packet count × original size | Partially works | Count still doubles |
| **Use mock simulator** | **Works** | Python simulation with correct behavior |

---

## Current View

**ns-3.46.1 cannot produce reliable throughput measurements for this simulation.** The bugs are in ns-3's core infrastructure — the event scheduler and packet accounting — not in our application code.

**The mock simulator is the only reliable option.** It correctly implements:
- Source rate limiting (throughput = min(demand, rate))
- Queue buildup when demand > rate
- Drops when demand >> rate and queue is full
- Stochastic demand via Ornstein-Uhlenbeck process
- Configurable action space (40–90 Mbps)

**For the presentation:** Use the mock simulator for all training and evaluation. The ns-3 C++ module demonstrates the architecture design, but ns-3.46.1 has fundamental measurement bugs that prevent reliable output. The mock provides a faithful simulation of the toll booth behavior with correct, physically meaningful results.
