#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Adaptive Network Traffic Shaping with Deep Reinforcement Learning: A PPO-Driven Token Bucket on ns-3],
  abstract: [
    Static Token Bucket Filters (TBFs) enforce a single
    contracted rate regardless of how traffic actually behaves. Set
    the rate too low and the shaper starves bursts and triggers TCP
    retransmissions; set it too high and bursts pass through
    unsmoothed and congest downstream queues. Real internet demand is
    non-stationary, so any fixed rate is wrong for most of the day.

    We replace the fixed rate with a learned policy. A
    Proximal Policy Optimization (PPO) agent observes a
    four-dimensional vector (queue, throughput, drop intensity,
    demand) and emits a continuous TBF rate every second. The agent
    is trained inside a custom Gymnasium environment that drives an
    ns-3 bottleneck topology, using a four-level curriculum that
    progresses from constant CBR through bursty on-off, mixed
    elephant/mice flows, and finally three weeks of real Cloudflare
    Radar traffic. We evaluate two action shapes, a wide variant
    ($a in [1, 100]$ Mbps) and a narrow variant
    ($a in [40, 90]$ Mbps), against six static baselines spanning
    30--80 Mbps.

    On real-world demand, the wide-action agent holds
    queue occupancy and drops far below the aggressive 30~Mbps static
    cap; the narrow-action agent recovers most of the throughput of a
    high static cap at substantially lower rate oscillation and
    roughly 28% fewer drops than the 30~Mbps cap. Stress tests at
    2$times$ and 3$times$ nominal demand show that both agents
    degrade gracefully, concentrating loss into narrow burst windows
    rather than spreading it uniformly across the trace.

    The compact four-scalar observation is
    sufficient to learn a non-trivial rate-control policy without
    per-flow state, packet traces, or topology maps, which is
    significant because four scalars are exactly what a production
    shaper can realistically expose to a control plane. An RL-shaped
    TBF replaces a contract with the past (a fixed rate fitted to
    yesterday's profile) with a contract over states, generalising
    across diurnal cycles without operator intervention.

    The action ceiling becomes the binding constraint
    under extreme overload, so dynamic action ranges and recurrent
    policies are the highest-impact next steps. Beyond the shaper
    itself, the most interesting direction is composition with
    adjacent ML controllers: pricing-driven demand shaping,
    application-layer encoders, and downstream router-queue
    controllers, jointly trained against a user-visible QoS reward.
  ],
  authors: (
    (
      name: "Prince Kwabena Appiah Boadu",
      department: [Department of Computer Engineering],
      organization: [Kwame Nkrumah University of Science and Technology],
      location: [Kumasi, Ghana],
      email: "pkappiahboadu@st.knust.edu.gh",
    ),
  ),
  index-terms: (
    "Reinforcement learning",
    "Traffic shaping",
    "Token bucket filter",
    "Network simulation",
    "ns-3",
    "PPO",
    "Quality of service",
  ),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction

Quality-of-Service (QoS) on a shared link is built on three primitives:
traffic contracts, which fix a Committed Information Rate (CIR) and a
burst allowance between a customer and a provider; policers, which
monitor offered traffic and drop or re-mark anything in excess; and
shapers, which absorb excess into a buffer and drain it at the
contracted rate, smoothing bursts at the cost of latency
@rfc2475 @tanenbaum2010networks. Among shapers, the Token Bucket Filter
(TBF) remains the canonical implementation in Linux's `tc` and in most
core-router QoS pipelines. Its parameters #emph[rate] and
#emph[burst] are set once and assumed to be a faithful summary of the
traffic profile they govern.

Real internet demand is not stationary. Cloudflare Radar shows backbone
traffic oscillating between roughly 10% and 100% of the daily peak
within a single 24-hour cycle @cloudflare-radar (@fig:diurnal). A
single static rate is therefore wrong for most of the day. Set it to
the mean and bursts overflow the bucket, triggering TCP retransmissions
that further amplify congestion. Set it to the peak and most of the
shaping action disappears, leaving downstream queues to absorb whatever
arrives. The operator's "safe" middle is a moving target.

#figure(
  image("figures/fig_cloudflare_profile.pdf", width: 100%),
  caption: [Three weeks of normalized hourly Cloudflare Radar traffic
    for a representative ASN. Demand swings between roughly 10% and
    100% of peak within a daily cycle. A static rate cap is correct
    only on the narrow band of hours where the curve crosses it.],
) <fig:diurnal>

This paper reports on an end-to-end system that replaces the fixed
TBF rate with a continuously learned one. A PPO agent
@schulman2017ppo observes a four-scalar summary of the bottleneck
(queue, throughput, drop intensity, current demand), emits a
continuous TBF rate in $[1, 100]$ Mbps once per second, and is trained
inside a custom Gymnasium @brockman2016openai environment that drives
an ns-3 @ns3 bottleneck. The agent is trained with a curriculum that
ends in real Cloudflare Radar traffic, then evaluated A/B against
static rate caps under both nominal and 2--3$times$ overload.

Our contributions are:
+ a clean, subprocess-isolated bridge between Stable-Baselines3 PPO
  @stable-baselines3 and ns-3 that supports parallel rollout workers;
+ a normalized, bounded multi-objective reward that keeps PPO's value
  head numerically well-conditioned across very different traffic
  regimes;
+ an A/B evaluation against six static baselines on real and
  stress-scaled traffic;
+ a precise account of where this approach holds and where it does
  not.

= Background and Related Work

The shared theme across recent ML-for-networking work is that static,
human-coded rules are no longer adequate for the dynamic regimes that
modern traffic produces @dulaj2025ml. Reinforcement learning has
become the dominant tool because the underlying control problems are
sequential decisions under uncertainty, exactly the setting where
trial-and-error in simulation pays off. We organise the most relevant
prior work by the layer of the stack it touches.

*Application layer.* Pensieve @mao2017pensieve frames adaptive video
bitrate selection as an MDP and learns a policy that beats hand-tuned
heuristics. Ahmed et al. @ahmed2023vanet push the same idea into
5G-VANET multimedia streaming, using distributed RL to jointly tune
quantization, GoP size, and frame rate to match an unstable wireless
channel. Thompson et al. @thompson2024gaming use real-time telemetry
plus session metadata (player count, input frequency) to prioritise
latency-critical game packets. All three optimise #emph[what is sent]
and how it is encoded.

*Transport and scheduling layer.* Jay et al. @jay2019deep treat
congestion control itself as an RL problem at the endpoint. Liao et
al. @liao2026agcs target Time-Sensitive Networking, using DRL to
compress Gate Control Lists by up to 69.5% so they fit in real switch
memory while preserving deadlines. Wang et al. @wang2023tsn5g hybridise
Double Q-learning with particle swarm optimisation to escape local
optima in NP-complete TSN-5G end-to-end scheduling.

*Infrastructure and queueing layer.* Iroko @ruffy2019iroko is the
closest in spirit to our work: an RL framework for prototyping
data-centre traffic-control policies. Kattepur et al.
@kattepur2021router cast router port-queue configuration as a POMDP
and let a model-based RL agent replace the manual "trial-and-error"
tuning that 5G slicing makes intractable. Shames et al.
@shames2006shaper, the earliest precedent we found, used a
Q-learning agent with a small neural network to learn token-generation
rates for a shaper from drop percentage and buffer occupancy. They
showed that even a simple RL controller can outperform a fixed
TBF when the topology shifts; our work updates this two-decade-old
idea with modern policy optimisation, real Cloudflare demand, and
ns-3 ground truth.

*Demand-side and economic layer.* SHIFT @choi2025shift takes a
different lever altogether: a multi-agent DRL pricing scheme that
shifts user demand spatially and temporally, flattening peak load
without any new hardware. This complements rather than competes with
shaping; pricing reduces the volume the shaper has to handle.

*Where this paper fits.* The papers above span the stack from
application encoding to economic incentives, but they leave a gap at
the egress-shaper layer: the single point where a contracted SLA is
mechanically enforced. Shames et al. addressed this gap in 2006 with
tabular Q-learning on a synthetic network; we revisit it with PPO,
a real ns-3 simulator, and live Cloudflare demand traces. Compared to
@kattepur2021router and @ruffy2019iroko, our agent operates on a
deliberately impoverished observation space (four scalars; no
per-flow state; no topology map), which is what an operator can
realistically expose from a production shaper without a deep
telemetry pipeline. Compared to @shames2006shaper, we extend from a
single static topology to a four-level curriculum that ends in real
diurnal traffic, and we use clipped policy updates @schulman2017ppo
to avoid the cascade of TCP timeout storms that an unconstrained
update (e.g.~rate jumping from 90 to 10~Mbps in one step) would
otherwise trigger.

*What this paper does.* Concretely, we (i) build a reproducible
PPO + ns-3 + Gymnasium loop with a one-shot subprocess bridge that
removes the long-lived-simulator failure modes of prior bespoke
integrations; (ii) propose a normalized, bounded multi-objective
reward that keeps the value head well-conditioned across very
different traffic regimes; (iii) characterise the failure modes of
static TBFs across the 30--80~Mbps range under real Cloudflare
demand; and (iv) provide an A/B comparison of two action shapes
(wide and narrow) under nominal and 2--3$times$ overload, including
an honest account of where the learned policy degrades.

= System Design

== Architecture

The system is a tight feedback loop between a Python policy and a C++
simulator (@fig:arch). At each control interval, the Gymnasium
environment emits an action (a TBF rate), spawns ns-3 as a one-shot
subprocess with that rate as a CLI argument, ingests one CSV line of
telemetry from the subprocess's stdout, and returns the next
observation and reward.

```bash
./ns3-sim --rate=52.5Mbps --burst=6500000 --source=80.0Mbps --duration=1.0
```

This subprocess model is deliberate. Every step is a fully isolated
process with no shared state, no port-binding conflicts, and no
long-lived simulator state to corrupt. Parallel rollout workers come
for free: the only contention is the OS process table.

#figure(
  placement: top,
  kind: image,
  supplement: [Fig.],
  block(
    width: 100%,
    inset: 6pt,
    stroke: 0.4pt + luma(60%),
    radius: 3pt,
    [
      #set text(size: 8pt)
      #grid(
        columns: (1fr, auto, 1fr),
        column-gutter: 8pt,
        align: (left + horizon, center + horizon, left + horizon),
        [
          *Python (Gymnasium)* \
          • `Ns3Env` wrapper \
          • PPO policy (SB3) \
          • Cloudflare loader
        ],
        [
          $arrow.r$ rate \
          $arrow.l$ obs
        ],
        [
          *ns-3 (C++)* \
          • TBF QueueDisc \
          • `RateTrafficSource` \
          • Bottleneck topology
        ],
      )
    ],
  ),
  caption: [Subprocess bridge between the Python policy and the C++
    simulator. Each control interval is a fresh ns-3 process invoked
    with the agent's chosen rate.],
) <fig:arch>

== ns-3 Bottleneck

The simulated topology is a single bottleneck link of 100~Mbps
physical capacity, with the TBF queue discipline acting as the
logical bottleneck where shaping happens. Senders inject traffic via
ns-3's `OnOffHelper` at a CBR rate that is itself driven from the
Cloudflare hourly demand series, so the agent's input traffic mirrors
a real ASN profile. Telemetry is sampled exclusively at the egress of
the TBF (queue depth, throughput, and drop count), so the control
policy sees only what an operator could plausibly export from a real
shaper.

The TBF #emph[rate] is rewritten every second from the agent's
action. The #emph[burst] is derived as a fixed multiple
($0.1 dot.c "rate" dot.c 1 "s"$) of the chosen rate, which keeps the
bucket window at a sensible 100~ms of the controlled rate and avoids
adding a second action dimension that the agent would otherwise need
to learn jointly.

== Observation, Action, Reward

The observation is normalized to $[0, 1]^4$:
$ o_t = (q_t / Q_max, T_t / T_max, tanh(d_t / d_max), D_t / D_max), $
where $q_t$ is queue occupancy (bytes), $T_t$ throughput (Mbps),
$d_t$ drop count over the interval, and $D_t$ the current demand.
Bounds ($Q_max = 5$~MB, $T_max = 100$~Mbps, $d_max = 100$,
$D_max = 100$~Mbps) are physical link properties or simulator caps
and are kept constant across curriculum levels. The agent never sees
raw byte counts or unscaled rates; it sees pressure signals.

The action space is a single continuous variable, the TBF rate in
Mbps. We compare two action shapes:
- *Wide*: $a_t in [1, 100]$ Mbps (the full physical range);
- *Narrow*: $a_t in [40, 90]$ Mbps (the operationally realistic
  band).

The reward combines three terms:
$ R_t = (alpha T_("norm") - beta Q_("norm") - gamma D_("norm")) / (alpha + beta + gamma) $ <eq:reward>
with $alpha = 1.0$, $beta = 0.5$, $gamma = 0.8$. Throughput and queue
are min--max scaled against their bounds; drops are passed through
$tanh$ rather than scaled linearly so the gradient remains sharp near
zero (where it must distinguish "rare drops" from "no drops") and
saturates at large values (so a buffer overflow during early training
does not produce a $-100$ reward that derails the value function).
Dividing by $alpha + beta + gamma$ keeps $R_t in [-1, +1]$, which
decouples PPO's learning rate from the choice of weights.

== Curriculum Training

The agent is trained across four progressively harder environments
(@tab:curriculum), with weights carried forward from one level
to the next. Each level adds one new failure mode that the previous
level did not expose.

#figure(
  placement: top,
  caption: [Training curriculum. Each level adds one new failure
    mode that the previous level did not expose.],
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    inset: (x: 6pt, y: 3pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt, bottom: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#f3f3f3") },
    table.header[Level][Traffic][Demand][Focus],
    [1. Basic], [Constant CBR], [60 Mbps], [Rate--demand match],
    [2. Bursty], [On--Off], [60 Mbps], [Temporal awareness],
    [3. Chaotic], [Elephant + mice], [80 Mbps], [Noise robustness],
    [4. Real], [Cloudflare Radar], [Real-world], [Generalization],
  ),
) <tab:curriculum>

For hyperparameter search and rapid iteration we additionally use a
mock simulator based on an Ornstein--Uhlenbeck process
@uhlenbeck1930ou,
$ d x_t = theta (mu - x_t) d t + sigma d W_t, $
with $mu = 70$~Mbps, $theta = 0.10$, $sigma = 20$~Mbps, plus a 20%
per-step probability of an additive 20--50~Mbps spike. The mock is
roughly two orders of magnitude faster than ns-3 and produces
trajectories that drift, spike, and recover in patterns that
genuinely resist memorization.

Hyperparameters used for the runs reported in this paper are
collected in @tab:hparams. Levels 1--3 train for 50--80~k environment
steps each; level 4 (real Cloudflare traffic) trains for an
additional 50~k steps starting from the level-3 weights. Total
wall-clock training time on a 16-core commodity workstation is
approximately 9~hours when running 8 ns-3 rollouts in parallel.

#figure(
  placement: top,
  caption: [PPO and environment hyperparameters. Values shared
    across curriculum levels except where noted.],
  table(
    columns: (auto, auto),
    align: (left, left),
    inset: (x: 6pt, y: 3pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt, bottom: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#f3f3f3") },
    table.header[Parameter][Value],
    [Discount $gamma$], [0.99],
    [PPO clip range], [0.2 (0.15 at level 4)],
    [Learning rate], [3e-4 (5e-5 at level 4)],
    [Rollout length $n$], [512 (256 at level 4)],
    [Minibatch size], [64],
    [GAE $lambda$], [0.95],
    [Entropy coefficient], [0.0],
    [Reward weights ($alpha,beta,gamma$)], [$(1.0,\ 0.5,\ 0.8)$],
    [Burst multiplier], [0.1 of rate],
    [Control interval], [1 s],
    [Episode length], [50 steps],
    [Queue ceiling $Q_max$], [5 MB],
    [Drop normalizer $d_max$], [100],
    [Action range (Wide)], [$[1, 100]$ Mbps],
    [Action range (Narrow)], [$[40, 90]$ Mbps],
  ),
) <tab:hparams>

#figure(
  placement: top,
  kind: "algorithm",
  supplement: [Algorithm],
  align(left, block(
    width: 100%,
    inset: 6pt,
    stroke: 0.4pt + luma(60%),
    radius: 3pt,
    [
      #set text(size: 8pt)
      #set par(leading: 0.55em)
      #set align(left)
      *Input:* policy $pi_theta$, value head $V_phi$, curriculum
      $cal(C) = (C_1, C_2, C_3, C_4)$, ns-3 binary $E$ \
      *for* level $C$ in $cal(C)$ *do* \
      #h(1.0em) *for* $i = 1, ..., n$ *do* \
      #h(2.0em) sample action $a_t tilde.op pi_theta (dot.c | o_t)$ \
      #h(2.0em) spawn $E$#sub[$C$] with rate $a_t$, demand $D_t$, $Delta t = 1$~s \
      #h(2.0em) read $(q_(t+1), T_(t+1), d_(t+1))$ from stdout \
      #h(2.0em) compute $R_t$ via @eq:reward; store
      $(o_t, a_t, R_t, o_(t+1))$ \
      #h(1.0em) update $(theta, phi)$ via PPO clipped objective \
      transfer $(theta, phi)$ to next level
    ],
  )),
  caption: [Curriculum PPO training loop. Each environment step is a
    fresh ns-3 subprocess; rollout buffers are populated in parallel
    across multiple workers.],
) <alg:train>

= Failure Modes of Static Shaping

Before introducing the agent, we map how a static TBF behaves under
real Cloudflare demand (@fig:static). The story is a clean
trichotomy.

#figure(
  placement: bottom,
  image("figures/fig_static_baselines.pdf", width: 100%),
  caption: [Static TBF baselines on a 100-second window of real
    Cloudflare demand at four fixed rates. (a) Throughput is capped
    at the rate ceiling; (b) queue occupancy spikes at low rates as
    the bucket overflows; (c) cumulative drops climb sharply for the
    30 and 40~Mbps caps and stay near zero only at 70~Mbps and
    above.],
  // scope: "parent",
) <fig:static>

*Low-rate starvation (30 / 40~Mbps).* Throughput is capped well below
available demand for most of the cycle. The bucket overflows in peak
hours (@fig:static b), and cumulative drops climb monotonically
(@fig:static c). For TCP this triggers retransmissions that
compound into a congestion spiral; for VoIP and video, the sustained
queue delay is itself an SLA failure.

*Middle-ground compromise (50 / 60~Mbps).* The "operator default":
eliminates the worst overflows but introduces a different failure
mode. Off-peak hours leave significant idle capacity, and unexpectedly
large surges still produce occasional drops. Drop counts are reduced
an order of magnitude relative to 30~Mbps but are not zero.

*High-rate permissiveness (70 / 80~Mbps).* Drops essentially
disappear, but the shaper has ceded any control over burst behaviour.
During heavy-hitter periods the queue still builds, just more slowly,
and the marginal throughput gain over 50~Mbps is smaller than the
marginal control loss.

The narrow band of "acceptable" static rates therefore shifts with
the time of day. This motivates an adaptive policy.

The trichotomy is also visible in classical operator playbooks:
practitioners choose static rates by inspecting historic peak/95th
percentile traffic and adding margin. That margin is wasted capacity
when the link is idle, and insufficient margin when traffic exceeds
the historic peak. The adaptive policy we propose makes this margin
state-conditional rather than time-invariant.

= Evaluation

We evaluate two trained variants, Wide ($a in [1, 100]$~Mbps) and
Narrow ($a in [40, 90]$~Mbps), against the 30, 50, and 70~Mbps
static baselines. All runs use the same real-traffic input; the only
difference is the rate-selection policy. Each window is 100 seconds
of simulated time at one-second control intervals, repeated
identically across scenarios so the demand trace is held constant.

== Real Traffic, Nominal Intensity

@fig:wide-ts shows the wide-action agent's trajectory on the
real-traffic window. Throughput tracks demand closely; the agent
periodically dips slightly below current demand to flush queue
buildup before it crosses into a drop event. Queue and drop traces
are correspondingly bursty; the agent is active rather than
conservative.

#figure(
  placement: top,
  image("figures/fig_ab_wide_timeseries.pdf", width: 80%),
  caption: [Wide-action agent versus static baselines on the
    real-traffic window. Top: throughput; middle: queue occupancy;
    bottom: per-step drops. The 70~Mbps baseline tops the throughput
    ranking but at the cost of zero adaptive control.],
  scope: "parent",
) <fig:wide-ts>

The narrow-action agent (omitted as a separate time-series figure for
brevity) produces visibly smoother rate curves, fewer large
corrections, and a tighter orbit around the demand signal, at the
expense of slightly slower reaction to abrupt spikes.

Aggregate behaviour is shown in @fig:summary. Three takeaways:

+ The Wide agent's average throughput sits well below the 70~Mbps cap
  because it spends much of its time at rates slightly below current
  demand to keep queue depth low. It is the #emph[strategy] that
  distinguishes the agent, not the headline throughput number.
+ The Narrow agent recovers most of the throughput while still cutting
  drops compared to the 30~Mbps static cap by roughly 28%.
+ Both agents trade some headline throughput for substantially reduced
  drop counts versus the low static rates that an operator would
  choose if they were prioritising "safety" over capacity.

The crucial qualitative difference between the agent and the static
baselines is not the average value on any single metric but the
#emph[shape] of the trajectories. The 70~Mbps cap maximises raw
throughput by ignoring queue depth entirely; the agent's throughput
is lower precisely because it actively flushes the buffer. This is a
strategy choice, not a performance loss. An operator who values low
end-to-end latency on a saturated link will read these numbers as a
win for the agent; one who only contracts for raw bytes-per-second
will not.

#figure(
  placement: bottom,
  image("figures/fig_summary_compare.pdf", width: 80%),
  caption: [Aggregate metrics across baselines and the two agents.
    Solid bars: wide action shape ($[1, 100]$~Mbps). Hatched bars:
    narrow action shape ($[40, 90]$~Mbps). Throughput, queue, and
    drops are reported as 100-second window averages or totals.],
  scope: "parent",
) <fig:summary>

== Stress: 2$times$ and 3$times$ Demand

The deployment question is generalisation. We scale the demand input
by 2$times$ and 3$times$ beyond the training distribution
(@fig:stress).

#figure(
  placement: top,
  image("figures/fig_stress_summary.pdf", width: 80%),
  caption: [Narrow-action agent under nominal (1$times$), 2$times$,
    and 3$times$ demand. Throughput and reward both collapse for all
    policies as the link saturates, but drop counts rise uniformly
    across baselines, indicating loss in the saturated regime is
    dominated by physics rather than policy.],
  scope: "parent",
) <fig:stress>

At 2$times$ demand the link is approaching saturation. The agent's
rate sits near the upper boundary of its action space far more
frequently than under nominal traffic; momentary buffer builds appear
in the queue trace but recover within a few control intervals
without cascading into sustained loss. At 3$times$ demand, physics
wins: genuine link saturation produces drop counts on the order of
$10^5$ per window for every policy, including the agent. The agent
does not eliminate loss in this regime; what it does is concentrate
loss into narrow burst windows rather than spread it uniformly across
the window. This is the honest limit of the design; a constrained
action space cannot reach above its ceiling, and the narrow agent's
ceiling becomes a real constraint.

= Discussion and Limitations

== Engineering Notes from the Implementation

Two engineering decisions deserve highlighting because they materially
affected reproducibility. First, our initial implementation kept ns-3
running as a long-lived subprocess and communicated rate updates over
a UNIX pipe. This intermittently froze the simulator: ns-3 buffers
its stdout in default mode, and the Python parent would block on a
read that the child had not yet flushed. Switching to a one-shot
subprocess per step eliminated the freeze entirely at a cost of about
$30$~ms of process-spawn overhead per step. Second, packet-counter
inconsistencies between TCP and the TBF queue discipline (driven
by ns-3's segmentation behaviour at the sender) produced reward
spikes during the first second of every episode. Resetting the
counters at episode boundaries rather than relying on running totals
solved the problem.

Roughly two-thirds of the development time on this project went into
plumbing of this kind: stable telemetry, deterministic seeding across
parallel workers, and CSV schema versioning to survive incremental
changes to the ns-3 build. None of it is novel science, but it is
where reproducibility lives.


== What the Result Means

The compact four-scalar observation is the result we set out to
validate, and the A/B evaluation supports it. The agent has no flow
table, no DPI, no topology map. It distils an entire bottleneck into
four numbers (queue, throughput, drop intensity, demand) and acts on
them. That this is sufficient for non-trivial rate control on real
traffic is the core finding.

The narrow-versus-wide comparison is more pragmatic than scientific:
constraining the action space to the operationally realistic band
trades exploration coverage for convergence speed and policy
smoothness. In production, the narrow agent is the one we would
deploy; the wide agent's wider exploration is principally a
diagnostic for what the policy #emph[would] do given the freedom.

*Limitations.* The single-bottleneck topology elides multi-tenant
fairness; the egress-only telemetry assumes per-class metrics are
exposed; and the per-step subprocess model adds a non-trivial fixed
cost (≈ 30--80~ms wall-clock per step on commodity hardware) that
limits how fast a deployed control loop can run. None of these are
inherent to the approach; they are engineering choices that fit the
simulator we had access to.

*Future work.* Three directions are worth pursuing in order of impact:
- *Recurrent policies.* An LSTM head over the four-scalar
  observation would give the agent an explicit short-term memory of
  recent queue trajectories, which the current MLP must reconstruct
  from instantaneous state.
- *Latency-aware reward.* Queue depth is a leading indicator of
  latency, but it is not latency. Adding a direct delay term to
  @eq:reward would let the agent trade queue depth against actual
  end-to-end delay.
- *Dynamic action ceilings.* The 3$times$ stress regime shows that a
  fixed upper bound on rate is the binding constraint at extreme
  overload. A policy that scales its action range with observed
  demand would handle this without retraining.
- *Composition with adjacent ML controllers.* The most interesting
  next step is composition rather than a bigger model. Pricing-based
  demand shaping @choi2025shift could feed our shaper a partially
  flattened arrival process; an application-layer RL encoder
  @ahmed2023vanet @thompson2024gaming could co-adapt video or game
  bitrates against the rate our agent advertises; and a downstream
  router-queue controller @kattepur2021router could absorb the
  residual bursts our shaper does not flatten. Each of these has been
  studied in isolation, but the joint training problem (multiple
  RL agents at different layers of the same path, sharing a reward
  that is anchored to user-visible QoS) is open.
- *Hardware-aware deployment.* Time-sensitive scheduling work like
  AGCS @liao2026agcs and DQHPSO @wang2023tsn5g shows that real
  switches impose hard memory and timing constraints that a
  software-only simulator hides. Porting our policy onto a programmable
  data-plane (P4 / DPDK) would expose those constraints and force a
  more honest cost model than the 30--80~ms subprocess overhead we
  currently pay.

= Conclusion

A static TBF is a contract with the past; it commits the network
to a rate set under the assumption that yesterday's traffic profile
is representative of today's. An RL-shaped TBF is a different kind
of contract: it commits to a #emph[policy] over states rather than a
specific rate, and so handles non-stationary traffic without
operator intervention. On real Cloudflare demand, our PPO agent
delivers competitive throughput with substantially reduced drop
counts versus the low and middle static rates that operators
typically deploy. Under 2--3$times$ overload it degrades gracefully
rather than catastrophically. The four-scalar observation is enough,
and the subprocess-isolated training loop is fast enough to make
this a reproducible, single-machine experiment. Source code and
data are available at
#link("https://github.com/blackprince001/network-shaping").
