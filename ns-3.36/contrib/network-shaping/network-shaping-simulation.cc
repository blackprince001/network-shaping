/*
 * network-shaping-simulation.cc — One-shot toll booth
 *
 * Invoked once per step. Runs for a fixed duration, outputs metrics, exits.
 * No Stop/Run loop. No stdin protocol. No state carryover.
 *
 * Usage:
 *   ./ns3.46.1-network-shaping-simulation-optimized \
 *     --rate=50Mbps --burst=5000000 --source=80Mbps --duration=1
 *
 * Output (stdout):
 *   <queue_bytes>,<throughput_mbps>,<drops>
 *
 * Topology:
 *   n0 (source@source_rate) --[RateLimiterQueueDisc]-- 100Mbps/5ms -- n1 (sink)
 *
 * The rate limiter IS the bottleneck. The 100 Mbps link is just wire.
 * Throughput = min(source_rate, rate_limit).
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/applications-module.h"
#include "ns3/rate-limiter-queue-disc.h"

#include <iostream>
#include <string>
#include <cstdint>

using namespace ns3;

int
main(int argc, char* argv[])
{
    std::string rateStr = "50Mbps";
    uint32_t burst = 5000000;
    std::string sourceStr = "80Mbps";
    double duration = 1.0;

    CommandLine cmd(__FILE__);
    cmd.AddValue("rate", "Rate limiter rate (e.g. 50Mbps)", rateStr);
    cmd.AddValue("burst", "Rate limiter burst in bytes", burst);
    cmd.AddValue("source", "Source data rate (e.g. 80Mbps)", sourceStr);
    cmd.AddValue("duration", "Simulation duration in seconds", duration);
    cmd.Parse(argc, argv);

    // Build topology
    NodeContainer nodes;
    nodes.Create(2);

    InternetStackHelper stack;
    stack.Install(nodes);

    PointToPointHelper link;
    link.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    link.SetChannelAttribute("Delay", StringValue("5ms"));
    NetDeviceContainer devices = link.Install(nodes);

    // Rate limiter on sender
    TrafficControlHelper tch;
    tch.SetRootQueueDisc("ns3::RateLimiterQueueDisc",
                         "Burst", UintegerValue(burst),
                         "Rate", DataRateValue(DataRate(rateStr)));
    QueueDiscContainer qdiscs = tch.Install(devices.Get(0));
    Ptr<RateLimiterQueueDisc> qdisc = DynamicCast<RateLimiterQueueDisc>(qdiscs.Get(0));

    Ipv4AddressHelper ip;
    ip.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer ifaces = ip.Assign(devices);

    // Sink
    PacketSinkHelper sink("ns3::UdpSocketFactory",
                          InetSocketAddress(Ipv4Address::GetAny(), 9));
    ApplicationContainer sinkApp = sink.Install(nodes.Get(1));
    sinkApp.Start(Seconds(0.0));
    Ptr<PacketSink> rx = DynamicCast<PacketSink>(sinkApp.Get(0));

    // Source at configured rate
    OnOffHelper onoff("ns3::UdpSocketFactory",
                      InetSocketAddress(ifaces.GetAddress(1), 9));
    onoff.SetAttribute("DataRate", StringValue(sourceStr));
    onoff.SetAttribute("PacketSize", UintegerValue(1024));
    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    ApplicationContainer apps = onoff.Install(nodes.Get(0));
    apps.Start(Seconds(0.0));

    // Run for specified duration
    Simulator::Stop(Seconds(duration));
    Simulator::Run();

    // Collect metrics
    QueueDisc::Stats stats = qdisc ? qdisc->GetStats() : QueueDisc::Stats();
    uint64_t totalRx = rx->GetTotalRx();

    double throughputMbps = (totalRx * 8.0) / (duration * 1e6);
    uint32_t queueBytes = qdisc ? qdisc->GetNBytes() : 0;
    uint32_t drops = stats.nTotalDroppedPackets;

    // Output: queue,throughput,drops
    std::cout << queueBytes << "," << throughputMbps << "," << drops << std::endl;

    Simulator::Destroy();
    return 0;
}
