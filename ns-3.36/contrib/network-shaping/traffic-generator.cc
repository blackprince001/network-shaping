#include "traffic-generator.h"

NS_OBJECT_ENSURE_REGISTERED(TrafficGenerator);

namespace ns3 {

TypeId
TrafficGenerator::GetTypeId(void)
{
  static TypeId tid = TypeId("ns3::TrafficGenerator")
    .SetParent<Object>()
    .SetGroupName("NetworkShaping")
    .AddConstructor<TrafficGenerator>();
  return tid;
}

TrafficGenerator::TrafficGenerator()
  : m_trafficType("constant"),
    m_demandGbps(60.0),
    m_port(7)
{
}

TrafficGenerator::~TrafficGenerator()
{
}

void
TrafficGenerator::Configure(NodeContainer& senders,
                            NodeContainer& receivers,
                            Ipv4InterfaceContainer& receiverIfaces,
                            uint16_t port)
{
  m_port = port;
  m_sinkApps.Stop(Seconds(0));

  uint32_t senderCount = senders.GetN();
  double ratePerSender = (m_demandGbps * 1e9) / senderCount;
  std::string rateStr = std::to_string(static_cast<uint64_t>(ratePerSender)) + "bps";

  PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
                               InetSocketAddress(Ipv4Address::GetAny(), m_port));
  m_sinkApps = sinkHelper.Install(receivers);
  m_sinkApps.Start(Seconds(0.0));

  if (m_trafficType == "constant")
  {
    for (uint32_t i = 0; i < senderCount; ++i)
    {
      OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress());
      onoff.SetAttribute("DataRate", StringValue(rateStr));
      onoff.SetAttribute("PacketSize", UintegerValue(1024));
      onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
      onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
      ApplicationContainer apps = onoff.Install(senders.Get(i));
      m_senderApps.Add(apps);
    }
  }
  else if (m_trafficType == "onoff")
  {
    for (uint32_t i = 0; i < senderCount; ++i)
    {
      OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress());
      onoff.SetAttribute("DataRate", StringValue(rateStr));
      onoff.SetAttribute("PacketSize", UintegerValue(1024));
      onoff.SetAttribute("OnTime", StringValue("ns3::UniformRandomVariable[Min=0.1|Max=1.0]"));
      onoff.SetAttribute("OffTime", StringValue("ns3::UniformRandomVariable[Min=0.1|Max=0.5]"));
      ApplicationContainer apps = onoff.Install(senders.Get(i));
      m_senderApps.Add(apps);
    }
  }
  else if (m_trafficType == "vbr")
  {
    for (uint32_t i = 0; i < senderCount; ++i)
    {
      OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress());
      onoff.SetAttribute("DataRate", StringValue(rateStr));
      onoff.SetAttribute("PacketSize", UintegerValue(512));
      onoff.SetAttribute("OnTime", StringValue("ns3::ParetoRandomVariable[Mean=0.5|Shape=2.5]"));
      onoff.SetAttribute("OffTime", StringValue("ns3::ParetoRandomVariable[Mean=0.3|Shape=2.0]"));
      ApplicationContainer apps = onoff.Install(senders.Get(i));
      m_senderApps.Add(apps);
    }
  }
  else
  {
    for (uint32_t i = 0; i < senderCount; ++i)
    {
      OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress());
      onoff.SetAttribute("DataRate", StringValue(rateStr));
      onoff.SetAttribute("PacketSize", UintegerValue(1024));
      onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
      onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
      ApplicationContainer apps = onoff.Install(senders.Get(i));
      m_senderApps.Add(apps);
    }
  }
}

void
TrafficGenerator::SetDemandGbps(double demandGbps)
{
  m_demandGbps = demandGbps;
}

void
TrafficGenerator::SetTrafficType(const std::string& type)
{
  m_trafficType = type;
}

void
TrafficGenerator::Start(Time start)
{
  m_senderApps.Start(start);
}

void
TrafficGenerator::Stop(Time stop)
{
  m_senderApps.Stop(stop);
  m_sinkApps.Stop(stop);
}

} // namespace ns3
