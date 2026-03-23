#include "bridge-env.h"
#include "ns3/traffic-control-module.h"
#include "ns3/point-to-point-module.h"
#include <iostream>
#include <sstream>

NS_OBJECT_ENSURE_REGISTERED(BridgeEnv);

namespace ns3 {

TypeId
BridgeEnv::GetTypeId(void)
{
  static TypeId tid = TypeId("ns3::BridgeEnv")
    .SetParent<Object>()
    .SetGroupName("NetworkShaping")
    .AddConstructor<BridgeEnv>();
  return tid;
}

BridgeEnv::BridgeEnv()
  : m_tbfQueueDisc(nullptr),
    m_bottleneckDevice(nullptr),
    m_totalBytes(0),
    m_prevTotalBytes(0),
    m_totalDrops(0),
    m_lastThroughputMbps(0.0)
{
}

BridgeEnv::~BridgeEnv()
{
}

void
BridgeEnv::SetTbfQueueDisc(Ptr<QueueDisc> qdisc)
{
  m_tbfQueueDisc = qdisc;
}

void
BridgeEnv::SetBottleneckDevice(Ptr<NetDevice> dev)
{
  m_bottleneckDevice = dev;
}

void
BridgeEnv::SendReady(void)
{
  ::std::cout << "READY" << std::endl;
  ::std::cout.flush();
}

void
BridgeEnv::HandleStep(void)
{
  ::std::string msg;
  ::std::getline(::std::cin, msg);

  if (msg == "STOP" || msg.empty())
  {
    ::std::cout << "BYE" << std::endl;
    ::std::cout.flush();
    Simulator::Stop();
    return;
  }

  if (msg.substr(0, 4) != "STEP")
  {
    Simulator::Schedule(Seconds(0.001), &BridgeEnv::HandleStep, this);
    return;
  }

  size_t comma1 = msg.find(',', 5);
  size_t comma2 = msg.find(',', comma1 + 1);
  double rateGps = ::std::stod(msg.substr(5, comma1 - 5));
  double burstMb = ::std::stod(msg.substr(comma1 + 1, comma2 - comma1 - 1));

  uint64_t rateBps = static_cast<uint64_t>(rateGps * 1e9);
  uint64_t burstBytes = static_cast<uint64_t>(burstMb * 1e6);

  if (m_tbfQueueDisc)
  {
    Ptr<TbfQueueDisc> tbf = DynamicCast<TbfQueueDisc>(m_tbfQueueDisc);
    if (tbf)
    {
      tbf->SetAttribute("Rate", DataRateValue(DataRate(rateBps)));
      tbf->SetAttribute("Burst", UintegerValue(burstBytes));
    }
  }

  uint32_t queueBytes = m_tbfQueueDisc ? m_tbfQueueDisc->GetNBytes() : 0;

  uint64_t txBytes = m_bottleneckDevice ? m_bottleneckDevice->GetTransmitBytes() : 0;
  uint32_t stepBytes = (txBytes >= m_prevTotalBytes)
                           ? static_cast<uint32_t>(txBytes - m_prevTotalBytes)
                           : 0;
  m_prevTotalBytes = static_cast<uint32_t>(txBytes);

  double throughputMbps = (stepBytes * 8.0) / 1e6;

  uint32_t drops = 0;
  if (m_tbfQueueDisc)
  {
    QueueDisc::Stats stats = m_tbfQueueDisc->GetStats();
    drops = stats.nTotalDroppedPackets;
  }

  ::std::cout << queueBytes << "," << throughputMbps << "," << drops << std::endl;
  ::std::cout.flush();

  Simulator::Schedule(Seconds(1.0), &BridgeEnv::HandleStep, this);
}

void
BridgeEnv::HandleStop(void)
{
  ::std::cout << "BYE" << std::endl;
  ::std::cout.flush();
  Simulator::Stop();
}

} // namespace ns3
