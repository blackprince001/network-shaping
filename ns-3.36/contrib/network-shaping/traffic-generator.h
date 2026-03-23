#ifndef TRAFFIC_GENERATOR_H
#define TRAFFIC_GENERATOR_H

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include <vector>
#include <string>

namespace ns3 {

class TrafficGenerator : public Object
{
public:
  static TypeId GetTypeId(void);

  TrafficGenerator();
  virtual ~TrafficGenerator();

  void Configure(NodeContainer& senders, NodeContainer& receivers,
                 Ipv4InterfaceContainer& receiverIfaces,
                 uint16_t port);

  void SetDemandGbps(double demandGbps);
  void SetTrafficType(const std::string& type);

  void Start(Time start);
  void Stop(Time stop);

private:
  ApplicationContainer m_senderApps;
  ApplicationContainer m_sinkApps;
  std::string m_trafficType;
  double m_demandGbps;
  uint16_t m_port;
};

} // namespace ns3

#endif // TRAFFIC_GENERATOR_H
