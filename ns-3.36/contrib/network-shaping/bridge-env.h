#ifndef BRIDGE_ENV_H
#define BRIDGE_ENV_H

#include "ns3/core-module.h"
#include <string>

namespace ns3 {

class BridgeEnv : public Object
{
public:
  static TypeId GetTypeId(void);

  BridgeEnv();
  virtual ~BridgeEnv();

  void SetTbfQueueDisc(Ptr<QueueDisc> qdisc);
  void SetBottleneckDevice(Ptr<NetDevice> dev);

  void SendReady(void);
  void HandleStep(void);
  void HandleStop(void);

private:
  void DoSendMetrics(uint32_t queueBytes, double throughputMbps, uint32_t drops);
  void RescheduleStep(void);

  Ptr<QueueDisc> m_tbfQueueDisc;
  Ptr<NetDevice> m_bottleneckDevice;

  uint32_t m_totalBytes;
  uint32_t m_prevTotalBytes;
  uint32_t m_totalDrops;
  double m_lastThroughputMbps;

  std::string m_message;
};

} // namespace ns3

#endif // BRIDGE_ENV_H
