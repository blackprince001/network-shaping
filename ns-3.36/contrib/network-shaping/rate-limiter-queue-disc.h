#ifndef RATE_LIMITER_QUEUE_DISC_H
#define RATE_LIMITER_QUEUE_DISC_H

#include "ns3/queue-disc.h"
#include "ns3/data-rate.h"

namespace ns3 {

/**
 * \ingroup traffic-control
 *
 * Token bucket rate limiter queue disc.
 * Similar to TBF but with:
 * - Guaranteed token reset on rate change
 * - Larger configurable internal queue
 * - Direct access to stats for Python integration
 */
class RateLimiterQueueDisc : public QueueDisc
{
public:
  static TypeId GetTypeId(void);

  RateLimiterQueueDisc();
  virtual ~RateLimiterQueueDisc();

  void SetRate(DataRate rate);
  DataRate GetRate() const;

  void SetBurst(uint32_t burst);
  uint32_t GetBurst() const;

protected:
  virtual bool CheckConfig(void);
  virtual bool DoEnqueue(Ptr<QueueDiscItem> item);
  virtual Ptr<QueueDiscItem> DoDequeue(void);
  virtual Ptr<const QueueDiscItem> DoPeek(void);
  virtual void InitializeParams(void);

private:
  void ReplenishTokens(void);

  DataRate m_rate;          // Token refill rate
  uint32_t m_burst;         // Maximum burst (bytes)
  uint32_t m_mtu;           // MTU for peak rate calc
  double m_btokens;         // Current burst tokens
  Time m_lastReplenish;     // Last token replenishment time
};

} // namespace ns3

#endif // RATE_LIMITER_QUEUE_DISC_H
