#include "rate-limiter-queue-disc.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/drop-tail-queue.h"

NS_LOG_COMPONENT_DEFINE("RateLimiterQueueDisc");

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED(RateLimiterQueueDisc);

TypeId
RateLimiterQueueDisc::GetTypeId(void)
{
  static TypeId tid = TypeId("ns3::RateLimiterQueueDisc")
    .SetParent<QueueDisc>()
    .SetGroupName("TrafficControl")
    .AddConstructor<RateLimiterQueueDisc>()
    .AddAttribute("Rate",
                  "Token refill rate",
                  DataRateValue(DataRate("50Mbps")),
                  MakeDataRateAccessor(&RateLimiterQueueDisc::m_rate),
                  MakeDataRateChecker())
    .AddAttribute("Burst",
                  "Maximum burst size in bytes",
                  UintegerValue(5000000),
                  MakeUintegerAccessor(&RateLimiterQueueDisc::m_burst),
                  MakeUintegerChecker<uint32_t>())
    .AddAttribute("Mtu",
                  "MTU for internal calculations",
                  UintegerValue(1500),
                  MakeUintegerAccessor(&RateLimiterQueueDisc::m_mtu),
                  MakeUintegerChecker<uint32_t>());
  return tid;
}

RateLimiterQueueDisc::RateLimiterQueueDisc()
  : m_btokens(0),
    m_lastReplenish(Seconds(0))
{
  NS_LOG_FUNCTION(this);
}

RateLimiterQueueDisc::~RateLimiterQueueDisc()
{
  NS_LOG_FUNCTION(this);
}

void
RateLimiterQueueDisc::SetRate(DataRate rate)
{
  NS_LOG_FUNCTION(this << rate);
  m_rate = rate;
  // Reset tokens on rate change to prevent stale state
  m_btokens = m_burst;
  m_lastReplenish = Simulator::Now();
}

DataRate
RateLimiterQueueDisc::GetRate() const
{
  return m_rate;
}

void
RateLimiterQueueDisc::SetBurst(uint32_t burst)
{
  NS_LOG_FUNCTION(this << burst);
  m_burst = burst;
}

uint32_t
RateLimiterQueueDisc::GetBurst() const
{
  return m_burst;
}

bool
RateLimiterQueueDisc::CheckConfig(void)
{
  NS_LOG_FUNCTION(this);
  if (GetNInternalQueues() == 0)
  {
    Ptr<DropTailQueue<QueueDiscItem>> queue = CreateObject<DropTailQueue<QueueDiscItem>>();
    queue->SetMaxSize(QueueSize("5MB"));
    AddInternalQueue(queue);
  }
  return true;
}

void
RateLimiterQueueDisc::InitializeParams(void)
{
  NS_LOG_FUNCTION(this);
  m_btokens = m_burst;
  m_lastReplenish = Simulator::Now();
}

void
RateLimiterQueueDisc::ReplenishTokens(void)
{
  Time now = Simulator::Now();
  Time delta = now - m_lastReplenish;
  double deltaSec = delta.GetSeconds();

  if (deltaSec > 0)
  {
    double newTokens = deltaSec * (m_rate.GetBitRate() / 8.0);
    m_btokens = std::min(m_btokens + newTokens, static_cast<double>(m_burst));
  }
  m_lastReplenish = now;
}

bool
RateLimiterQueueDisc::DoEnqueue(Ptr<QueueDiscItem> item)
{
  NS_LOG_FUNCTION(this << item);

  bool ok = GetInternalQueue(0)->Enqueue(item);
  if (!ok)
  {
    DropBeforeEnqueue(item, "Queue full");
  }
  return ok;
}

Ptr<QueueDiscItem>
RateLimiterQueueDisc::DoDequeue(void)
{
  NS_LOG_FUNCTION(this);

  // Replenish tokens based on elapsed time
  ReplenishTokens();

  // Check if we have enough tokens for the next packet
  Ptr<const QueueDiscItem> item = GetInternalQueue(0)->Peek();
  if (!item)
  {
    return 0;
  }

  uint32_t pktSize = item->GetSize();
  if (m_btokens >= pktSize)
  {
    // Enough tokens — dequeue and consume
    m_btokens -= pktSize;
    return GetInternalQueue(0)->Dequeue();
  }

  // Not enough tokens — packet stays in queue
  NS_LOG_LOGIC("Not enough tokens (" << m_btokens << " < " << pktSize << ")");
  return 0;
}

Ptr<const QueueDiscItem>
RateLimiterQueueDisc::DoPeek(void)
{
  NS_LOG_FUNCTION(this);
  return GetInternalQueue(0)->Peek();
}

} // namespace ns3
