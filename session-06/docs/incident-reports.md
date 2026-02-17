# Incident Reports — Q4 2025

## INC-001: FIX Gateway Outage (P1)

**Date:** 2025-11-15
**Duration:** 23 minutes (09:32 — 09:55 EST)
**Affected Systems:** FIX Gateway, Order Router
**Affected Clients:** 12 (including HEDGE_FUND_A, HEDGE_FUND_C, RETAIL_GROUP_B)

### Timeline

- 09:30 — Market data spike begins: 50,000 quote updates/sec (normal: 5,000/sec)
- 09:32 — FIX gateway connection pool reaches max capacity (200 connections)
- 09:32 — First client disconnection alerts fire
- 09:35 — On-call engineer acknowledges alert, begins investigation
- 09:38 — Root cause identified: connection pool exhaustion due to market data spike
- 09:42 — Decision made to increase pool size via hot config reload
- 09:48 — Pool size increased to 500, connections begin recovering
- 09:55 — All 12 clients reconnected, order flow resumed

### Root Cause

A sudden market data spike (10x normal volume) caused the FIX gateway to open connections faster than they could be recycled. The connection pool had a hard limit of 200, which was sufficient for normal operations but not for burst scenarios. When the pool was exhausted, new client connections were rejected.

### Impact

- 12 clients unable to submit orders for 23 minutes during peak morning trading
- Estimated missed trading opportunity cost: $340,000 across affected clients
- HEDGE_FUND_A escalated to account management
- No data loss or incorrect executions

### Remediation

1. **Immediate:** Increased connection pool from 200 to 500
2. **Short-term:** Added circuit breaker with 80% pool utilization threshold (alerts at 160 connections)
3. **Long-term:** Implement dynamic pool sizing based on market data volume
4. **Monitoring:** Added dashboard for connection pool utilization, alerting at 70% and 90%

---

## INC-002: Market Data Feed Delay (P3)

**Date:** 2025-12-03
**Duration:** 5 minutes (10:15 — 10:20 EST)
**Affected Systems:** Market Data Handler
**Affected Clients:** All (degraded, not offline)

### Timeline

- 10:15 — Level 2 market data from NYSE begins showing 500ms latency spikes
- 10:16 — Automated latency alert triggers (threshold: 100ms)
- 10:17 — On-call verifies the issue is upstream (Reuters NJ datacenter)
- 10:18 — Reuters acknowledges the issue on their status page
- 10:20 — Reuters resolves the network issue, latency returns to normal

### Root Cause

Reuters experienced a network issue in their New Jersey datacenter that affected Level 2 market data distribution. The issue was entirely on the upstream provider's side. Our market data handler correctly received and processed the data, but the data itself was delayed.

### Impact

- All clients received delayed Level 2 data for 5 minutes
- 3 clients (QUANT_FUND_D, ALGO_TRADER_E, HEDGE_FUND_A) reported worse-than-expected fills on limit orders placed during the window
- Total estimated client impact: $12,000 in suboptimal fills

### Remediation

1. **Immediate:** None required — issue resolved by upstream provider
2. **Short-term:** Add staleness detection for market data (alert if last update > 200ms ago)
3. **Long-term:** Evaluate secondary market data provider for failover capability
4. **Client communication:** Sent incident report to affected clients within 2 hours

---

## INC-003: Order Rejection Spike (P4)

**Date:** 2025-12-20
**Duration:** 2 minutes (14:00 — 14:02 EST)
**Affected Systems:** Risk Engine
**Affected Clients:** 3 (HEDGE_FUND_A, RETAIL_GROUP_B, MARKET_MAKER_F)

### Timeline

- 13:55 — Routine config deployment pushed to risk engine
- 14:00 — Order rejection rate spikes to 45% (normal: <2%)
- 14:00 — Anomaly detection system triggers alert within 15 seconds
- 14:01 — On-call engineer identifies config error in deployment
- 14:02 — Config rolled back, rejection rate returns to normal

### Root Cause

A configuration deployment contained an error in the risk limits for 3 client accounts. The max position size was set to $100 instead of $100,000 (missing three zeros). This caused nearly all orders from these clients to be rejected by the pre-trade risk check.

### Impact

- 3 clients experienced 2 minutes of order rejections
- 47 orders rejected (all were valid)
- Clients were immediately notified and able to resubmit
- No financial loss (rejections, not incorrect executions)

### Remediation

1. **Immediate:** Rolled back config to previous known-good version
2. **Short-term:** Added config validation rules (position limit must be > $1,000 and < $100M)
3. **Long-term:** Implement config diff review requirement before deployment
4. **Process:** Added mandatory smoke test after risk config deployments

---

## INC-004: Sequencer Failover (P2)

**Date:** 2026-01-08
**Duration:** <1 second (automatic failover)
**Affected Systems:** Order Sequencer
**Affected Clients:** None (seamless failover)

### Timeline

- 11:45:00.000 — Primary sequencer node loses heartbeat
- 11:45:00.000 — Watchdog detects missing heartbeat
- 11:45:00.340 — Failover to secondary sequencer completes (340μs)
- 11:45:01.000 — Secondary sequencer processing orders normally
- 12:00 — Engineering team begins root cause investigation

### Root Cause

The primary sequencer node experienced a kernel page fault triggered by a memory-mapped file growing beyond the pre-allocated region. The mmap region was sized for 1 million orders per day, but the pre-market session had already allocated 80% of that by 11:45. When the region needed to grow, the kernel page fault handler stalled the process long enough to miss the heartbeat deadline (100μs).

### Impact

- No client impact — failover was seamless
- No orders lost — secondary had full replication up to the failure point
- Classified as P2 due to primary node failure, even though clients were unaffected

### Remediation

1. **Immediate:** Increased mmap pre-allocation to 2x expected daily volume
2. **Short-term:** Added monitoring for mmap region utilization (alert at 60%)
3. **Long-term:** Implement dynamic mmap growth during off-peak hours
4. **Testing:** Added load test scenario for mmap exhaustion to quarterly disaster recovery tests
