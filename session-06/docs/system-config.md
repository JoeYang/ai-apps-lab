# Trading System Configuration Reference

## Network Architecture

The trading infrastructure uses a physically isolated network separated from the corporate environment. This separation ensures that corporate traffic (email, web browsing, file sharing) cannot interfere with latency-sensitive trading operations.

### Core Network Layout

- **Market Data Network:** Dedicated 10Gbps links from each exchange (NYSE, NASDAQ, CME, BATS). Each link has a redundant backup path. Market data arrives via UDP multicast on dedicated NICs with kernel bypass (DPDK) enabled.
- **Order Routing Network:** Separate from market data. Uses TCP with kernel bypass for sub-microsecond latency. Two redundant paths to each exchange with automatic failover. MTU set to 9000 (jumbo frames) to reduce packet overhead.
- **Internal Communication:** Between trading components (sequencer, risk engine, order router), communication uses shared memory via memory-mapped files. No TCP on the hot path — shared memory eliminates network stack overhead entirely.
- **Management Network:** Out-of-band IPMI/BMC for hardware health monitoring. Separate physical switches from trading network. Used for remote console access, firmware updates, and hardware diagnostics.

### Server Hardware

All trading servers use identical hardware for consistency:
- CPU: Intel Xeon Gold 6348 (28 cores, 2.6GHz base, 3.5GHz turbo)
- RAM: 256GB DDR4-3200 ECC (8x32GB, quad-channel)
- Storage: 2x Intel Optane P5800X 800GB (OS + logs), 4x Samsung PM1733 3.2TB NVMe (data)
- NIC: Solarflare X2522 (market data), Mellanox ConnectX-6 (order routing)
- OS: RHEL 8.8 with PREEMPT_RT kernel patch for deterministic latency

### Latency Budget

End-to-end order latency budget from market data tick to order on the wire:
- Market data receive and decode: 5μs
- Signal/strategy processing: 15μs
- Risk check (pre-trade): 3μs
- Order encoding and send: 2μs
- **Total budget: 25μs**
- Current measured p99: 22μs

---

## Risk Engine Configuration

### Default Risk Limits

These limits apply to all new client accounts. Each limit is configurable per client through the risk admin portal.

| Limit | Default Value | Description |
|-------|---------------|-------------|
| Max Position Size | $10,000,000 per symbol | Maximum notional value of position in any single symbol |
| Max Daily Loss | $500,000 per client | Maximum loss in a single trading day |
| Max Order Rate | 100 orders/second | Maximum order submission rate |
| Fat Finger Threshold | 10x average order size | Orders exceeding this multiple are flagged |
| Kill Switch | $1,000,000 daily loss | Automatic trading halt for the client |
| Max Open Orders | 5,000 per client | Maximum number of open (unfilled) orders |
| Max Notional Per Order | $5,000,000 | Maximum notional value of a single order |

### Client-Specific Overrides

Some clients have negotiated custom limits:

- **HEDGE_FUND_A:** Max position $50M, max daily loss $2M, kill switch at $5M
- **MARKET_MAKER_F:** Max order rate 1,000/sec, max open orders 50,000 (market making requires high order rates)
- **QUANT_FUND_D:** Max position $25M, fat finger threshold 50x (they trade in large blocks)
- **ALGO_TRADER_E:** Max order rate 500/sec, max open orders 10,000

### Pre-Trade Risk Checks

Every order passes through these checks in sequence (total budget: 3μs):

1. **Symbol validation** (0.1μs) — is the symbol tradeable and not halted?
2. **Position limit check** (0.5μs) — would this order push the position over the limit?
3. **Daily loss check** (0.3μs) — has the client exceeded their daily loss limit?
4. **Order rate check** (0.2μs) — is the client exceeding their order rate?
5. **Fat finger check** (0.4μs) — is the order size reasonable compared to history?
6. **Notional check** (0.2μs) — does the single order notional exceed the limit?
7. **Duplicate check** (0.3μs) — is this a duplicate of a recent order? (prevents double-sends)

---

## Client Onboarding

### Process

New client onboarding follows a standard 5-step process:

1. **Legal & Compliance** (1-2 weeks)
   - Client signs trading agreement and risk disclosure
   - KYC/AML verification completed by compliance team
   - Regulatory registrations filed as needed

2. **Account Setup** (1-2 days)
   - Create client account in the trading system
   - Set default risk limits (can be customized later)
   - Generate FIX credentials (CompID, password, certificate)

3. **Connectivity Testing** (2-3 days)
   - Client establishes FIX session to our certification environment
   - Test basic order flow: new order, cancel, replace, execution reports
   - Test market data connectivity if applicable
   - Verify message sequencing and recovery

4. **UAT Sign-off** (1 day)
   - Client runs their full test suite against certification environment
   - Both sides verify order matching, execution reports, and drop copy
   - Sign-off document signed by both parties

5. **Production Go-Live** (1 day)
   - FIX credentials activated for production
   - Start with reduced limits for first trading day
   - Monitor closely for first week
   - Full limits enabled after first week if no issues

### Recent Onboarding

- **HEDGE_FUND_G:** Onboarded 2025-10-15, specializes in equity long/short
- **RETAIL_BROKER_H:** Onboarded 2025-11-01, retail order flow aggregator
- **QUANT_FUND_I:** Onboarded 2025-12-10, high-frequency statistical arbitrage
