# Trading System Runbooks

## RB-001: FIX Gateway Reconnection

### When to Use
When a FIX session disconnects or a client reports they cannot submit orders.

### Steps

1. **Check gateway logs** for the disconnect reason:
   - `grep "DISCONNECT" /var/log/fix-gateway/session.log | tail -20`
   - Common reasons: network timeout, heartbeat timeout, sequence number mismatch

2. **If sequence number mismatch:**
   - Check the expected vs received sequence numbers in the log
   - Request a resend from the counterparty: `fix-admin resend-request --session <SID> --begin <SEQ>`
   - If resend fails, reset sequence numbers: `fix-admin reset-seq --session <SID>` (requires client coordination)

3. **If network timeout:**
   - Verify connectivity: `ping <exchange-endpoint>` and `traceroute <exchange-endpoint>`
   - Check for packet loss: `mtr -n --report <exchange-endpoint>`
   - If the issue is on our side, check the network switch port status
   - If the issue is on the exchange side, check their status page and open a support ticket

4. **If heartbeat timeout:**
   - Check system load: `top -bn1 | head -20`
   - High CPU can delay heartbeat responses past the 30-second timeout
   - Check if GC pauses are the cause: `grep "GC pause" /var/log/fix-gateway/gc.log`
   - If CPU is fine, check network latency to the counterparty

5. **Restart the FIX session** if recovery fails:
   - `fix-admin restart --session <SID> --reset-on-logon`
   - This resets sequence numbers (some exchanges require pre-approval)
   - Verify order state after restart: `fix-admin verify-orders --session <SID>`

### Escalation
- If the issue persists after 10 minutes, escalate to the network team
- If multiple sessions are affected, escalate to the infrastructure lead
- If client orders are stuck, notify the trading desk immediately

---

## RB-002: High Latency Investigation

### When to Use
When p99 latency exceeds the alert threshold (currently 500μs for order processing, 1ms for end-to-end).

### Steps

1. **Determine scope:**
   - Is the latency spike system-wide or isolated to specific clients/symbols?
   - Check the latency dashboard: `http://grafana:3000/d/trading-latency`
   - If isolated to one client, check their order patterns for unusual volume or size

2. **Check application-level causes:**
   - Run `perf top -p <PID>` on the order processing service to identify hot functions
   - Check for lock contention: `perf lock record -p <PID>` then `perf lock report`
   - Check thread pool utilization: `jstack <PID> | grep -c "RUNNABLE"`

3. **Check JVM causes (Java components):**
   - GC pauses: `grep "GC pause" /var/log/trading/gc.log | tail -20`
   - If GC pauses > 100μs, check heap utilization and consider tuning:
     - Young gen size: `-Xmn` should be ~40% of total heap
     - GC algorithm: ZGC for latency-sensitive paths, G1 for throughput paths
   - JIT deoptimization: check for "uncommon trap" in the JIT compilation log

4. **Check system-level causes:**
   - CPU frequency scaling: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`
   - NUMA topology: ensure the process is pinned to the correct NUMA node
   - Kernel scheduling: check for context switch storms with `vmstat 1`
   - Interrupt coalescing: `ethtool -c <NIC>` — disable for latency-sensitive NICs

5. **Check network causes:**
   - Capture packets: `tcpdump -i <NIC> -w /tmp/capture.pcap -s 128 'port 8080'`
   - Measure network RTT: `ping -c 100 <exchange-endpoint> | tail -1`
   - Check for retransmissions: `ss -ti | grep retrans`
   - Check NIC ring buffer drops: `ethtool -S <NIC> | grep drop`

### Escalation
- If the cause is external (exchange-side latency), document and notify clients
- If the cause is hardware, engage the infrastructure team for replacement
- If latency exceeds 10ms, consider activating the backup trading path

---

## RB-003: Market Data Feed Recovery

### When to Use
When market data stops updating or becomes stale for any symbol or exchange.

### Steps

1. **Check feed handler status:**
   - Process alive: `systemctl status market-data-handler`
   - Last message received: `curl http://localhost:9090/metrics | grep last_message_ts`
   - If the process is dead, restart: `systemctl restart market-data-handler`

2. **Check multicast connectivity:**
   - Verify group membership: `netstat -gn | grep <MULTICAST_GROUP>`
   - If not joined, check IGMP configuration and network switch settings
   - Test with a packet capture: `tcpdump -i <MDATA_NIC> -c 100 'udp and dst <MULTICAST_GROUP>'`

3. **Check for packet loss:**
   - NIC statistics: `ethtool -S <MDATA_NIC> | grep -E "drop|error|miss"`
   - If drops are increasing, the ring buffer may be too small: `ethtool -G <MDATA_NIC> rx 4096`
   - Kernel buffer: `sysctl net.core.rmem_max` — should be at least 16MB for market data

4. **Check data decoding:**
   - If packets arrive but aren't decoded, check the schema version
   - Exchange schema updates happen quarterly — verify our decoder matches
   - Check the decoder error log: `grep "DECODE_ERROR" /var/log/market-data/decoder.log`

5. **Failover to backup feed:**
   - If primary feed is down: `mdata-admin failover --to backup`
   - Verify backup is receiving: `curl http://localhost:9090/metrics | grep backup_feed_active`
   - Notify clients that backup feed may have slightly different characteristics (latency, depth)

6. **Post-recovery validation:**
   - Compare our top-of-book prices with a reference source
   - Check for any stale data that was served during the outage
   - If stale data was served, notify affected clients with timestamps

### Escalation
- If both primary and backup feeds are down, this is a P1 — escalate to infrastructure and trading desk
- If the issue is exchange-side, monitor their status page and open a support ticket
- If stale data caused incorrect executions, escalate to compliance

---

## RB-004: Risk Engine Alert Handling

### When to Use
When the risk engine triggers a limit breach, kill switch activation, or anomalous trading pattern.

### Steps

1. **Identify the alert type:**
   - Position limit breach: client exceeded max position size ($10M default)
   - Daily loss limit: client exceeded max daily loss ($500K default)
   - Order rate limit: client exceeding 100 orders/second
   - Fat finger alert: order size > 10x client's average
   - Kill switch: daily loss exceeded $1M — all trading halted for the client

2. **For position limit breach:**
   - Check current position: `risk-admin show-position --client <CID> --symbol <SYM>`
   - Verify the limit is correct: `risk-admin show-limits --client <CID>`
   - If the limit is wrong, update via: `risk-admin update-limit --client <CID> --field max_position --value <NEW>`
   - If the limit is correct and the client is over, notify the trading desk

3. **For kill switch activation:**
   - This is critical — the client's trading is halted
   - Verify the loss calculation: `risk-admin show-pnl --client <CID> --date today`
   - If the loss is real, contact the client's risk manager
   - If the loss is a calculation error (e.g., bad market data), override: `risk-admin override-killswitch --client <CID> --reason "bad mdata" --approver <YOUR_ID>`
   - Kill switch overrides require approval from a senior risk officer

4. **For fat finger alerts:**
   - Check the flagged order: `risk-admin show-order --order-id <OID>`
   - Compare with client's historical average: `risk-admin show-history --client <CID> --symbol <SYM>`
   - If it's a genuine fat finger, the order was already blocked — notify the client
   - If it's intentional (large block trade), add a temporary exception: `risk-admin add-exception --client <CID> --order-id <OID> --reason "block trade"`

### Escalation
- Kill switch activations always escalate to the head of risk
- Multiple clients hitting limits simultaneously may indicate a market event — notify the trading desk
- Any suspected market manipulation patterns escalate to compliance immediately
