"""
Session 3, Task 2: Your First MCP Server
==========================================
A simple MCP server that exposes tools for trading system operations.

This is the same functionality from Session 2's tool_use.py, but now
exposed as a standard MCP server that ANY MCP client can connect to.

To test it standalone:
  python 01_first_mcp_server.py

To connect from Claude Code:
  Add to ~/.claude/claude_code_config.json (see Task 3)
"""

import json
import math
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP(
    name="trading-ops",
    instructions="Trading system operations tools. Use these to check system health, search logs, and perform calculations.",
)


# ============================================================
# Tools — functions the LLM can call
# ============================================================
# Compare this to Session 2: instead of defining JSON schemas manually,
# the MCP SDK generates them from your Python type hints and docstrings.

@mcp.tool()
def get_system_metrics(system_name: str, metric_type: str) -> str:
    """Get current metrics for a trading system component.

    Args:
        system_name: System to query. One of: matching-engine, fix-gateway, market-data-feed, risk-engine
        metric_type: Metric type. One of: cpu, memory, latency, throughput, all
    """
    metrics = {
        "matching-engine": {
            "cpu": {"current": 67, "avg_1h": 52, "peak_1h": 89, "unit": "%"},
            "memory": {"used_gb": 28.4, "total_gb": 64, "unit": "GB"},
            "latency": {"p50_us": 12, "p99_us": 45, "p999_us": 120, "unit": "microseconds"},
            "throughput": {"orders_per_sec": 45000, "peak_1h": 78000, "unit": "orders/sec"},
        },
        "fix-gateway": {
            "cpu": {"current": 34, "avg_1h": 28, "peak_1h": 55, "unit": "%"},
            "memory": {"used_gb": 8.2, "total_gb": 32, "unit": "GB"},
            "latency": {"p50_us": 85, "p99_us": 340, "p999_us": 1200, "unit": "microseconds"},
            "throughput": {"messages_per_sec": 12000, "peak_1h": 25000, "unit": "msgs/sec"},
        },
        "market-data-feed": {
            "cpu": {"current": 78, "avg_1h": 65, "peak_1h": 95, "unit": "%"},
            "memory": {"used_gb": 45.1, "total_gb": 128, "unit": "GB"},
            "latency": {"p50_us": 5, "p99_us": 22, "p999_us": 88, "unit": "microseconds"},
            "throughput": {"updates_per_sec": 2500000, "peak_1h": 4200000, "unit": "updates/sec"},
        },
        "risk-engine": {
            "cpu": {"current": 45, "avg_1h": 38, "peak_1h": 72, "unit": "%"},
            "memory": {"used_gb": 16.8, "total_gb": 64, "unit": "GB"},
            "latency": {"p50_us": 150, "p99_us": 800, "p999_us": 3500, "unit": "microseconds"},
            "throughput": {"checks_per_sec": 30000, "peak_1h": 55000, "unit": "checks/sec"},
        },
    }

    if system_name not in metrics:
        return json.dumps({"error": f"Unknown system: {system_name}", "available": list(metrics.keys())})

    if metric_type == "all":
        return json.dumps(metrics[system_name], indent=2)
    elif metric_type in metrics[system_name]:
        return json.dumps(metrics[system_name][metric_type], indent=2)
    else:
        return json.dumps({"error": f"Unknown metric: {metric_type}", "available": ["cpu", "memory", "latency", "throughput", "all"]})


@mcp.tool()
def search_logs(system_name: str, severity: str, time_range_minutes: int = 30) -> str:
    """Search recent log entries for a trading system component.

    Args:
        system_name: System to search. One of: matching-engine, fix-gateway, risk-engine
        severity: Minimum severity. One of: ERROR, WARN, INFO, DEBUG
        time_range_minutes: How many minutes back to search (default: 30)
    """
    logs = {
        "matching-engine": [
            {"time": "09:29:58.123", "severity": "WARN", "msg": "Order book depth below threshold for TSLA: 3 levels (min: 5)"},
            {"time": "09:30:01.456", "severity": "ERROR", "msg": "Reject: order 12345 price 185.50 outside circuit breaker band [180.00-190.00] for AAPL"},
            {"time": "09:30:02.789", "severity": "ERROR", "msg": "Reject: duplicate client order ID 'HFUND-98765' from session HEDGE_FUND_A"},
            {"time": "09:30:05.012", "severity": "WARN", "msg": "Matching latency spike: p99=320us (threshold: 100us) during NVDA auction"},
        ],
        "fix-gateway": [
            {"time": "09:28:12.345", "severity": "WARN", "msg": "Sequence gap detected: expected 45678, received 45680 from HEDGE_FUND_A"},
            {"time": "09:29:45.678", "severity": "ERROR", "msg": "Session HEDGE_FUND_A: 142 rejects in last 60s (threshold: 30)"},
            {"time": "09:30:01.234", "severity": "INFO", "msg": "Throttling HEDGE_FUND_A: rate reduced to 50 orders/sec"},
        ],
        "risk-engine": [
            {"time": "09:29:30.111", "severity": "WARN", "msg": "HEDGE_FUND_A margin utilisation at 94% (warning threshold: 90%)"},
            {"time": "09:29:55.222", "severity": "ERROR", "msg": "HEDGE_FUND_A buying power insufficient: required $2.3M, available $890K"},
            {"time": "09:30:00.333", "severity": "ERROR", "msg": "Position limit approaching: HEDGE_FUND_A TSLA position 48,500/50,000 (97%)"},
        ],
    }

    if system_name not in logs:
        return json.dumps({"error": f"Unknown system: {system_name}", "available": list(logs.keys())})

    severity_order = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
    min_level = severity_order.get(severity, 0)
    filtered = [entry for entry in logs.get(system_name, []) if severity_order.get(entry["severity"], 0) >= min_level]

    return json.dumps(filtered, indent=2)


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: Math expression, e.g., '(100 * 0.23) / 4.5' or '28.4 / 64 * 100'
    """
    try:
        allowed_names = {"sqrt": math.sqrt, "pow": pow, "abs": abs, "round": round}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# Resources — data the LLM can read
# ============================================================
# Resources are like files or documents the LLM can access.
# Unlike tools (which take arguments), resources are identified by URI.

@mcp.resource("config://trading-systems")
def get_system_config() -> str:
    """Current trading system configuration and thresholds."""
    config = {
        "systems": {
            "matching-engine": {
                "cpu_warn_threshold": 80,
                "cpu_critical_threshold": 95,
                "latency_p99_warn_us": 100,
                "latency_p99_critical_us": 500,
            },
            "fix-gateway": {
                "max_reject_rate_pct": 5,
                "heartbeat_timeout_sec": 30,
                "max_messages_per_sec": 50000,
            },
            "risk-engine": {
                "margin_warn_threshold_pct": 90,
                "margin_critical_threshold_pct": 95,
                "position_limit_warn_pct": 90,
            },
        },
        "escalation_policy": {
            "critical": "Page on-call immediately via PagerDuty",
            "warning": "Notify #trading-ops Slack channel",
            "info": "Log only, no notification",
        },
    }
    return json.dumps(config, indent=2)


@mcp.resource("runbook://fix-reconnection")
def get_fix_reconnection_runbook() -> str:
    """Standard runbook for FIX session reconnection procedures."""
    return """# FIX Session Reconnection Runbook

## Symptoms
- Heartbeat timeout alerts
- Sequence gap detection
- Client reporting connectivity issues

## Steps
1. Check FIX gateway logs for the affected session
2. Verify network connectivity to the client (ping, traceroute)
3. Check if the sequence numbers are in sync
4. If sequence gap: initiate ResendRequest for missing messages
5. If heartbeat timeout: send TestRequest to verify session liveness
6. If session is down: initiate logout/login sequence
7. Verify order state reconciliation after reconnection
8. Notify client of reconnection status

## Escalation
- If reconnection fails after 3 attempts: escalate to network team
- If data loss suspected: escalate to risk team immediately
"""


# ============================================================
# Prompts — reusable prompt templates
# ============================================================
# Prompts are pre-built workflows that clients can offer to users.

@mcp.prompt()
def investigate_incident(system_name: str, description: str) -> str:
    """Investigate a trading system incident step by step.

    Args:
        system_name: The affected system
        description: Brief description of the incident
    """
    return f"""You are investigating an incident in the {system_name} system.

Description: {description}

Please follow this investigation procedure:
1. First, check the current metrics for {system_name} using get_system_metrics
2. Search the logs for {system_name} at ERROR and WARN levels
3. Read the system configuration to check threshold values
4. Analyse the data and provide:
   - Root cause assessment
   - Current impact
   - Recommended immediate actions
   - Whether escalation is needed (and to whom)
"""


# ============================================================
# Run the server
# ============================================================
if __name__ == "__main__":
    mcp.run()
