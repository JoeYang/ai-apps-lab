"""
Session 3, Task 4: Trading Database MCP Server
================================================
A production-style MCP server that queries a real SQLite database
of trading data. This is what an internal tool for your org could look like.

Tools:
  - query_orders: Search and filter the order book
  - query_events: Search system events by time/severity/system
  - get_client_positions: View a client's current positions
  - get_client_summary: Full client overview (orders, rejections, positions)
  - run_sql: Run arbitrary read-only SQL (for ad-hoc analysis)

Resources:
  - database schema
  - system thresholds config

Prompts:
  - investigate_client: Guided investigation of client issues
  - daily_report: Generate a daily trading summary

Setup: python 02_setup_database.py  (run once to create the database)
"""

import json
import os
import sqlite3

from mcp.server.fastmcp import FastMCP

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading.db")

mcp = FastMCP(
    name="trading-db",
    instructions="""Trading database query tools. Use these to investigate orders,
system events, and client positions. Always use the specific query tools first.
Fall back to run_sql only for complex queries that the other tools can't handle.""",
)


def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# Tools
# ============================================================

@mcp.tool()
def query_orders(
    client_id: str = "",
    symbol: str = "",
    status: str = "",
    side: str = "",
    limit: int = 20,
) -> str:
    """Query the orders table with optional filters.

    Args:
        client_id: Filter by client (e.g., 'HEDGE_FUND_A'). Empty for all.
        symbol: Filter by symbol (e.g., 'TSLA'). Empty for all.
        status: Filter by status: FILLED or REJECTED. Empty for all.
        side: Filter by side: BUY or SELL. Empty for all.
        limit: Max rows to return (default 20).
    """
    conn = get_db()
    query = "SELECT * FROM orders WHERE 1=1"
    params = []

    if client_id:
        query += " AND client_id = ?"
        params.append(client_id)
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if status:
        query += " AND status = ?"
        params.append(status)
    if side:
        query += " AND side = ?"
        params.append(side)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    result = [dict(row) for row in rows]
    return json.dumps(result, indent=2)


@mcp.tool()
def query_events(
    system: str = "",
    severity: str = "",
    limit: int = 20,
) -> str:
    """Query system events with optional filters.

    Args:
        system: Filter by system (e.g., 'risk-engine', 'fix-gateway'). Empty for all.
        severity: Minimum severity: ERROR, WARN, INFO. Empty for all.
        limit: Max rows to return (default 20).
    """
    conn = get_db()
    query = "SELECT * FROM system_events WHERE 1=1"
    params = []

    if system:
        query += " AND system = ?"
        params.append(system)
    if severity:
        severity_map = {"INFO": ("INFO", "WARN", "ERROR"), "WARN": ("WARN", "ERROR"), "ERROR": ("ERROR",)}
        levels = severity_map.get(severity, (severity,))
        query += f" AND severity IN ({','.join('?' * len(levels))})"
        params.extend(levels)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    result = [dict(row) for row in rows]
    return json.dumps(result, indent=2)


@mcp.tool()
def get_client_positions(client_id: str) -> str:
    """Get all current positions for a client.

    Args:
        client_id: Client identifier, e.g., 'HEDGE_FUND_A'
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM positions WHERE client_id = ?", (client_id,)
    ).fetchall()
    conn.close()

    if not rows:
        return json.dumps({"error": f"No positions found for {client_id}"})

    positions = [dict(row) for row in rows]
    total_margin = sum(p["margin_used"] for p in positions)
    total_pnl = sum(p["unrealised_pnl"] for p in positions)

    return json.dumps({
        "client_id": client_id,
        "positions": positions,
        "total_margin_used": total_margin,
        "total_unrealised_pnl": total_pnl,
    }, indent=2)


@mcp.tool()
def get_client_summary(client_id: str) -> str:
    """Get a comprehensive summary for a client: order stats, rejection breakdown, and positions.

    Args:
        client_id: Client identifier, e.g., 'HEDGE_FUND_A'
    """
    conn = get_db()

    # Order stats
    total_orders = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE client_id = ?", (client_id,)
    ).fetchone()[0]

    filled = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE client_id = ? AND status = 'FILLED'", (client_id,)
    ).fetchone()[0]

    rejected = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE client_id = ? AND status = 'REJECTED'", (client_id,)
    ).fetchone()[0]

    # Rejection breakdown
    reject_reasons = conn.execute(
        "SELECT reject_reason, COUNT(*) as count FROM orders WHERE client_id = ? AND status = 'REJECTED' GROUP BY reject_reason ORDER BY count DESC",
        (client_id,),
    ).fetchall()

    # Latency stats
    latency = conn.execute(
        "SELECT AVG(latency_us) as avg, MAX(latency_us) as max, MIN(latency_us) as min FROM orders WHERE client_id = ?",
        (client_id,),
    ).fetchone()

    # Positions
    positions = conn.execute(
        "SELECT * FROM positions WHERE client_id = ?", (client_id,)
    ).fetchall()

    conn.close()

    reject_rate = (rejected / total_orders * 100) if total_orders > 0 else 0

    return json.dumps({
        "client_id": client_id,
        "order_stats": {
            "total": total_orders,
            "filled": filled,
            "rejected": rejected,
            "reject_rate_pct": round(reject_rate, 1),
        },
        "rejection_breakdown": [{"reason": r[0], "count": r[1]} for r in reject_reasons],
        "latency_us": {
            "avg": round(latency[0], 1) if latency[0] else None,
            "max": latency[1],
            "min": latency[2],
        },
        "positions": [dict(p) for p in positions],
        "total_margin": sum(p["margin_used"] for p in positions),
    }, indent=2)


@mcp.tool()
def run_sql(query: str) -> str:
    """Run a read-only SQL query against the trading database.
    Only SELECT statements are allowed. Use this for complex ad-hoc analysis
    that the other tools can't handle.

    Args:
        query: SQL SELECT query to execute
    """
    if not query.strip().upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries are allowed"})

    conn = get_db()
    try:
        rows = conn.execute(query).fetchall()
        result = [dict(row) for row in rows]
        return json.dumps(result[:100], indent=2) if len(result) > 100 else json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        conn.close()


# ============================================================
# Resources
# ============================================================

@mcp.resource("schema://trading-db")
def get_schema() -> str:
    """Database schema for the trading database."""
    return """# Trading Database Schema

## orders
| Column | Type | Description |
|--------|------|-------------|
| order_id | INTEGER PK | Unique order identifier |
| timestamp | TEXT | ISO format timestamp |
| client_id | TEXT | Client identifier |
| symbol | TEXT | Ticker symbol |
| side | TEXT | BUY or SELL |
| quantity | INTEGER | Order quantity |
| price | REAL | Order price |
| status | TEXT | FILLED or REJECTED |
| reject_reason | TEXT | Null if filled, reason if rejected |
| fill_price | REAL | Actual fill price (null if rejected) |
| latency_us | INTEGER | Processing latency in microseconds |

## system_events
| Column | Type | Description |
|--------|------|-------------|
| event_id | INTEGER PK | Event identifier |
| timestamp | TEXT | ISO format timestamp |
| system | TEXT | System name |
| severity | TEXT | ERROR, WARN, INFO |
| message | TEXT | Event description |

## positions
| Column | Type | Description |
|--------|------|-------------|
| client_id | TEXT | Client identifier |
| symbol | TEXT | Ticker symbol |
| net_quantity | INTEGER | Net position (negative = short) |
| avg_price | REAL | Average entry price |
| unrealised_pnl | REAL | Unrealised P&L |
| margin_used | REAL | Margin allocated |
"""


@mcp.resource("config://thresholds")
def get_thresholds() -> str:
    """System thresholds and alerting configuration."""
    return json.dumps({
        "order_reject_rate": {"warn_pct": 5, "critical_pct": 15},
        "margin_utilisation": {"warn_pct": 85, "critical_pct": 95},
        "latency": {
            "matching_engine_p99_warn_us": 100,
            "matching_engine_p99_critical_us": 500,
            "fix_gateway_p99_warn_us": 500,
        },
        "position_limits": {
            "single_stock_max": 50000,
            "warn_pct_of_limit": 90,
        },
    }, indent=2)


# ============================================================
# Prompts
# ============================================================

@mcp.prompt()
def investigate_client(client_id: str) -> str:
    """Investigate issues for a specific trading client.

    Args:
        client_id: The client to investigate, e.g., 'HEDGE_FUND_A'
    """
    return f"""Investigate trading issues for client {client_id}.

Follow this procedure:
1. Get the client summary using get_client_summary
2. Check system events for any related errors
3. Review the thresholds configuration
4. If reject rate is high, analyse the rejection breakdown
5. Check positions for any concentration or limit issues

Provide:
- Current status assessment
- Root cause of any issues
- Recommended actions
- Risk level (low/medium/high/critical)
"""


@mcp.prompt()
def daily_report() -> str:
    """Generate a daily trading operations summary."""
    return """Generate a daily trading operations report.

Use the available tools to gather data, then produce a report covering:

1. **Order Volume Summary**: Total orders, fill rate, rejection rate by client
2. **Top Issues**: Clients with highest rejection rates and why
3. **System Health**: Any system events at WARN or ERROR level
4. **Position Concentrations**: Any clients near position limits
5. **Latency Summary**: Average and peak latencies

Format as a clear, concise report suitable for the morning trading ops standup.
Use run_sql for any aggregate queries the other tools can't handle.
"""


if __name__ == "__main__":
    mcp.run()
