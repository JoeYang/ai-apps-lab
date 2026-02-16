"""
Set up a sample SQLite database with trading data.
Run this once before using the trading-db MCP server.

Run: python 02_setup_database.py
"""

import sqlite3
import random
from datetime import datetime, timedelta

DB_PATH = "session-03/trading.db"


def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Orders table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            client_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            status TEXT NOT NULL,
            reject_reason TEXT,
            fill_price REAL,
            latency_us INTEGER
        )
    """)

    # System events table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_events (
            event_id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            system TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL
        )
    """)

    # Client positions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            client_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            net_quantity INTEGER NOT NULL,
            avg_price REAL NOT NULL,
            unrealised_pnl REAL NOT NULL,
            margin_used REAL NOT NULL,
            PRIMARY KEY (client_id, symbol)
        )
    """)

    # Generate sample data
    clients = ["HEDGE_FUND_A", "HEDGE_FUND_B", "MARKET_MAKER_1", "PROP_DESK_1", "ALGO_TRADER_1"]
    symbols = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]
    statuses = ["FILLED", "FILLED", "FILLED", "FILLED", "REJECTED"]  # 80% fill rate
    reject_reasons = ["insufficient_margin", "position_limit", "price_out_of_band", "duplicate_order_id", "stale_market_data"]

    base_time = datetime(2026, 2, 16, 9, 30, 0)

    # Generate 500 orders
    for i in range(500):
        ts = base_time + timedelta(seconds=random.randint(0, 3600))
        client = random.choice(clients)
        symbol = random.choice(symbols)
        side = random.choice(["BUY", "SELL"])
        qty = random.choice([100, 200, 500, 1000, 2000, 5000])
        price = round(random.uniform(100, 500), 2)
        status = random.choice(statuses)

        reject_reason = None
        fill_price = None
        latency = random.randint(5, 200)

        if status == "REJECTED":
            reject_reason = random.choice(reject_reasons)
            # Make HEDGE_FUND_A have more rejections
            if client == "HEDGE_FUND_A":
                reject_reason = random.choice(["insufficient_margin", "insufficient_margin", "position_limit"])
        else:
            fill_price = round(price * random.uniform(0.999, 1.001), 2)

        cursor.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (i + 1, ts.isoformat(), client, symbol, side, qty, price, status, reject_reason, fill_price, latency),
        )

    # Generate system events
    events = [
        ("09:28:00", "market-data-feed", "ERROR", "NASDAQ feed latency spike: 450ms (threshold: 10ms)"),
        ("09:28:05", "market-data-feed", "WARN", "Stale data detected for TSLA, NVDA — last update 800ms ago"),
        ("09:28:15", "matching-engine", "WARN", "CPU usage spike to 89% — garbage collection pause detected"),
        ("09:28:30", "risk-engine", "ERROR", "Bulk rejections triggered: stale market data for 12 symbols"),
        ("09:29:00", "risk-engine", "ERROR", "HEDGE_FUND_A margin utilisation at 94% — approaching critical threshold"),
        ("09:29:15", "fix-gateway", "WARN", "Sequence gap from HEDGE_FUND_A: expected 45678, received 45680"),
        ("09:29:30", "fix-gateway", "ERROR", "HEDGE_FUND_A reject rate at 23% — throttling to 50 orders/sec"),
        ("09:29:45", "risk-engine", "ERROR", "HEDGE_FUND_A buying power insufficient: need $2.3M, have $890K"),
        ("09:30:00", "market-data-feed", "INFO", "NASDAQ feed latency normalised to 8ms"),
        ("09:30:15", "matching-engine", "INFO", "CPU usage back to 67% — normal range"),
        ("09:30:30", "risk-engine", "WARN", "HEDGE_FUND_A margin still at 92% — pending position close"),
        ("09:31:00", "fix-gateway", "INFO", "HEDGE_FUND_A throttle lifted — rate restored to normal"),
    ]

    for i, (time_str, system, severity, message) in enumerate(events):
        ts = f"2026-02-16T{time_str}"
        cursor.execute(
            "INSERT INTO system_events VALUES (?, ?, ?, ?, ?)",
            (i + 1, ts, system, severity, message),
        )

    # Generate positions
    position_data = [
        ("HEDGE_FUND_A", "TSLA", 48500, 220.50, -125000, 2_130_000),
        ("HEDGE_FUND_A", "NVDA", 15000, 445.00, 89000, 1_335_000),
        ("HEDGE_FUND_A", "AAPL", 8000, 185.00, 12000, 296_000),
        ("HEDGE_FUND_B", "MSFT", 5000, 380.00, 45000, 380_000),
        ("HEDGE_FUND_B", "GOOGL", -3000, 165.00, -18000, 99_000),
        ("MARKET_MAKER_1", "TSLA", -2000, 221.00, 3000, 88_400),
        ("MARKET_MAKER_1", "AAPL", 1500, 184.50, 2250, 55_350),
        ("PROP_DESK_1", "NVDA", 10000, 448.00, -30000, 896_000),
        ("ALGO_TRADER_1", "META", 3000, 520.00, 15000, 312_000),
    ]

    for row in position_data:
        cursor.execute("INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?)", row)

    conn.commit()
    conn.close()
    print(f"Database created at {DB_PATH}")
    print(f"  - 500 orders")
    print(f"  - {len(events)} system events")
    print(f"  - {len(position_data)} positions")


if __name__ == "__main__":
    create_database()
