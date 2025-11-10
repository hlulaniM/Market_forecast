import argparse
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests
from tqdm import tqdm

API_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000
INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
}


def _build_dataframe(payload: List[List]) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    frame = pd.DataFrame(payload, columns=columns)
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame = frame.drop(columns=["ignore"])
    return frame


def fetch_history(symbol: str, interval: str, start_ts: int, end_ts: int, throttle: float) -> pd.DataFrame:
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval '{interval}'.")

    step = INTERVAL_TO_MS[interval] * MAX_LIMIT
    expected_iterations = max(1, math.ceil((end_ts - start_ts) / step))
    results: List[List] = []

    progress = tqdm(total=expected_iterations, desc=f"{symbol} {interval}", unit="batch")
    cursor = start_ts

    while cursor < end_ts:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": MAX_LIMIT,
            "startTime": cursor,
        }
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        batch = response.json()

        if not batch:
            break

        results.extend(batch)
        last_open = batch[-1][0]
        cursor = last_open + INTERVAL_TO_MS[interval]
        progress.update(1)

        if len(batch) < MAX_LIMIT:
            break

        time.sleep(throttle)

    progress.close()

    if not results:
        raise RuntimeError("No data returned from Binance.")

    frame = _build_dataframe(results)
    start_dt = pd.to_datetime(start_ts, unit="ms", utc=True)
    end_dt = pd.to_datetime(end_ts, unit="ms", utc=True)
    frame = frame[(frame["open_time"] >= start_dt) & (frame["open_time"] <= end_dt)]
    frame = frame.sort_values("open_time").reset_index(drop=True)
    return frame


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect historical Binance OHLCV data into raw storage.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol, e.g., BTCUSDT")
    parser.add_argument("--interval", default="1m", help="Kline interval, e.g., 1m, 5m")
    parser.add_argument(
        "--days", type=int, default=60, help="Number of trailing days to fetch if start/end not provided."
    )
    parser.add_argument("--start", help="ISO-8601 UTC start time, e.g., 2025-09-01T00:00:00")
    parser.add_argument("--end", help="ISO-8601 UTC end time, e.g., 2025-11-08T00:00:00")
    parser.add_argument("--throttle", type=float, default=0.3, help="Sleep seconds between API requests")
    parser.add_argument(
        "--output",
        default="data/raw",
        help="Directory where the parquet dataset will be stored",
    )
    args = parser.parse_args()

    if args.start and args.end:
        start_dt = parse_datetime(args.start)
        end_dt = parse_datetime(args.end)
    else:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=args.days)

    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    df = fetch_history(args.symbol, args.interval, start_ts, end_ts, throttle=args.throttle)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{args.symbol.lower()}_{args.interval}_{start_dt:%Y%m%d}_{end_dt:%Y%m%d}.parquet"
    path = output_dir / filename
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
