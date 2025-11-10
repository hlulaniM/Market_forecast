import argparse
import os
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv

API_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines(symbol: str, interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    response = requests.get(API_URL, params=params, timeout=10)
    response.raise_for_status()
    dataset = response.json()

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
    frame = pd.DataFrame(dataset, columns=columns)
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


def save_dataset(df: pd.DataFrame, symbol: str, output_dir: str, fmt: str = "parquet") -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol.lower()}_{timestamp}.{fmt}"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return path


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Fetch recent Binance OHLCV data for a symbol.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol, e.g., BTCUSDT")
    parser.add_argument("--interval", default="1m", help="Kline interval (default: 1m)")
    parser.add_argument("--limit", type=int, default=1000, help="Number of klines to fetch (1-1000)")
    parser.add_argument("--output", default="data", help="Directory to save the dataset")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Output file format")
    args = parser.parse_args()

    df = fetch_klines(args.symbol, interval=args.interval, limit=args.limit)
    print(df.head())

    output_path = save_dataset(df, args.symbol, args.output, fmt=args.format)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
