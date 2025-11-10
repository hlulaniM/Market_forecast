import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class CleanStats:
    duplicates_removed: int
    missing_timestamps: int
    remaining_nans: int


PRICE_COLS = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "open_time" not in df.columns:
        raise ValueError("Dataset must include an 'open_time' column.")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df


def clean_dataset(df: pd.DataFrame, freq: str = "1min") -> tuple[pd.DataFrame, CleanStats]:
    df = df.copy()
    duplicates = df.duplicated(subset="open_time").sum()
    df = df.drop_duplicates(subset="open_time")
    df = df.sort_values("open_time").set_index("open_time")

    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, tz="UTC")
    df = df.reindex(expected_index)
    missing = df[PRICE_COLS + VOLUME_COLS].isna().any(axis=1).sum()

    df[PRICE_COLS] = df[PRICE_COLS].ffill()
    df[VOLUME_COLS] = df[VOLUME_COLS].fillna(0)

    remaining_nans = df.isna().sum().sum()
    stats = CleanStats(duplicates_removed=int(duplicates), missing_timestamps=int(missing), remaining_nans=int(remaining_nans))

    df = df.reset_index().rename(columns={"index": "open_time"})
    return df, stats


def normalize_numeric(df: pd.DataFrame, columns: List[str]) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[columns])
    scaled_df = pd.DataFrame(scaled_values, columns=[f"{col}_z" for col in columns])
    result = pd.concat([df.reset_index(drop=True), scaled_df], axis=1)
    return result, scaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw Binance OHLCV datasets and store processed outputs.")
    parser.add_argument("input", help="Path to the raw parquet dataset")
    parser.add_argument("--output", default="data/processed", help="Directory to store cleaned parquet")
    parser.add_argument("--freq", default="1min", help="Expected frequency for reindexing, default 1min")
    parser.add_argument("--normalize", action="store_true", help="Add z-score normalized columns for numeric features")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_dataset(input_path)
    cleaned_df, stats = clean_dataset(df_raw, freq=args.freq)

    if args.normalize:
        numeric_cols = PRICE_COLS + VOLUME_COLS
        cleaned_df, _ = normalize_numeric(cleaned_df, numeric_cols)

    output_path = output_dir / f"{input_path.stem.replace('_1m', '')}_clean.parquet"
    cleaned_df.to_parquet(output_path, index=False)

    print(f"Saved cleaned dataset to {output_path}")
    print(
        "Summary: duplicates_removed={stats.duplicates_removed}, missing_timestamps={stats.missing_timestamps}, remaining_nans={stats.remaining_nans}".format(
            stats=stats
        )
    )


if __name__ == "__main__":
    main()
