import argparse
from pathlib import Path
from typing import List

import joblib
import pandas as pd
import ta
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = BASE_DIR / "data" / "features"
DEFAULT_SCALER_DIR = BASE_DIR / "models" / "scalers"

TECH_FEATURES = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "ema_20",
    "ema_50",
    "ema_200",
    "atr_14",
    "roc_5",
    "roc_15",
    "bb_high",
    "bb_low",
    "bb_percent",
]


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rsi = ta.momentum.RSIIndicator(close=df["close"], window=14)
    df["rsi_14"] = rsi.rsi()

    macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    ema20 = ta.trend.EMAIndicator(close=df["close"], window=20)
    ema50 = ta.trend.EMAIndicator(close=df["close"], window=50)
    ema200 = ta.trend.EMAIndicator(close=df["close"], window=200)
    df["ema_20"] = ema20.ema_indicator()
    df["ema_50"] = ema50.ema_indicator()
    df["ema_200"] = ema200.ema_indicator()

    atr = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr_14"] = atr.average_true_range()

    df["roc_5"] = df["close"].pct_change(periods=5)
    df["roc_15"] = df["close"].pct_change(periods=15)

    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_percent"] = bb.bollinger_pband()

    df["return_1m"] = df["close"].pct_change().fillna(0)

    return df


def attach_sentiment(df: pd.DataFrame, sentiment_path: Path) -> pd.DataFrame:
    sentiment = pd.read_parquet(sentiment_path)
    if "timestamp" not in sentiment.columns or "sentiment_score" not in sentiment.columns:
        raise ValueError("Sentiment file must include 'timestamp' and 'sentiment_score' columns.")

    sentiment["timestamp"] = pd.to_datetime(sentiment["timestamp"], utc=True)
    sentiment = sentiment.sort_values("timestamp").set_index("timestamp")

    minute_series = sentiment["sentiment_score"].resample("1min").mean().ffill()
    sentiment_conf = sentiment.get("sentiment_confidence")
    if sentiment_conf is not None:
        sentiment_conf = sentiment_conf.resample("1min").mean().ffill()

    df = df.copy()
    df = df.merge(
        minute_series.rename("sentiment_score"),
        left_on="open_time",
        right_index=True,
        how="left",
    )
    if sentiment_conf is not None:
        df = df.merge(
            sentiment_conf.rename("sentiment_confidence"),
            left_on="open_time",
            right_index=True,
            how="left",
        )
    df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
    if "sentiment_confidence" in df.columns:
        df["sentiment_confidence"] = df["sentiment_confidence"].fillna(df["sentiment_confidence"].median())
    return df


def scale_features(df: pd.DataFrame, columns: List[str]) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[columns])
    scaled_df = pd.DataFrame(scaled, columns=[f"{col}_scaled" for col in columns])
    result = pd.concat([df.reset_index(drop=True), scaled_df], axis=1)
    return result, scaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Engineer technical features and persist scalers.")
    parser.add_argument("input", help="Cleaned parquet dataset path")
    parser.add_argument("--symbol", required=True, help="Symbol identifier used for output naming")
    parser.add_argument("--output", default=str(DEFAULT_FEATURE_DIR), help="Directory for feature parquet output")
    parser.add_argument("--scaler-dir", default=str(DEFAULT_SCALER_DIR), help="Directory to store fitted scalers")
    parser.add_argument("--sentiment-file", help="Optional sentiment parquet aligned via timestamp")
    parser.add_argument("--include-sentiment-placeholder", action="store_true", help="Attach zero sentiment column when no file provided")
    args = parser.parse_args()

    input_path = Path(args.input)
    df = pd.read_parquet(input_path)

    df_features = add_technical_features(df)

    sentiment_columns: List[str] = []
    if args.sentiment_file:
        df_features = attach_sentiment(df_features, Path(args.sentiment_file))
        sentiment_columns.append("sentiment_score")
        if "sentiment_confidence" in df_features.columns:
            sentiment_columns.append("sentiment_confidence")
    elif args.include_sentiment_placeholder:
        df_features["sentiment_score"] = 0.0
        sentiment_columns.append("sentiment_score")

    df_features = df_features.dropna(subset=TECH_FEATURES).reset_index(drop=True)

    numeric_for_scaling = TECH_FEATURES + ["return_1m"] + sentiment_columns
    df_features, scaler = scale_features(df_features, numeric_for_scaling)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / f"{args.symbol.lower()}_features.parquet"
    df_features.to_parquet(feature_path, index=False)

    scaler_dir = Path(args.scaler_dir)
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = scaler_dir / f"{args.symbol.lower()}_feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    print(f"Saved engineered features to {feature_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Feature rows: {len(df_features)}")


if __name__ == "__main__":
    main()
