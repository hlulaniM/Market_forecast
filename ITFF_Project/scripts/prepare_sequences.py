import argparse
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd


LABEL_SHIFT = 1
DEFAULT_SEQUENCE_LENGTH = 60
TARGET_COLUMN = "return_1m_scaled"


FEATURE_COLUMNS = [
    "open_z",
    "high_z",
    "low_z",
    "close_z",
    "volume_z",
    "quote_asset_volume_z",
    "number_of_trades_z",
    "taker_buy_base_asset_volume_z",
    "taker_buy_quote_asset_volume_z",
    "rsi_14_scaled",
    "macd_scaled",
    "macd_signal_scaled",
    "macd_hist_scaled",
    "ema_20_scaled",
    "ema_50_scaled",
    "ema_200_scaled",
    "atr_14_scaled",
    "roc_5_scaled",
    "roc_15_scaled",
    "bb_high_scaled",
    "bb_low_scaled",
    "bb_percent_scaled",
    TARGET_COLUMN,
]


LABEL_COLUMNS = {
    "direction": "direction_label",
    "level_up": "level_up_label",
}


LEVEL_THRESHOLD = 0.0015


def attach_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    shifted_return = df[TARGET_COLUMN].shift(-LABEL_SHIFT).ffill()
    future_close = df["close"].shift(-LABEL_SHIFT).ffill()
    df[LABEL_COLUMNS["direction"]] = (shifted_return > 0).astype(int)
    df[LABEL_COLUMNS["level_up"]] = ((df["close"] * (1 + LEVEL_THRESHOLD)) <= future_close).astype(int)
    return df


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    label_cols: Dict[str, str],
    seq_len: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    feature_array = df[feature_cols].to_numpy(dtype=np.float32)
    label_arrays = {name: df[col].to_numpy(dtype=np.float32) for name, col in label_cols.items()}
    num_samples = len(df) - seq_len

    X = np.stack([feature_array[i : i + seq_len] for i in range(num_samples)])
    y = {name: label_array[seq_len:] for name, label_array in label_arrays.items()}

    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    splits = {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, num_samples),
    }

    X_splits = {split: X[slice_] for split, slice_ in splits.items()}
    y_splits = {split: {name: labels[slice_] for name, labels in y.items()} for split, slice_ in splits.items()}

    return X_splits, y_splits


def save_splits(
    X_splits: Dict[str, np.ndarray],
    y_splits: Dict[str, Dict[str, np.ndarray]],
    symbol: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, X in X_splits.items():
        np.save(output_dir / f"{symbol.lower()}_{split}_X.npy", X)
        for label_name, y in y_splits[split].items():
            np.save(output_dir / f"{symbol.lower()}_{split}_{label_name}.npy", y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LSTM-ready datasets from engineered features.")
    parser.add_argument("--symbol", required=True, help="Symbol identifier")
    parser.add_argument("--features", required=True, help="Path to engineered feature parquet")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH, help="Sequence length")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--output", default="data/datasets", help="Directory for output numpy files")
    parser.add_argument("--save-metadata", action="store_true", help="Persist metadata JSON and scaler copy")
    parser.add_argument("--scaler", help="Optional scaler path to copy into dataset folder")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    df = attach_labels(df)

    X_splits, y_splits = build_sequences(
        df,
        feature_cols=FEATURE_COLUMNS,
        label_cols=LABEL_COLUMNS,
        seq_len=args.sequence_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    output_dir = Path(args.output)
    save_splits(X_splits, y_splits, args.symbol, output_dir)

    if args.save_metadata:
        metadata = {
            "symbol": args.symbol,
            "sequence_length": args.sequence_length,
            "feature_columns": FEATURE_COLUMNS,
            "label_columns": LABEL_COLUMNS,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
        }
        import json

        with open(output_dir / f"{args.symbol.lower()}_metadata.json", "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        if args.scaler:
            scaler = joblib.load(args.scaler)
            joblib.dump(scaler, output_dir / f"{args.symbol.lower()}_feature_scaler.pkl")

    print(f"Saved sequences for {args.symbol} to {output_dir}")
    for split, X in X_splits.items():
        print(
            f"  {split}: X={X.shape}, labels={{"
            + ", ".join(f"{name}:{y_splits[split][name].shape}" for name in y_splits[split])
            + "}}"
        )


if __name__ == "__main__":
    main()
