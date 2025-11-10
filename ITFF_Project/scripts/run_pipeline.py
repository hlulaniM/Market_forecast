import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline for ITFF dataset and model refresh.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--include-sentiment", action="store_true")
    parser.add_argument("--train-direction", action="store_true")
    parser.add_argument("--train-level", action="store_true")
    parser.add_argument("--model-type", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mc-samples", type=int, default=30)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    days = str(args.days)

    run([
        "python",
        "scripts/collect_binance_data.py",
        "--symbol",
        symbol,
        "--days",
        days,
        "--output",
        "data/raw",
    ], BASE_DIR)

    raw_path = next(sorted((BASE_DIR / "data" / "raw").glob(f"{symbol.lower()}_*_clean.parquet"), reverse=True), None)
    if raw_path is None:
        raw_path = next(sorted((BASE_DIR / "data" / "raw").glob(f"{symbol.lower()}_1m_*.parquet"), reverse=True))
        run([
            "python",
            "scripts/clean_ohlcv.py",
            str(raw_path),
            "--output",
            "data/processed",
            "--normalize",
        ], BASE_DIR)
        raw_path = next(sorted((BASE_DIR / "data" / "processed").glob(f"{symbol.lower()}_*_clean.parquet"), reverse=True))

    sentiment_file = None
    if args.include_sentiment:
        run([
            "python",
            "scripts/fetch_sentiment.py",
            "--symbol",
            symbol,
            "--limit",
            "200",
            "--output",
            "data/sentiment",
        ], BASE_DIR)
        sentiment_file = next(sorted((BASE_DIR / "data" / "sentiment").glob(f"{symbol.lower()}_sentiment_*.parquet"), reverse=True))

    feature_cmd = [
        "python",
        "scripts/engineer_features.py",
        str(raw_path),
        "--symbol",
        symbol,
    ]
    if sentiment_file:
        feature_cmd += ["--sentiment-file", str(sentiment_file)]
    run(feature_cmd, BASE_DIR)

    run([
        "python",
        "scripts/prepare_sequences.py",
        "--symbol",
        symbol,
        "--features",
        f"data/features/{symbol.lower()}_features.parquet",
        "--save-metadata",
        "--scaler",
        f"models/scalers/{symbol.lower()}_feature_scaler.pkl",
        "--output",
        "data/datasets",
    ], BASE_DIR)

    if args.train_direction:
        if args.model_type == "transformer":
            run([
                "python",
                "scripts/train_transformer.py",
                "--symbol",
                symbol,
                "--label",
                "direction",
                "--mc-samples",
                str(args.mc_samples),
                "--output",
                "models/training",
            ], BASE_DIR)
        else:
            run([
                "python",
                "scripts/train_lstm.py",
                "--symbol",
                symbol,
                "--label",
                "direction",
                "--output",
                "models/training",
            ], BASE_DIR)

    if args.train_level:
        train_cmd = [
            "python",
            "scripts/train_lstm.py",
            "--symbol",
            symbol,
            "--label",
            "level_up",
            "--threshold",
            str(args.threshold),
            "--output",
            "models/training",
        ]
        if args.class_weight != "none":
            train_cmd += ["--class-weight", args.class_weight]
        run(train_cmd, BASE_DIR)


if __name__ == "__main__":
    main()
