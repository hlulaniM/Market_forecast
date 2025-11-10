import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = BASE_DIR / "data" / "datasets"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "training"
DEFAULT_OUTPUT_DIR = BASE_DIR / "reports"


def load_data(dataset_dir: Path, symbol: str, label: str):
    prefix = f"{symbol.lower()}_test"
    X = np.load(dataset_dir / f"{prefix}_X.npy")
    y = np.load(dataset_dir / f"{prefix}_{label}.npy")
    return X, y


def load_model(model_dir: Path, symbol: str, label: str, model_type: str) -> tf.keras.Model:
    if model_type == "transformer":
        path = model_dir / f"{symbol.lower()}_{label}_transformer_best.keras"
    else:
        path = model_dir / f"{symbol.lower()}_{label}_best.keras"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint {path}")
    return tf.keras.models.load_model(path)


def predict(model: tf.keras.Model, X: np.ndarray, model_type: str, mc_samples: int) -> np.ndarray:
    if model_type == "transformer" and mc_samples > 1:
        preds = [model(X, training=True).numpy().flatten() for _ in range(mc_samples)]
        return np.mean(preds, axis=0)
    return model.predict(X, verbose=0).flatten()


def sweep_thresholds(y_true: np.ndarray, probs: np.ndarray, steps: int) -> list[dict]:
    thresholds = np.linspace(0.1, 0.9, steps)
    rows = []
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    for th in thresholds:
        preds = (probs >= th).astype(int)
        rows.append(
            {
                "threshold": round(float(th), 4),
                "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
                "auc": round(float(auc), 4),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold sweep evaluation for ITFF models")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--label", choices=["direction", "level_up"], default="direction")
    parser.add_argument("--model-type", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--steps", type=int, default=17)
    parser.add_argument("--mc-samples", type=int, default=30)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(dataset_dir, args.symbol, args.label)
    model = load_model(model_dir, args.symbol, args.label, args.model_type)
    probs = predict(model, X, args.model_type, args.mc_samples)

    rows = sweep_thresholds(y, probs, args.steps)
    out_path = output_dir / f"{args.symbol.lower()}_{args.label}_{args.model_type}_thresholds.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["threshold", "precision", "recall", "f1", "auc"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote threshold analysis to {out_path}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
