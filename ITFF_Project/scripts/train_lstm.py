import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = BASE_DIR / "data" / "datasets"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "training"


def load_split(dataset_dir: Path, symbol: str, split: str, label_keys: Tuple[str, ...]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    prefix = f"{symbol.lower()}_{split}"
    X = np.load(dataset_dir / f"{prefix}_X.npy")
    labels = {key: np.load(dataset_dir / f"{prefix}_{key}.npy") for key in label_keys}
    return X, labels


def build_lstm_model(input_shape: Tuple[int, int], dropout: float = 0.3) -> Sequential:
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dropout(dropout / 2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[BinaryAccuracy(name="accuracy"), Precision(name="precision"), Recall(name="recall")],
    )
    return model


def train_model(
    dataset_dir: Path,
    symbol: str,
    label_key: str,
    epochs: int,
    batch_size: int,
    patience: int,
    output_dir: Path,
    class_weight_strategy: Optional[str],
    threshold: float,
) -> Dict[str, float]:
    label_keys = (label_key,)
    X_train, y_train_dict = load_split(dataset_dir, symbol, "train", label_keys)
    X_val, y_val_dict = load_split(dataset_dir, symbol, "val", label_keys)
    X_test, y_test_dict = load_split(dataset_dir, symbol, "test", label_keys)

    y_train = y_train_dict[label_key]
    y_val = y_val_dict[label_key]
    y_test = y_test_dict[label_key]

    model = build_lstm_model(input_shape=X_train.shape[1:])

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{symbol.lower()}_{label_key}_best.keras"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True),
    ]

    class_weight = None
    if class_weight_strategy == "balanced":
        unique, counts = np.unique(y_train, return_counts=True)
        total = y_train.shape[0]
        class_weight = {int(cls): total / (len(unique) * count) for cls, count in zip(unique, counts)}

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    results = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    prob_test = model.predict(X_test, verbose=0).flatten()
    pred_test = (prob_test >= threshold).astype(int)
    inference_sample = prob_test[:10]

    precision = precision_score(y_test, pred_test, zero_division=0)
    recall = recall_score(y_test, pred_test, zero_division=0)
    f1 = f1_score(y_test, pred_test, zero_division=0)
    auc = roc_auc_score(y_test, prob_test) if len(np.unique(y_test)) > 1 else float("nan")
    brier = float(np.mean((prob_test - y_test) ** 2))
    report = classification_report(y_test, pred_test, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, pred_test).tolist()

    metrics = {
        "history": history.history,
        "test_metrics": results,
        "probability_threshold": threshold,
        "class_weight": class_weight,
        "probabilities_sample": inference_sample.tolist(),
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "brier_score": brier,
        "classification_report": report,
        "confusion_matrix": cm,
        "checkpoint_path": str(checkpoint_path),
    }

    metrics_path = output_dir / f"{symbol.lower()}_{label_key}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline LSTM model for directional and level targets.")
    parser.add_argument("--symbol", required=True, help="Symbol key for dataset files")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Directory containing prepared numpy arrays")
    parser.add_argument("--label", choices=["direction", "level_up"], default="direction", help="Label target")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save checkpoints and metrics")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none", help="Optional class weighting strategy")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for evaluation metrics")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output)

    metrics = train_model(
        dataset_dir=dataset_dir,
        symbol=args.symbol,
        label_key=args.label,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        output_dir=output_dir,
        class_weight_strategy=None if args.class_weight == "none" else args.class_weight,
        threshold=args.threshold,
    )

    print(f"Training complete for {args.symbol} - {args.label}")
    print("Test metrics:")
    for key, value in metrics["test_metrics"].items():
        print(f"  {key}: {value:.4f}")
    print(f"  precision_score: {metrics['precision_score']:.4f}")
    print(f"  recall_score: {metrics['recall_score']:.4f}")
    print(f"  f1_score: {metrics['f1_score']:.4f}")
    print(f"  roc_auc: {metrics['roc_auc']:.4f}")
    print(f"  brier_score: {metrics['brier_score']:.4f}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
