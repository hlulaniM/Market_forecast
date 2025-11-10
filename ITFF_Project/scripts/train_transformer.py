import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Dense,
    GlobalAveragePooling1D,
    Add,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = BASE_DIR / "data" / "datasets"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "training"


def load_split(dataset_dir: Path, symbol: str, split: str, label_keys: Tuple[str, ...]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    prefix = f"{symbol.lower()}_{split}"
    X = np.load(dataset_dir / f"{prefix}_X.npy")
    labels = {key: np.load(dataset_dir / f"{prefix}_{key}.npy") for key in label_keys}
    return X, labels


def positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    angles = tf.range(seq_len, dtype=tf.float32)[:, None] / tf.pow(10000.0, (2 * (tf.range(d_model, dtype=tf.float32) // 2)) / d_model)
    angles = tf.where(tf.cast(tf.range(d_model)[None, :] % 2, tf.bool), tf.cos(angles), tf.sin(angles))
    return angles


def transformer_encoder(inputs: tf.Tensor, num_heads: int, key_dim: int, ff_dim: int, dropout: float) -> tf.Tensor:
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(inputs, inputs, training=True)
    attn_output = Dropout(dropout)(attn_output, training=True)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ff_output = Dense(ff_dim, activation="relu")(out1)
    ff_output = Dropout(dropout)(ff_output, training=True)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output, training=True)
    return LayerNormalization(epsilon=1e-6)(out1 + ff_output)


def build_transformer_model(sequence_length: int, feature_dim: int, dropout: float = 0.3) -> Model:
    inputs = Input(shape=(sequence_length, feature_dim))
    positions = positional_encoding(sequence_length, feature_dim)
    x = inputs + positions
    x = transformer_encoder(x, num_heads=4, key_dim=feature_dim, ff_dim=128, dropout=dropout)
    x = transformer_encoder(x, num_heads=4, key_dim=feature_dim, ff_dim=128, dropout=dropout)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x, training=True)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout)(x, training=True)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def train_model(
    dataset_dir: Path,
    symbol: str,
    label_key: str,
    epochs: int,
    batch_size: int,
    patience: int,
    dropout: float,
    output_dir: Path,
    mc_samples: int,
) -> Dict[str, float]:
    label_keys = (label_key,)
    X_train, y_train_dict = load_split(dataset_dir, symbol, "train", label_keys)
    X_val, y_val_dict = load_split(dataset_dir, symbol, "val", label_keys)
    X_test, y_test_dict = load_split(dataset_dir, symbol, "test", label_keys)

    y_train = y_train_dict[label_key]
    y_val = y_val_dict[label_key]
    y_test = y_test_dict[label_key]

    model = build_transformer_model(sequence_length=X_train.shape[1], feature_dim=X_train.shape[2], dropout=dropout)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{symbol.lower()}_{label_key}_transformer_best.keras"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    results = model.evaluate(X_test, y_test, verbose=0, return_dict=True)

    # Monte Carlo dropout predictions
    preds = []
    for _ in range(mc_samples):
        preds.append(model(X_test, training=True).numpy().flatten())
    preds = np.stack(preds)
    mean_probs = preds.mean(axis=0)
    std_probs = preds.std(axis=0)

    metrics = {
        "history": history.history,
        "test_metrics": results,
        "brier_score": brier_score(y_test, mean_probs),
        "mc_mean_samples": float(mean_probs.mean()),
        "mc_mean_std": float(std_probs.mean()),
        "sample_predictions": mean_probs[:10].tolist(),
        "sample_uncertainty": std_probs[:10].tolist(),
        "checkpoint_path": str(checkpoint_path),
    }

    metrics_path = output_dir / f"{symbol.lower()}_{label_key}_transformer_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer model with MC Dropout for ITFF datasets.")
    parser.add_argument("--symbol", required=True, help="Symbol key for dataset files")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--label", choices=["direction", "level_up"], default="direction")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    metrics = train_model(
        dataset_dir=Path(args.dataset_dir),
        symbol=args.symbol,
        label_key=args.label,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        dropout=args.dropout,
        output_dir=Path(args.output),
        mc_samples=args.mc_samples,
    )

    print(f"Transformer training complete for {args.symbol} - {args.label}")
    print("Test metrics:")
    for key, value in metrics["test_metrics"].items():
        print(f"  {key}: {value:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    print(f"MC mean std: {metrics['mc_mean_std']:.4f}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
