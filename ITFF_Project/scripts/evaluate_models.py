import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = BASE_DIR / "data" / "datasets"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "training"
DEFAULT_REPORT_DIR = BASE_DIR / "reports"


def load_dataset(dataset_dir: Path, symbol: str, split: str):
    prefix = f"{symbol.lower()}_{split}"
    X = np.load(dataset_dir / f"{prefix}_X.npy")
    y_dir = np.load(dataset_dir / f"{prefix}_direction.npy")
    y_lvl = np.load(dataset_dir / f"{prefix}_level_up.npy")
    return X, {"direction": y_dir, "level_up": y_lvl}


def load_model(model_dir: Path, symbol: str, label: str, model_type: str) -> tf.keras.Model:
    if model_type == "transformer":
        path = model_dir / f"{symbol.lower()}_{label}_transformer_best.keras"
    else:
        path = model_dir / f"{symbol.lower()}_{label}_best.keras"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return tf.keras.models.load_model(path)


def predict_with_uncertainty(model: tf.keras.Model, X: np.ndarray, model_type: str, mc_samples: int) -> np.ndarray:
    if model_type == "transformer" and mc_samples > 1:
        preds = []
        for _ in range(mc_samples):
            preds.append(model(X, training=True).numpy().flatten())
        probs = np.mean(preds, axis=0)
    else:
        probs = model.predict(X, verbose=0).flatten()
    return probs


def plot_curves(y_true: np.ndarray, probs: np.ndarray, output_dir: Path, title_prefix: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    precision, recall, _ = precision_recall_curve(y_true, probs)
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=20, strategy="uniform")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(fpr, tpr, label="ROC")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title(f"{title_prefix} ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")

    axes[1].plot(recall, precision)
    axes[1].set_title(f"{title_prefix} Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    axes[2].plot(prob_pred, prob_true, marker="o")
    axes[2].plot([0, 1], [0, 1], "--", color="gray")
    axes[2].set_title(f"{title_prefix} Calibration")
    axes[2].set_xlabel("Mean Predicted Value")
    axes[2].set_ylabel("Fraction of Positives")

    plt.tight_layout()
    roc_path = output_dir / f"{title_prefix.lower().replace(' ', '_')}_curves.png"
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix(y_true: np.ndarray, preds: np.ndarray, output_dir: Path, title_prefix: str) -> None:
    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"{title_prefix} Confusion Matrix")
    cm_path = output_dir / f"{title_prefix.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained models and generate calibration plots.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--label", choices=["direction", "level_up"], default="direction")
    parser.add_argument("--model-type", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mc-samples", type=int, default=30)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_test, labels = load_dataset(dataset_dir, args.symbol, "test")
    y_true = labels[args.label]

    model = load_model(model_dir, args.symbol, args.label, args.model_type)
    probs = predict_with_uncertainty(model, X_test, args.model_type, args.mc_samples)
    preds = (probs >= args.threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier_score": float(np.mean((probs - y_true) ** 2)),
        "classification_report": classification_report(y_true, preds, zero_division=0, output_dict=True),
        "threshold": args.threshold,
        "model_type": args.model_type,
        "mc_samples": args.mc_samples if args.model_type == "transformer" else 1,
    }

    metrics_path = output_dir / f"{args.symbol.lower()}_{args.label}_{args.model_type}_evaluation.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    title_prefix = f"{args.symbol} {args.label} {args.model_type}".upper()
    plot_curves(y_true, probs, output_dir, title_prefix)
    save_confusion_matrix(y_true, preds, output_dir, title_prefix)

    print(f"Evaluation artifacts saved to {output_dir}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
