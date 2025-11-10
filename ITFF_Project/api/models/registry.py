import json
from pathlib import Path
from typing import Any, Dict

import joblib
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models" / "training"
DATASETS_DIR = BASE_DIR / "data" / "datasets"


class ModelRegistry:
    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load(self, symbol: str, target: str = "direction", model_type: str = "lstm") -> Dict[str, Any]:
        key = f"{symbol.lower()}_{target}_{model_type}"
        if key in self._cache:
            return self._cache[key]

        if model_type == "transformer":
            model_path = MODELS_DIR / f"{symbol.lower()}_{target}_transformer_best.keras"
        else:
            model_path = MODELS_DIR / f"{symbol.lower()}_{target}_best.keras"

        scaler_path = DATASETS_DIR / f"{symbol.lower()}_feature_scaler.pkl"
        metadata_path = DATASETS_DIR / f"{symbol.lower()}_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found for {symbol}-{target} ({model_type}) at {model_path}")

        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())

        bundle = {"model": model, "scaler": scaler, "metadata": metadata, "model_type": model_type}
        self._cache[key] = bundle
        return bundle


registry = ModelRegistry()
