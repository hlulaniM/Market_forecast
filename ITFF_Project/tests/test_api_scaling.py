from typing import Any, Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from api.main import app


class DummyModel:
    def __init__(self, response: float = 0.8) -> None:
        self.response = response
        self.last_input: np.ndarray | None = None

    def predict(self, sequence: np.ndarray, verbose: int = 0) -> np.ndarray:
        self.last_input = sequence
        return np.array([[self.response]], dtype=np.float32)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def build_bundle(scaler: StandardScaler, model: DummyModel) -> Dict[str, Any]:
    return {
        "model": model,
        "scaler": scaler,
        "metadata": {"sequence_length": 2, "feature_columns": ["feat_a", "feat_b"]},
        "model_type": "lstm",
    }


def test_sequence_is_scaled(monkeypatch, client: TestClient) -> None:
    raw_sequence = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32))
    expected_scaled = scaler.transform(raw_sequence)

    model = DummyModel()
    bundle = build_bundle(scaler, model)

    monkeypatch.setattr("api.routers.predict.registry.load", lambda *args, **kwargs: bundle)

    payload = {
        "symbol": "BTCUSDT",
        "target": "direction",
        "sequence": raw_sequence.tolist(),
        "threshold": 0.5,
        "model_type": "lstm",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    assert model.last_input is not None
    np.testing.assert_allclose(model.last_input[0], expected_scaled.astype(np.float32))


def test_missing_scaler_falls_back(monkeypatch, client: TestClient) -> None:
    raw_sequence = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    model = DummyModel()
    bundle = {
        "model": model,
        "scaler": None,
        "metadata": {"sequence_length": 2, "feature_columns": ["feat_a", "feat_b"]},
        "model_type": "lstm",
    }

    monkeypatch.setattr("api.routers.predict.registry.load", lambda *args, **kwargs: bundle)

    payload = {
        "symbol": "BTCUSDT",
        "target": "direction",
        "sequence": raw_sequence.tolist(),
        "threshold": 0.5,
        "model_type": "lstm",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert model.last_input is not None
    np.testing.assert_allclose(model.last_input[0], raw_sequence)

