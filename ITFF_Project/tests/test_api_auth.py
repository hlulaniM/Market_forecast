import importlib
from typing import Any, Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import config


def _dummy_bundle() -> Dict[str, Any]:
    class DummyModel:
        def predict(self, sequence, verbose=0):
            return np.array([[0.7]], dtype=np.float32)

        def __call__(self, *args, **kwargs):
            return self.predict(*args, **kwargs)

    return {
        "model": DummyModel(),
        "scaler": None,
        "metadata": {"sequence_length": 2, "feature_columns": ["feat_a", "feat_b"]},
        "model_type": "lstm",
    }


def create_client(monkeypatch, token: str, rate_limit: str | None = "5/minute") -> TestClient:
    monkeypatch.setenv("ITFF_API_TOKEN", token)
    if rate_limit is not None:
        monkeypatch.setenv("ITFF_API_RATE_LIMIT", rate_limit)
    else:
        monkeypatch.delenv("ITFF_API_RATE_LIMIT", raising=False)

    config.reset_settings_cache()
    import api.main as main

    importlib.reload(main)

    monkeypatch.setattr("api.routers.predict.registry.load", lambda *_, **__: _dummy_bundle())

    return TestClient(main.app)


def _payload():
    return {
        "symbol": "BTCUSDT",
        "target": "direction",
        "sequence": [[0.1, 0.2], [0.3, 0.4]],
        "threshold": 0.5,
        "model_type": "lstm",
    }


def test_missing_token_rejected(monkeypatch):
    client = create_client(monkeypatch, token="secret")

    response = client.post("/predict", json=_payload())
    assert response.status_code == 401


def test_valid_token_allows_request(monkeypatch):
    client = create_client(monkeypatch, token="secret")

    response = client.post("/predict", json=_payload(), headers={"X-API-Token": "secret"})
    assert response.status_code == 200


def test_rate_limit_enforced(monkeypatch):
    client = create_client(monkeypatch, token="secret", rate_limit="2/minute")
    headers = {"X-API-Token": "secret"}

    first = client.post("/predict", json=_payload(), headers=headers)
    second = client.post("/predict", json=_payload(), headers=headers)
    third = client.post("/predict", json=_payload(), headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429

